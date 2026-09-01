"""Plant AI health and stress monitoring module using Google Coral USB Edge TPU and multi-spectral vegetative indices."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import time
import cv2
import numpy as np

try:
    import ai_edge_litert.interpreter as tflite
except ImportError:
    tflite = None


@dataclass
class PlantHealthResult:
    """Detection and canopy health analysis result for a plant region."""
    bbox: tuple[int, int, int, int]  # (x, y, w, h) in pixel coordinates
    label: str                       # e.g., "Potted Plant"
    confidence: float                # 0.0 to 1.0
    status: str                      # "HEALTHY", "YELLOWING WARNING", "BROWNING ALERT"
    healthy_pct: float               # Percentage of green foliage
    chlorosis_pct: float             # Percentage of yellowing foliage
    necrosis_pct: float              # Percentage of browning foliage
    mean_hue: float                  # Mean Hue value (0-180 in OpenCV)
    color_bgr: tuple[int, int, int]  # Visual status color (BGR)


def analyze_crop_health(crop: np.ndarray) -> tuple[str, float, float, float, float, tuple[int, int, int]]:
    """Analyze plant canopy health using pure Hue (HSV) color space for green, yellowing, and browning.
    
    Returns:
        tuple of (status_text, healthy_pct, yellowing_pct, browning_pct, mean_hue, color_bgr)
    """
    if crop is None or crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
        return "NO VEGETATION", 0.0, 0.0, 0.0, 0.0, (128, 128, 128)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0]  # OpenCV Hue: 0 - 180 (corresponds to 0° - 360°)
    s = hsv[..., 1]  # Saturation: 0 - 255
    v = hsv[..., 2]  # Value / Brightness: 0 - 255

    # 1. Healthy Green Foliage: Hue in [32, 88] (64° - 176°)
    healthy_mask = (s >= 35) & (v >= 35) & (h >= 32) & (h <= 88)

    # 2. Yellowing Foliage (Chlorosis / Nutrient stress): Hue in [18, 31] (36° - 62°)
    yellow_mask = (s >= 35) & (v >= 35) & (h >= 18) & (h < 32)

    # 3. Browning / Dried Foliage (Necrosis / Drying stress): Hue in [0, 17] or [165, 180] with S >= 50
    brown_mask = (s >= 50) & (v >= 35) & ((h < 18) | (h >= 165))

    foliage_mask = healthy_mask | yellow_mask | brown_mask
    n_healthy = int(np.count_nonzero(healthy_mask))
    n_yellow = int(np.count_nonzero(yellow_mask))
    n_brown = int(np.count_nonzero(brown_mask))
    n_total = n_healthy + n_yellow + n_brown

    mean_hue = float(np.mean(h[foliage_mask])) if np.any(foliage_mask) else 0.0

    # If no significant foliage pixels are found, reject as non-plant
    if n_total < 30:
        return "NO VEGETATION", 0.0, 0.0, 0.0, mean_hue, (128, 128, 128)

    healthy_pct = (n_healthy / n_total) * 100.0
    yellowing_pct = (n_yellow / n_total) * 100.0
    browning_pct = (n_brown / n_total) * 100.0

    # Decision Matrix with user-friendly terminology (Healthy, Yellowing, Browning)
    if browning_pct >= 20.0 or (browning_pct >= 10.0 and yellowing_pct >= 30.0):
        status = "BROWNING ALERT"
        color = (0, 0, 230)  # Red
    elif yellowing_pct >= 18.0:
        status = "YELLOWING WARNING"
        color = (0, 180, 255)  # Amber
    else:
        status = "HEALTHY"
        color = (30, 210, 30)  # Green

    return status, healthy_pct, yellowing_pct, browning_pct, mean_hue, color


class PlantAIDetector:
    """2-Stage Plant AI detector combining Edge TPU plant localization with canopy stress profiling."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        labels_path: Path | str | None = None,
        confidence_threshold: float = 0.45,
        target_classes: list[int] | None = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        # COCO class 63 = potted plant (0-indexed in coco_labels.txt is line 64 / index 63 or 58)
        self.target_classes = target_classes or [58, 63, 64]
        self.is_available = False
        self.labels: dict[int, str] = {}
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.last_latency_ms = 0.0

        base_dir = Path(__file__).resolve().parent.parent
        self.model_path = Path(model_path) if model_path else base_dir / "models" / "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
        self.labels_path = Path(labels_path) if labels_path else base_dir / "models" / "coco_labels.txt"

        self._load_labels()
        self._init_edgetpu()

    def _load_labels(self) -> None:
        """Load COCO label mappings."""
        if self.labels_path.is_file():
            try:
                with open(self.labels_path, "r", encoding="utf-8") as f:
                    for idx, line in enumerate(f):
                        name = line.strip()
                        if name and name != "n/a":
                            self.labels[idx] = name
            except Exception as e:
                print(f"[PlantAI] Warning: could not load labels from {self.labels_path}: {e}")

    def _init_edgetpu(self) -> bool:
        """Attempt to initialize LiteRT with libedgetpu.so.1 delegate."""
        if tflite is None or not self.model_path.is_file():
            self.is_available = False
            return False

        try:
            delegate = tflite.load_delegate("libedgetpu.so.1")
            self.interpreter = tflite.Interpreter(
                model_path=str(self.model_path),
                experimental_delegates=[delegate],
            )
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.is_available = True
            print(f"[PlantAI] Coral Edge TPU initialized successfully with {self.model_path.name}")
            return True
        except Exception as e:
            self.is_available = False
            self.interpreter = None
            print(f"[PlantAI] Coral Edge TPU unavailable ({e}). Plant AI will remain inactive.")
            return False

    def detect_and_analyze(self, frame: np.ndarray, min_score: float | None = None) -> tuple[list[PlantHealthResult], float]:
        """Execute Stage 1 localization and Stage 2 canopy stress profiling.
        
        Returns:
            tuple of (results_list, latency_ms)
        """
        if frame is None or frame.size == 0:
            return [], 0.0

        threshold = min_score if min_score is not None else self.confidence_threshold
        h, w = frame.shape[:2]
        results: list[PlantHealthResult] = []
        latency_ms = 0.0

        if self.is_available and self.interpreter is not None:
            try:
                start_time = time.perf_counter()
                
                # Model input shape is typically (1, 300, 300, 3) uint8 RGB
                in_shape = self.input_details[0]["shape"]
                target_h, target_w = in_shape[1], in_shape[2]

                # Convert BGR -> RGB and resize
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_tensor = cv2.resize(rgb, (target_w, target_h))
                input_tensor = np.expand_dims(input_tensor, axis=0)

                self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)
                self.interpreter.invoke()

                # TFLite postprocessed SSD outputs:
                # Output 0: boxes [1, N, 4] -> [ymin, xmin, ymax, xmax] (normalized 0..1)
                # Output 1: classes [1, N]
                # Output 2: scores [1, N]
                # Output 3: count [1]
                boxes = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
                classes = self.interpreter.get_tensor(self.output_details[1]["index"])[0]
                scores = self.interpreter.get_tensor(self.output_details[2]["index"])[0]
                count = int(self.interpreter.get_tensor(self.output_details[3]["index"])[0])

                latency_ms = (time.perf_counter() - start_time) * 1000.0
                self.last_latency_ms = latency_ms

                for i in range(min(count, len(scores))):
                    score = float(scores[i])
                    if score < threshold:
                        continue

                    class_id = int(classes[i])
                    label_name = self.labels.get(class_id, f"class_{class_id}")

                    is_plant = (
                        class_id in self.target_classes
                        or "plant" in label_name.lower()
                        or "flower" in label_name.lower()
                        or "pot" in label_name.lower()
                        or "vase" in label_name.lower()
                    )

                    # Strictly filter for plant-related objects
                    if not is_plant:
                        continue  # Discard non-plant objects (desks, wires, chairs, keyboards, etc.)

                    ymin, xmin, ymax, xmax = boxes[i]
                    bx = max(0, int(xmin * w))
                    by = max(0, int(ymin * h))
                    bw = min(w - bx, int((xmax - xmin) * w))
                    bh = min(h - by, int((ymax - ymin) * h))

                    if bw < 15 or bh < 15:
                        continue

                    crop = frame[by : by + bh, bx : bx + bw]
                    status, healthy_pct, chlorosis_pct, necrosis_pct, mean_hue, color_bgr = analyze_crop_health(crop)

                    # If no vegetation is found inside the detected object box, discard as false positive
                    if status == "NO VEGETATION":
                        continue

                    results.append(
                        PlantHealthResult(
                            bbox=(bx, by, bw, bh),
                            label="Potted Plant",
                            confidence=score,
                            status=status,
                            healthy_pct=healthy_pct,
                            chlorosis_pct=chlorosis_pct,
                            necrosis_pct=necrosis_pct,
                            mean_hue=mean_hue,
                            color_bgr=color_bgr,
                        )
                    )
            except Exception as e:
                print(f"[PlantAI] Edge TPU inference error: {e}")

        # When TPU is disconnected or no plants detected, returns empty results
        return results, latency_ms

    def draw_overlay(
        self,
        frame: np.ndarray,
        results: list[PlantHealthResult],
        latency_ms: float = 0.0,
    ) -> np.ndarray:
        """Annotate frame with bounding boxes, health status badges, and Coral Edge TPU telemetry."""
        if frame is None or frame.size == 0:
            return frame

        out = frame.copy()
        h, w = out.shape[:2]

        scale = max(0.35, min(0.58, w / 700.0))
        font = cv2.FONT_HERSHEY_SIMPLEX
        tag_h = int(36 * (scale / 0.5))

        # Draw detected plant bounding boxes & status badges
        for res in results:
            bx, by, bw, bh = res.bbox
            color = res.color_bgr

            # Main bounding box
            cv2.rectangle(out, (bx, by), (bx + bw, by + bh), color, 2)

            # Header tag with user-friendly terms
            status_text = f"{res.status} ({res.healthy_pct:.0f}%)"
            sub_text = f"Green:{res.healthy_pct:.0f}% Yellow:{res.chlorosis_pct:.0f}% Brown:{res.necrosis_pct:.0f}%"
            conf_text = f"{res.label} {int(res.confidence * 100)}%"

            tag_y1 = max(0, by - tag_h) if by >= tag_h else by
            tag_y2 = by if by >= tag_h else min(h, by + tag_h)
            tag_w = max(bw, int(220 * (scale / 0.5)))

            # Tag background
            cv2.rectangle(out, (bx, tag_y1), (min(w, bx + tag_w), tag_y2), color, -1)
            # Text color (dark on yellow/green, white on red)
            text_color = (0, 0, 0) if res.status != "BROWNING ALERT" else (255, 255, 255)

            line1_y = tag_y1 + int(16 * (scale / 0.5))
            line2_y = tag_y1 + int(32 * (scale / 0.5))
            cv2.putText(out, f"{conf_text} | {status_text}", (bx + 4, line1_y), font, scale, text_color, 1, cv2.LINE_AA)
            cv2.putText(out, sub_text, (bx + 4, line2_y), font, scale * 0.85, text_color, 1, cv2.LINE_AA)

        # Bottom-right HUD badge (completely avoids top bounding box header tags)
        fps = 1000.0 / max(latency_ms, 0.1) if latency_ms > 0 else 0.0
        if self.is_available:
            hud_text = f"Coral TPU: {latency_ms:.1f} ms ({fps:.0f} FPS)"
            dot_color = (0, 255, 0)
            hud_bg_w = int(240 * (scale / 0.5))
        else:
            hud_text = "TPU: Disconnected (Plant AI Inactive)"
            dot_color = (0, 0, 255)
            hud_bg_w = int(280 * (scale / 0.5))

        hud_bg_h = int(24 * (scale / 0.5))
        hud_x = max(6, w - hud_bg_w - 6)
        hud_y = max(6, h - hud_bg_h - 6)

        overlay = out.copy()
        cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_bg_w, hud_y + hud_bg_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)
        cv2.rectangle(out, (hud_x, hud_y), (hud_x + hud_bg_w, hud_y + hud_bg_h), (80, 80, 80), 1)

        cv2.circle(out, (hud_x + 10, hud_y + hud_bg_h // 2), max(3, int(4 * (scale / 0.5))), dot_color, -1)
        cv2.putText(out, hud_text, (hud_x + 20, hud_y + int(hud_bg_h * 0.70)), font, scale * 0.85, (255, 255, 255), 1, cv2.LINE_AA)

        return out
