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
    canopy_coverage: float           # Percentage of bbox area that is plant foliage
    vi: float = 0.0                  # Hue-based pseudo-NDVI vegetation index (0.0 to 1.0)


def _create_rdylgn_lut() -> np.ndarray:
    """Create 256-entry BGR lookup table for RdYlGn (Red-Yellow-Green) colormap."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    control_points = [
        (0, (0, 0, 200)),      # Red
        (64, (0, 128, 255)),   # Orange
        (128, (0, 230, 230)),  # Yellow
        (192, (0, 200, 100)),  # Yellow-Green
        (255, (0, 180, 0)),    # Green
    ]
    for i in range(len(control_points) - 1):
        idx0, c0 = control_points[i]
        idx1, c1 = control_points[i + 1]
        steps = idx1 - idx0
        for step in range(steps):
            t = step / steps
            lut[idx0 + step] = [
                int(c0[c] * (1.0 - t) + c1[c] * t) for c in range(3)
            ]
    lut[255] = control_points[-1][1]
    return lut


RDYLGN_LUT = _create_rdylgn_lut()


def hue_to_pseudo_ndvi(hue: float) -> float:
    """Map OpenCV Hue (0-180) to a linear pseudo-NDVI vegetation index from 0.0 to 1.0.
    
    Linear calibration from 0 to 60:
      - 0.00: Red/Brown / Necrosis (H = 0 or H >= 165)
      - 0.50: Yellow / Chlorosis (H = 30)
      - 1.00: Vibrant Green / Healthy (H >= 60)
    """
    h = 0.0 if hue >= 165.0 else float(hue)
    return float(np.clip(h / 60.0, 0.0, 1.0))


def analyze_crop_health(crop: np.ndarray) -> tuple[str, float, float, float, float, tuple[int, int, int], float]:
    """Analyze plant canopy health using pure Hue (HSV) color space for green and yellowing.
    
    Returns:
        tuple of (status_text, healthy_pct, yellowing_pct, browning_pct, mean_hue, color_bgr, canopy_coverage)
    """
    if crop is None or crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
        return "NO VEGETATION", 0.0, 0.0, 0.0, 0.0, (128, 128, 128), 0.0

    total_bbox_pixels = crop.shape[0] * crop.shape[1]

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0]  # OpenCV Hue: 0 - 180 (corresponds to 0° - 360°)
    s = hsv[..., 1]  # Saturation: 0 - 255
    v = hsv[..., 2]  # Value / Brightness: 0 - 255

    # Plant canopy masks within detected crop box:
    healthy_mask = (s >= 35) & (v >= 35) & (h >= 32) & (h <= 88)
    yellow_mask = (s >= 35) & (v >= 35) & (h >= 18) & (h < 32)
    brown_mask = (s >= 50) & (v >= 35) & ((h < 18) | (h >= 165))
    foliage_mask = healthy_mask | yellow_mask | brown_mask
    n_plant = int(np.count_nonzero(foliage_mask))

    if n_plant < 30:
        return "NO VEGETATION", 0.0, 0.0, 0.0, 0.0, (128, 128, 128), 0.0

    n_healthy = int(np.count_nonzero(healthy_mask))
    n_yellow = int(np.count_nonzero(yellow_mask))
    n_brown = int(np.count_nonzero(brown_mask))

    healthy_pct = (n_healthy / n_plant) * 100.0
    yellowing_pct = (n_yellow / n_plant) * 100.0
    browning_pct = (n_brown / n_plant) * 100.0

    # Mean hue across living plant pixels (healthy + yellow), or all foliage if only brown
    green_yellow_mask = healthy_mask | yellow_mask
    if np.any(green_yellow_mask):
        mean_hue = float(np.mean(h[green_yellow_mask]))
    elif np.any(brown_mask):
        mean_hue = float(np.mean(h[brown_mask]))
    else:
        mean_hue = 0.0

    canopy_coverage = (n_plant / total_bbox_pixels) * 100.0 if total_bbox_pixels > 0 else 0.0

    if browning_pct >= 20.0 or (browning_pct >= 10.0 and (browning_pct + yellowing_pct) >= 45.0):
        status = "BROWNING ALERT"
        color = (0, 0, 230)  # Red
    elif yellowing_pct >= 25.0:
        status = "YELLOWING WARNING"
        color = (0, 180, 255)  # Amber
    else:
        status = "HEALTHY"
        color = (30, 210, 30)  # Green

    return status, healthy_pct, yellowing_pct, browning_pct, mean_hue, color, canopy_coverage


class PlantAIDetector:
    """2-Stage Plant AI detector combining Edge TPU plant localization with canopy stress profiling."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        confidence_threshold: float = 0.08,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.is_available = False
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.last_latency_ms = 0.0

        base_dir = Path(__file__).resolve().parent.parent
        yolo_model = base_dir / "models" / "yolo26n_e100.tflite"

        if model_path:
            self.model_path = Path(model_path)
        else:
            self.model_path = yolo_model

        self._init_edgetpu()

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
        """Execute Stage 1 Edge TPU localization (YOLO) and Stage 2 canopy stress profiling.
        
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
                
                in_det = self.input_details[0]
                target_h, target_w = in_det["shape"][1], in_det["shape"][2]

                # Convert BGR -> RGB and resize
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (target_w, target_h))

                # Handle INT8 / UINT8 / FLOAT32 input quantization
                if in_det["dtype"] == np.int8:
                    scale, zp = in_det["quantization"]
                    input_tensor = ((resized / 255.0) / scale + zp).astype(np.int8)
                elif in_det["dtype"] == np.uint8:
                    scale, zp = in_det["quantization"]
                    if scale > 0:
                        input_tensor = ((resized / 255.0) / scale + zp).astype(np.uint8)
                    else:
                        input_tensor = resized.astype(np.uint8)
                else:
                    input_tensor = (resized / 255.0).astype(np.float32)

                input_tensor = np.expand_dims(input_tensor, axis=0)
                self.interpreter.set_tensor(in_det["index"], input_tensor)
                self.interpreter.invoke()
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                self.last_latency_ms = latency_ms

                candidate_boxes: list[tuple[int, int, int, int]] = []
                candidate_scores: list[float] = []

                # YOLO architecture (single output tensor [1, 5, 8400] or [1, num_anchors, channels])
                if len(self.output_details) >= 1:
                    out_det = self.output_details[0]
                    raw_out = self.interpreter.get_tensor(out_det["index"])[0]
                    if out_det["quantization"][0] > 0:
                        out_scale, out_zp = out_det["quantization"]
                        out_float = (raw_out.astype(np.float32) - out_zp) * out_scale
                    else:
                        out_float = raw_out.astype(np.float32)

                    if out_float.shape[0] < out_float.shape[1]:
                        out_float = out_float.T  # Transpose [5, 8400] -> [8400, 5]

                    cx = out_float[:, 0]
                    cy = out_float[:, 1]
                    bw_norm = out_float[:, 2]
                    bh_norm = out_float[:, 3]
                    scores = out_float[:, 4] if out_float.shape[1] == 5 else np.max(out_float[:, 4:], axis=1)

                    mask = scores >= threshold
                    cx, cy, bw_norm, bh_norm, scores = cx[mask], cy[mask], bw_norm[mask], bh_norm[mask], scores[mask]

                    boxes_for_nms = []
                    for c_x, c_y, b_w, b_h in zip(cx, cy, bw_norm, bh_norm):
                        bx = int((c_x - b_w / 2.0) * w)
                        by = int((c_y - b_h / 2.0) * h)
                        box_w = int(b_w * w)
                        box_h = int(b_h * h)
                        boxes_for_nms.append([bx, by, box_w, box_h])

                    indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores.tolist(), threshold, 0.45)
                    if len(indices) > 0:
                        for idx in indices.flatten():
                            candidate_boxes.append(tuple(boxes_for_nms[idx]))
                            candidate_scores.append(float(scores[idx]))

                # Stage 2: Canopy stress profiling on all candidate plant regions
                for (bx, by, bw, bh), score in zip(candidate_boxes, candidate_scores):
                    bx_c = max(0, min(w - 1, bx))
                    by_c = max(0, min(h - 1, by))
                    bw_c = max(5, min(w - bx_c, bw))
                    bh_c = max(5, min(h - by_c, bh))

                    crop = frame[by_c : by_c + bh_c, bx_c : bx_c + bw_c]
                    status, healthy_pct, chlorosis_pct, necrosis_pct, mean_hue, color_bgr, canopy_coverage = analyze_crop_health(crop)

                    # Discard regions with zero foliage pixels
                    if status == "NO VEGETATION":
                        continue

                    results.append(
                        PlantHealthResult(
                            bbox=(bx_c, by_c, bw_c, bh_c),
                            label="Plant",
                            confidence=score,
                            status=status,
                            healthy_pct=healthy_pct,
                            chlorosis_pct=chlorosis_pct,
                            necrosis_pct=necrosis_pct,
                            mean_hue=mean_hue,
                            color_bgr=color_bgr,
                            canopy_coverage=canopy_coverage,
                            vi=hue_to_pseudo_ndvi(mean_hue),
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
        show_bbox: bool = True,
    ) -> np.ndarray:
        """Annotate frame with bounding boxes, health status badges, and Coral Edge TPU telemetry."""
        if frame is None or frame.size == 0:
            return frame

        out = frame.copy()
        h, w = out.shape[:2]

        scale = max(0.35, min(0.58, w / 700.0))
        font = cv2.FONT_HERSHEY_SIMPLEX
        tag_h = int(22 * (scale / 0.5))

        # Draw detected plant bounding boxes & status badges
        for res in results:
            bx, by, bw, bh = res.bbox
            vi = res.vi if res.vi > 0.0 else hue_to_pseudo_ndvi(res.mean_hue)

            # VI determines bbox color: <0.3 red, 0.3-0.45 amber, >0.45 green
            if vi < 0.3:
                color = (0, 0, 230)  # Red
            elif vi <= 0.45:
                color = (0, 180, 255)  # Amber
            else:
                color = (30, 210, 30)  # Green

            # Main bounding box
            if show_bbox:
                cv2.rectangle(out, (bx, by), (bx + bw, by + bh), color, 2)

            # Header tag with user-friendly terms & coverage
            line1_text = f"VI: {vi:.2f}, Coverage: {res.canopy_coverage:.0f}%"

            tag_y1 = max(0, by - tag_h) if by >= tag_h else by
            tag_y2 = by if by >= tag_h else min(h, by + tag_h)
            tag_w = max(bw, int(180 * (scale / 0.5)))

            # Tag background
            cv2.rectangle(out, (bx, tag_y1), (min(w, bx + tag_w), tag_y2), color, -1)
            # Text color (dark on yellow/green, white on red)
            text_color = (255, 255, 255) if vi < 0.3 else (0, 0, 0)

            line1_y = tag_y1 + int(16 * (scale / 0.5))
            cv2.putText(out, line1_text, (bx + 4, line1_y), font, scale, text_color, 1, cv2.LINE_AA)

        self._draw_hud(out, latency_ms, scale)
        return out


    def draw_full_frame_heatmap(
        self,
        frame: np.ndarray,
        latency_ms: float = 0.0,
        show_hud: bool = True,
        dim_background: bool = True,
    ) -> np.ndarray:
        """Render pseudo-NDVI heatmap overlay across all plant foliage in the full frame with dimmed background."""
        if frame is None or frame.size == 0:
            return frame

        h, w = frame.shape[:2]
        scale = max(0.35, min(0.58, w / 700.0))

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_chan = hsv[..., 0]
        s_chan = hsv[..., 1]
        v_chan = hsv[..., 2]

        # Strict plant foliage masks:
        # Healthy green foliage (H in [34, 60), S >= 70 to exclude white background and perlite, V <= 205 to exclude bright specular reflections)
        healthy_mask = (
            (s_chan >= 70) & (s_chan <= 245) &
            (v_chan >= 35) & (v_chan <= 205) &
            (h_chan >= 34) & (h_chan < 60)
        )
        # Yellowing/chlorotic foliage (H in [29, 34), S >= 60 to exclude tray/reflections, V <= 205)
        yellow_mask = (
            (s_chan >= 60) & (s_chan <= 245) &
            (v_chan >= 35) & (v_chan <= 205) &
            (h_chan >= 29) & (h_chan < 34)
        )
        # Foliage mask strictly contains healthy green and yellowing foliage.
        foliage_mask = healthy_mask | yellow_mask

        # Dim background (non-foliage) pixels for high visual contrast
        if dim_background:
            dimmed = cv2.convertScaleAbs(frame, alpha=0.4)
            out = dimmed.copy()
        else:
            out = frame.copy()

        if np.any(foliage_mask):
            h_float = h_chan.astype(np.float32)
            lut_indices = np.zeros_like(h_float)

            # Yellow foliage: map H [29, 34) -> LUT indices [80, 160) (amber to yellow-green)
            m_y = (h_chan >= 29) & (h_chan < 34)
            lut_indices[m_y] = 80.0 + ((h_float[m_y] - 29.0) / 5.0) * 80.0

            # Green foliage: map H [34, 60) -> LUT indices [160, 255] (vibrant green)
            m_g = (h_chan >= 34) & (h_chan < 60)
            lut_indices[m_g] = 160.0 + ((h_float[m_g] - 34.0) / 26.0) * 95.0

            lut_indices = np.clip(lut_indices, 0, 255).astype(np.uint8)
            colored_foliage = RDYLGN_LUT[lut_indices]

            out[foliage_mask] = colored_foliage[foliage_mask]

        self._draw_heatmap_legend(out, scale)
        if show_hud:
            self._draw_hud(out, latency_ms, scale)
        return out

    def _draw_heatmap_legend(self, out: np.ndarray, scale: float) -> None:
        """Draw pseudo-NDVI RdYlGn color legend bar in the bottom-left corner."""
        h, w = out.shape[:2]
        if w < 120 or h < 60:
            return
        font = cv2.FONT_HERSHEY_SIMPLEX

        bar_w = int(140 * (scale / 0.5))
        bar_h = int(10 * (scale / 0.5))
        pad = int(8 * (scale / 0.5))
        bg_w = bar_w + pad * 2
        bg_h = int(36 * (scale / 0.5))

        leg_x = 10
        leg_y = max(6, h - bg_h - 10)

        # Semi-transparent background
        overlay = out.copy()
        cv2.rectangle(overlay, (leg_x, leg_y), (leg_x + bg_w, leg_y + bg_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)
        cv2.rectangle(out, (leg_x, leg_y), (leg_x + bg_w, leg_y + bg_h), (80, 80, 80), 1)

        # Text labels
        text_y = leg_y + int(12 * (scale / 0.5))
        cv2.putText(out, "Stressed", (leg_x + pad, text_y), font, scale * 0.7, (0, 100, 255), 1, cv2.LINE_AA)
        healthy_text = "Healthy"
        (th_w, _), _ = cv2.getTextSize(healthy_text, font, scale * 0.7, 1)
        cv2.putText(out, healthy_text, (leg_x + pad + bar_w - th_w, text_y), font, scale * 0.7, (50, 220, 50), 1, cv2.LINE_AA)

        # Gradient bar
        bar_x = leg_x + pad
        bar_y = leg_y + int(18 * (scale / 0.5))
        grad_1d = RDYLGN_LUT[np.linspace(0, 255, bar_w).astype(np.uint8)]
        grad_2d = np.tile(grad_1d, (bar_h, 1, 1))
        out[bar_y : bar_y + bar_h, bar_x : bar_x + bar_w] = grad_2d
        cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (120, 120, 120), 1)

    def _draw_hud(self, out: np.ndarray, latency_ms: float = 0.0, scale: float = 0.5) -> None:
        """Draw Coral Edge TPU latency and status badge in bottom-right corner."""
        h, w = out.shape[:2]
        if w < 120 or h < 60:
            return
        font = cv2.FONT_HERSHEY_SIMPLEX
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
