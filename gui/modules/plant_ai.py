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


# =============================================================================
# Constants & LUT
# =============================================================================

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

CONF_THRESHOLD = 0.10
IOM_THRESHOLD = 0.50


# =============================================================================
# Helper Functions
# =============================================================================

def _compute_scale(w: int) -> float:
    """Compute font/UI scale factor from frame width."""
    return max(0.35, min(0.58, w / 700.0))


def _build_foliage_masks(
    hsv: np.ndarray,
    thresholds: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (healthy, yellow, foliage) boolean masks from an HSV image.

    Args:
        hsv: H x W x 3 HSV image (OpenCV convention, H in [0, 180]).
        thresholds: Dict with keys:
            h_healthy_lo, h_healthy_hi  -- inclusive hue range for healthy green
            h_yellow_lo,  h_yellow_hi   -- inclusive hue range for yellow
            s_min, v_min                -- minimum saturation and value
            and optionally s_max, v_max for upper bounds
    """
    h_chan = hsv[..., 0]
    s_chan = hsv[..., 1]
    v_chan = hsv[..., 2]

    s_min = thresholds.get("s_min", 35)
    v_min = thresholds.get("v_min", 35)
    s_max = thresholds.get("s_max", 255)
    v_max = thresholds.get("v_max", 255)

    sv_ok = (s_chan >= s_min) & (s_chan <= s_max) & (v_chan >= v_min) & (v_chan <= v_max)
    healthy_mask = sv_ok & (h_chan >= thresholds["h_healthy_lo"]) & (h_chan <= thresholds["h_healthy_hi"])
    yellow_mask  = sv_ok & (h_chan >= thresholds["h_yellow_lo"])  & (h_chan <  thresholds["h_yellow_hi"])
    foliage_mask = healthy_mask | yellow_mask
    return healthy_mask, yellow_mask, foliage_mask


def _draw_semitransparent_bg(
    out: np.ndarray,
    rect: tuple[int, int, int, int],
    alpha: float = 0.75,
) -> None:
    """Draw a semi-transparent dark panel at rect = (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = rect
    overlay = out.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0, out)
    cv2.rectangle(out, (x1, y1), (x2, y2), (80, 80, 80), 1)


def _status_color(vi: float) -> tuple[int, int, int]:
    """Map a pseudo-NDVI vegetation index to a BGR status color.

    Thresholds:
        vi < 0.3  -> red    (stressed)
        vi <= 0.45 -> amber  (warning)
        vi > 0.45 -> green  (healthy)
    """
    if vi < 0.3:
        return (0, 0, 230)    # Red
    if vi <= 0.45:
        return (0, 180, 255)  # Amber
    return (30, 210, 30)      # Green


def _intersection_over_minimum(
    box_a: tuple[int, int, int, int] | list[int],
    box_b: tuple[int, int, int, int] | list[int],
) -> float:
    """Return intersection area divided by the smaller box area for (x, y, w, h) boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    smaller_area = min(max(0, aw) * max(0, ah), max(0, bw) * max(0, bh))
    if smaller_area == 0:
        return 0.0
    overlap_w = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    overlap_h = max(0, min(ay + ah, by + bh) - max(ay, by))
    return overlap_w * overlap_h / smaller_area


# Threshold presets
# _ANALYZE_THRESHOLDS = dict(
#     h_healthy_lo=32, h_healthy_hi=88,
#     h_yellow_lo=18,  h_yellow_hi=32,
#     s_min=35, v_min=35,
# )

_HUE_THRESHOLDS = dict(
    h_healthy_lo=34, h_healthy_hi=59,
    h_yellow_lo=29,  h_yellow_hi=34,
    s_min=70, v_min=35, s_max=245, v_max=205,
)

# =============================================================================
# Public Types
# =============================================================================

@dataclass
class PlantHealthResult:
    """Detection and canopy health analysis result for a plant region."""
    bbox: tuple[int, int, int, int]  # (x, y, w, h) in pixel coordinates
    label: str                       # e.g., "Potted Plant"
    confidence: float                # 0.0 to 1.0
    status: str                      # "HEALTHY", "YELLOWING WARNING"
    healthy_pct: float               # Percentage of green foliage
    chlorosis_pct: float             # Percentage of yellowing foliage
    mean_hue: float                  # Mean Hue value (0-180 in OpenCV)
    color_bgr: tuple[int, int, int]  # Visual status color (BGR)
    canopy_coverage: float           # Percentage of bbox area that is plant foliage
    vi: float = 0.0                  # Hue-based pseudo-NDVI vegetation index (0.0 to 1.0)


# =============================================================================
# Analysis Functions
# =============================================================================

def hue_to_pseudo_ndvi(hue: float) -> float:
    """Map OpenCV Hue (0-180) to a linear pseudo-NDVI vegetation index from 0.0 to 1.0.
    
    Linear calibration from 0 to 60:
      - 0.00: Red/Brown / Necrosis (H = 0 or H >= 165)
      - 0.50: Yellow / Chlorosis (H = 30)
      - 1.00: Vibrant Green / Healthy (H >= 60)
    """
    h = 0.0 if hue >= 165.0 else float(hue)
    return float(np.clip(h / 60.0, 0.0, 1.0))


def analyze_crop_health(crop: np.ndarray) -> tuple[str, float, float, float, tuple[int, int, int], float]:
    """Analyze plant canopy health using pure Hue (HSV) color space for green and yellowing.
    
    Brown pixels are ignored -- they do not count toward coverage or health status.

    Returns:
        tuple of (status_text, healthy_pct, yellowing_pct, mean_hue, color_bgr, canopy_coverage)
    """
    if crop is None or crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
        return "NO VEGETATION", 0.0, 0.0, 0.0, (128, 128, 128), 0.0

    total_bbox_pixels = crop.shape[0] * crop.shape[1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    healthy_mask, yellow_mask, foliage_mask = _build_foliage_masks(hsv, _HUE_THRESHOLDS)

    n_plant = int(np.count_nonzero(foliage_mask))
    if n_plant < 30:
        return "NO VEGETATION", 0.0, 0.0, 0.0, (128, 128, 128), 0.0

    n_healthy = int(np.count_nonzero(healthy_mask))
    n_yellow  = int(np.count_nonzero(yellow_mask))
    healthy_pct   = (n_healthy / n_plant) * 100.0
    yellowing_pct = (n_yellow  / n_plant) * 100.0

    mean_hue = float(np.mean(hsv[..., 0][foliage_mask]))
    canopy_coverage = (n_plant / total_bbox_pixels) * 100.0

    if yellowing_pct >= 25.0:
        status = "YELLOWING WARNING"
        color: tuple[int, int, int] = (0, 180, 255)  # Amber
    else:
        status = "HEALTHY"
        color = (30, 210, 30)  # Green

    return status, healthy_pct, yellowing_pct, mean_hue, color, canopy_coverage


# =============================================================================
# PlantAIDetector Class
# =============================================================================

class PlantAIDetector:
    """2-Stage Plant AI detector combining Edge TPU plant localization with canopy stress profiling."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        confidence_threshold: float = CONF_THRESHOLD,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.is_available = False
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.last_latency_ms = 0.0

        base_dir = Path(__file__).resolve().parent.parent
        yolo_model = base_dir / "models" / "yolo26n_exp2.tflite"
        self.model_path = Path(model_path) if model_path else yolo_model
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
                    input_tensor = ((resized / 255.0) / scale + zp).astype(np.uint8) if scale > 0 else resized.astype(np.uint8)
                else:
                    input_tensor = (resized / 255.0).astype(np.float32)

                self.interpreter.set_tensor(in_det["index"], np.expand_dims(input_tensor, axis=0))
                self.interpreter.invoke()
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                self.last_latency_ms = latency_ms

                candidate_boxes: list[tuple[int, int, int, int]] = []
                candidate_scores: list[float] = []

                # YOLO architecture (single output tensor [1, 5, 8400] or [1, num_anchors, channels])
                if len(self.output_details) >= 1:
                    out_det = self.output_details[0]
                    raw_out = self.interpreter.get_tensor(out_det["index"])[0]
                    out_scale, out_zp = out_det["quantization"]
                    out_float = (raw_out.astype(np.float32) - out_zp) * out_scale if out_scale > 0 else raw_out.astype(np.float32)

                    if out_float.shape[0] < out_float.shape[1]:
                        out_float = out_float.T  # Transpose [5, 8400] -> [8400, 5]

                    cx, cy = out_float[:, 0], out_float[:, 1]
                    bw_norm, bh_norm = out_float[:, 2], out_float[:, 3]
                    scores = out_float[:, 4] if out_float.shape[1] == 5 else np.max(out_float[:, 4:], axis=1)
                    class_ids = np.argmax(out_float[:, 4:], axis=1) if out_float.shape[1] > 5 else None

                    mask = scores >= threshold
                    cx, cy, bw_norm, bh_norm, scores = cx[mask], cy[mask], bw_norm[mask], bh_norm[mask], scores[mask]
                    if class_ids is not None:
                        class_ids = class_ids[mask]

                    boxes_for_nms = [
                        [int((c_x - b_w / 2.0) * w), int((c_y - b_h / 2.0) * h), int(b_w * w), int(b_h * h)]
                        for c_x, c_y, b_w, b_h in zip(cx, cy, bw_norm, bh_norm)
                    ]

                    indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores.tolist(), threshold, 0.45)
                    if len(indices) > 0:
                        # Normal IoU NMS misses small boxes nested inside larger boxes:
                        # their IoU can be low even when they depict the same plant.
                        # Apply IoM only to NMS survivors, keeping the best score first.
                        kept_indices: list[int] = []
                        for idx in sorted(indices.flatten(), key=lambda i: float(scores[i]), reverse=True):
                            if any(
                                (class_ids is None or class_ids[idx] == class_ids[kept])
                                and _intersection_over_minimum(boxes_for_nms[idx], boxes_for_nms[kept]) > IOM_THRESHOLD
                                for kept in kept_indices
                            ):
                                continue
                            kept_indices.append(idx)

                        for idx in kept_indices:
                            candidate_boxes.append(tuple(boxes_for_nms[idx]))
                            candidate_scores.append(float(scores[idx]))

                # Stage 2: Canopy stress profiling on all candidate plant regions
                for (bx, by, bw, bh), score in zip(candidate_boxes, candidate_scores):
                    bx_c = max(0, min(w - 1, bx))
                    by_c = max(0, min(h - 1, by))
                    bw_c = max(5, min(w - bx_c, bw))
                    bh_c = max(5, min(h - by_c, bh))

                    crop = frame[by_c : by_c + bh_c, bx_c : bx_c + bw_c]
                    status, healthy_pct, chlorosis_pct, mean_hue, color_bgr, canopy_coverage = analyze_crop_health(crop)

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
        scale = _compute_scale(w)
        font = cv2.FONT_HERSHEY_SIMPLEX
        tag_h = int(22 * (scale / 0.5))

        # Draw detected plant bounding boxes & status badges
        for res in results:
            bx, by, bw, bh = res.bbox
            vi = res.vi if res.vi > 0.0 else hue_to_pseudo_ndvi(res.mean_hue)
            color = _status_color(vi)

            if show_bbox:
                cv2.rectangle(out, (bx, by), (bx + bw, by + bh), color, 2)

            line1_text = f"VI: {vi:.2f}, Coverage: {res.canopy_coverage:.0f}%"
            tag_y1 = max(0, by - tag_h) if by >= tag_h else by
            tag_y2 = by if by >= tag_h else min(h, by + tag_h)
            tag_w = max(bw, int(180 * (scale / 0.5)))

            cv2.rectangle(out, (bx, tag_y1), (min(w, bx + tag_w), tag_y2), color, -1)
            text_color = (255, 255, 255) if vi < 0.3 else (0, 0, 0)
            cv2.putText(out, line1_text, (bx + 4, tag_y1 + int(16 * (scale / 0.5))), font, scale, text_color, 1, cv2.LINE_AA)

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
        scale = _compute_scale(w)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        _, _, foliage_mask = _build_foliage_masks(hsv, _HUE_THRESHOLDS)

        out = cv2.convertScaleAbs(frame, alpha=0.4).copy() if dim_background else frame.copy()

        if np.any(foliage_mask):
            h_chan = hsv[..., 0]
            h_float = h_chan.astype(np.float32)
            lut_indices = np.zeros_like(h_float)

            # Yellow foliage: map H [29, 34) -> LUT indices [80, 160) (amber to yellow-green)
            m_y = (h_chan >= 29) & (h_chan < 34)
            lut_indices[m_y] = 80.0 + ((h_float[m_y] - 29.0) / 5.0) * 80.0

            # Green foliage: map H [34, 60) -> LUT indices [160, 255] (vibrant green)
            m_g = (h_chan >= 34) & (h_chan < 60)
            lut_indices[m_g] = 160.0 + ((h_float[m_g] - 34.0) / 26.0) * 95.0

            colored_foliage = RDYLGN_LUT[np.clip(lut_indices, 0, 255).astype(np.uint8)]
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
        pad   = int(8  * (scale / 0.5))
        bg_w  = bar_w + pad * 2
        bg_h  = int(36 * (scale / 0.5))
        leg_x = 10
        leg_y = max(6, h - bg_h - 10)

        _draw_semitransparent_bg(out, (leg_x, leg_y, leg_x + bg_w, leg_y + bg_h))

        text_y = leg_y + int(12 * (scale / 0.5))
        cv2.putText(out, "Stressed", (leg_x + pad, text_y), font, scale * 0.7, (0, 100, 255), 1, cv2.LINE_AA)
        healthy_text = "Healthy"
        (th_w, _), _ = cv2.getTextSize(healthy_text, font, scale * 0.7, 1)
        cv2.putText(out, healthy_text, (leg_x + pad + bar_w - th_w, text_y), font, scale * 0.7, (50, 220, 50), 1, cv2.LINE_AA)

        bar_x = leg_x + pad
        bar_y = leg_y + int(18 * (scale / 0.5))
        grad_1d = RDYLGN_LUT[np.linspace(0, 255, bar_w).astype(np.uint8)]
        out[bar_y : bar_y + bar_h, bar_x : bar_x + bar_w] = np.tile(grad_1d, (bar_h, 1, 1))
        cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (120, 120, 120), 1)

    def _draw_hud(self, out: np.ndarray, latency_ms: float = 0.0, scale: float = 0.5) -> None:
        """Draw Coral Edge TPU latency and status badge in bottom-right corner."""
        h, w = out.shape[:2]
        if w < 120 or h < 60:
            return
        font = cv2.FONT_HERSHEY_SIMPLEX
        fps = 1000.0 / max(latency_ms, 0.1) if latency_ms > 0 else 0.0

        if self.is_available:
            hud_text  = f"Coral TPU: {latency_ms:.1f} ms ({fps:.0f} FPS)"
            dot_color = (0, 255, 0)
            hud_bg_w  = int(240 * (scale / 0.5))
        else:
            hud_text  = "TPU: Disconnected (Plant AI Inactive)"
            dot_color = (0, 0, 255)
            hud_bg_w  = int(280 * (scale / 0.5))

        hud_bg_h = int(24 * (scale / 0.5))
        hud_x = max(6, w - hud_bg_w - 6)
        hud_y = max(6, h - hud_bg_h - 6)

        _draw_semitransparent_bg(out, (hud_x, hud_y, hud_x + hud_bg_w, hud_y + hud_bg_h))

        cv2.circle(out, (hud_x + 10, hud_y + hud_bg_h // 2), max(3, int(4 * (scale / 0.5))), dot_color, -1)
        cv2.putText(out, hud_text, (hud_x + 20, hud_y + int(hud_bg_h * 0.70)), font, scale * 0.85, (255, 255, 255), 1, cv2.LINE_AA)
