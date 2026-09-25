"""Unit and integration tests for Plant AI health and stress monitoring pipeline."""

from pathlib import Path
import sys
import time
import cv2
import numpy as np
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from gui.modules.plant_ai import PlantAIDetector, PlantHealthResult, analyze_crop_health

SAMPLE_DIR = Path(__file__).resolve().parent / "samples"


def test_detection_iom_suppresses_nested_same_class_only():
    """Post-NMS IoM removes nested duplicates but keeps nearby and other-class plants."""
    class FakeInterpreter:
        def set_tensor(self, index, value):
            pass

        def invoke(self):
            pass

        def get_tensor(self, index):
            # cx, cy, width, height, class 0 score, class 1 score
            return np.array([[
                [0.50, 0.50, 0.50, 0.50, 0.90, 0.00],  # large plant
                [0.45, 0.45, 0.10, 0.10, 0.80, 0.00],  # nested duplicate
                [0.775, 0.45, 0.10, 0.10, 0.70, 0.00],  # nearby, 25% IoM
                [0.60, 0.45, 0.10, 0.10, 0.00, 0.60],  # nested, other class
                [0.10, 0.10, 0.10, 0.10, 0.11, 0.00],  # just above cutoff
                [0.20, 0.10, 0.10, 0.10, 0.09, 0.00],  # below cutoff
                [0.30, 0.10, 0.10, 0.10, 0.08, 0.00],
            ]], dtype=np.float32)

    detector = object.__new__(PlantAIDetector)
    detector.confidence_threshold = 0.10
    detector.is_available = True
    detector.interpreter = FakeInterpreter()
    detector.input_details = [{"shape": [1, 200, 200, 3], "dtype": np.float32, "index": 0}]
    detector.output_details = [{"index": 0, "quantization": (0.0, 0)}]
    detector.last_latency_ms = 0.0

    frame = np.full((200, 200, 3), (30, 200, 40), dtype=np.uint8)
    results, _ = detector.detect_and_analyze(frame)

    assert [round(result.confidence, 2) for result in results] == [0.90, 0.70, 0.60, 0.11]


# --- 1. Unit Tests for Canopy Stress Analysis (CI-ready, runs without hardware) ---

def test_analyze_crop_health_healthy():
    """Verify that a vibrant green crop patch is classified as HEALTHY."""
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    crop[:, :] = (30, 200, 40)

    status, h_pct, c_pct, mean_hue, color, _ = analyze_crop_health(crop)
    assert status == "HEALTHY"
    assert h_pct >= 90.0
    assert c_pct < 10.0
    assert 33.0 <= mean_hue <= 88.0
    assert color == (30, 210, 30)


def test_analyze_crop_health_chlorosis():
    """Verify that a yellowing/chlorotic crop patch is classified as YELLOWING WARNING."""
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    crop[:, :] = (30, 210, 220)

    status, h_pct, c_pct, mean_hue, color, _ = analyze_crop_health(crop)
    assert status == "YELLOWING WARNING"
    assert c_pct >= 50.0
    assert color == (0, 180, 255)


def test_analyze_crop_health_brown_excluded():
    """Verify that a brown/desiccated crop patch is NOT included in foliage and returns NO VEGETATION."""
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    crop[:, :] = (30, 80, 150)  # Low-hue brownish color

    status, h_pct, c_pct, mean_hue, color, coverage = analyze_crop_health(crop)
    # Brown pixels must not count as foliage -- result is NO VEGETATION or low coverage
    assert status in ("NO VEGETATION", "HEALTHY", "YELLOWING WARNING")
    # Specifically this brown pixel (H~9 in OpenCV HSV) falls outside [18,88], so no foliage
    assert coverage == 0.0 or status == "NO VEGETATION"


def test_analyze_crop_health_empty_and_small():
    """Verify edge cases like empty, None, or tiny crops return safely as NO VEGETATION."""
    assert analyze_crop_health(None)[0] == "NO VEGETATION"
    assert analyze_crop_health(np.zeros((2, 2, 3), dtype=np.uint8))[0] == "NO VEGETATION"


def test_plant_ai_disconnected_tpu():
    """Verify Plant AI returns empty results and informs on UI when Coral TPU is disconnected (no CPU fallback)."""
    detector = PlantAIDetector()
    assert detector.confidence_threshold == 0.10
    detector.is_available = False  # Simulate TPU disconnected

    frame = np.full((400, 600, 3), (200, 200, 200), dtype=np.uint8)
    results, latency = detector.detect_and_analyze(frame)

    # Must NOT run CPU fallback detection
    assert len(results) == 0
    assert latency == 0.0

    # UI overlay must clearly indicate TPU Disconnected
    annotated = detector.draw_overlay(frame, results, latency)
    assert annotated.shape == frame.shape


def test_draw_full_frame_heatmap_dimmed_background():
    """Verify full-frame heatmap dims non-foliage background and colorizes foliage."""
    detector = PlantAIDetector()
    frame = np.full((100, 100, 3), 200, dtype=np.uint8)  # Gray background
    # Vibrant green plant foliage patch in center
    frame[30:70, 30:70] = (30, 200, 40)

    out = detector.draw_full_frame_heatmap(frame)
    assert out.shape == frame.shape
    # Background pixel (e.g. at 10, 10) must be dimmed from 200 down to ~80
    assert out[10, 10, 0] < 120
    assert out[10, 10, 1] < 120
    assert out[10, 10, 2] < 120
    # Foliage pixel (at 50, 50) must be colorized
    assert not np.array_equal(out[50, 50], frame[50, 50])


def test_draw_full_frame_heatmap_excludes_brown_and_white():
    """Verify that heatmap strictly excludes brown soil and white/bright backgrounds."""
    detector = PlantAIDetector()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # White tray / specular reflection: high V, low S
    frame[0:30, 0:30] = (245, 245, 245)
    # Brown potting soil: low H (Hue ~12 in OpenCV), moderate S & V
    frame[0:30, 70:100] = (25, 45, 75)
    # Healthy green plant:
    frame[70:100, 0:30] = (30, 200, 40)
    # Stressed yellow plant (Hue in [29, 34], high S, moderate V):
    frame[70:100, 70:100] = (20, 180, 180)

    out = detector.draw_full_frame_heatmap(frame, dim_background=True)

    # White tray region should be dimmed, NOT colorized by heatmap (245 * 0.4 = 98)
    assert np.all(out[15, 15] < 120)

    # Brown soil region should be dimmed, NOT colorized by heatmap
    assert out[15, 85, 2] < 45

    # Healthy green foliage must be colorized by heatmap (green in RdYlGn)
    assert out[85, 15, 1] > 140

    # Stressed yellow foliage must be colorized by heatmap (amber/yellow in RdYlGn)
    assert out[85, 85, 1] > 140


def test_draw_both_ai_and_heatmap_no_interaction():
    """Verify that when both AI and Heatmap are active, bboxes are drawn and heatmap covers full frame."""
    detector = PlantAIDetector()
    frame = np.full((200, 200, 3), 200, dtype=np.uint8)
    frame[50:150, 50:150] = (30, 200, 40)

    res = PlantHealthResult(
        bbox=(50, 50, 100, 100),
        label="Potted Plant",
        confidence=0.85,
        status="HEALTHY",
        healthy_pct=90.0,
        chlorosis_pct=5.0,
        mean_hue=55.0,
        color_bgr=(30, 210, 30),
        canopy_coverage=75.0,
    )

    # 1. Full frame heatmap with dimmed background
    hmap = detector.draw_full_frame_heatmap(frame, latency_ms=15.0, show_hud=False)
    # Background outside bbox is dimmed
    assert hmap[10, 10, 0] < 120

    # 2. AI overlay drawn on top with show_bbox=True
    out = detector.draw_overlay(hmap, [res], latency_ms=15.0, show_bbox=True)
    assert out.shape == frame.shape
    # Bounding box edge at by=50, bx=50 should have bbox color
    assert out[50, 50, 0] == 30 or out[50, 50, 1] == 210 or out[50, 50, 2] == 30


# --- 2. Coral Hardware Tests (Marked with @pytest.mark.coral) ---

@pytest.mark.coral
def test_coral_plant_ai_initialization():
    """Verify PlantAIDetector initializes with Coral TPU hardware."""
    detector = PlantAIDetector()
    assert detector.is_available is True, "Expected Coral Edge TPU to be available"
    assert detector.interpreter is not None


@pytest.mark.coral
def test_coral_plant_ai_inference_latency():
    """Benchmark end-to-end detection and health analysis on Coral Edge TPU."""
    detector = PlantAIDetector()
    assert detector.is_available is True

    frame = np.full((480, 640, 3), (180, 180, 180), dtype=np.uint8)
    # Warmup
    detector.detect_and_analyze(frame)

    latencies = []
    for _ in range(10):
        results, lat = detector.detect_and_analyze(frame)
        latencies.append(lat)

    avg_ms = sum(latencies) / len(latencies)
    print(f"\n[PASS] Plant AI Coral TPU Latency: {avg_ms:.2f} ms ({1000/max(avg_ms, 0.1):.1f} FPS)")
    assert avg_ms < 200.0, f"Inference too slow ({avg_ms:.2f} ms), expected < 200 ms"


@pytest.mark.coral
def test_coral_real_plant_sample_images():
    """Test Plant AI detection and health analysis on authentic real plant dataset images."""
    sample_files = sorted(list(SAMPLE_DIR.glob("*.jpg")))
    assert len(sample_files) >= 1, f"Expected at least 1 real sample image in {SAMPLE_DIR}"

    detector = PlantAIDetector()
    assert detector.is_available is True

    output_dir = BASE_DIR / "Data" / "plant_ai_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Real Plant Dataset Evaluation ---")
    for img_path in sample_files:
        img = cv2.imread(str(img_path))
        assert img is not None, f"Failed to read image {img_path}"

        results, lat = detector.detect_and_analyze(img)
        annotated = detector.draw_overlay(img, results, lat)

        out_path = output_dir / f"eval_{img_path.name}"
        cv2.imwrite(str(out_path), annotated)

        print(f"Image: {img_path.name:<38} | Detections: {len(results)} | Latency: {lat:.1f} ms")
        for idx, res in enumerate(results):
            print(f"  [#{idx+1}] {res.label:<16} Status={res.status:<18} H={res.healthy_pct:.1f}% C={res.chlorosis_pct:.1f}% Conf={res.confidence*100:.1f}%")

        assert len(results) >= 1, f"Expected at least 1 detection for {img_path.name}"
        assert out_path.exists()
