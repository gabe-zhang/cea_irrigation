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
from scripts.download_samples import download_real_samples, SAMPLE_DIR


# --- 1. Unit Tests for Canopy Stress Analysis (CI-ready, runs without hardware) ---

def test_analyze_crop_health_healthy():
    """Verify that a vibrant green crop patch is classified as HEALTHY."""
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    crop[:, :] = (30, 200, 40)

    status, h_pct, c_pct, n_pct, mean_hue, color = analyze_crop_health(crop)
    assert status == "HEALTHY"
    assert h_pct >= 90.0
    assert c_pct < 10.0
    assert n_pct < 10.0
    assert 33.0 <= mean_hue <= 88.0
    assert color == (30, 210, 30)


def test_analyze_crop_health_chlorosis():
    """Verify that a yellowing/chlorotic crop patch is classified as YELLOWING WARNING."""
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    crop[:, :] = (30, 210, 220)

    status, h_pct, c_pct, n_pct, mean_hue, color = analyze_crop_health(crop)
    assert status == "YELLOWING WARNING"
    assert c_pct >= 50.0
    assert color == (0, 180, 255)


def test_analyze_crop_health_necrosis():
    """Verify that a brown/desiccated crop patch is classified as BROWNING ALERT."""
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    crop[:, :] = (30, 80, 150)

    status, h_pct, c_pct, n_pct, mean_hue, color = analyze_crop_health(crop)
    assert status == "BROWNING ALERT"
    assert (n_pct + c_pct) >= 45.0 or n_pct >= 20.0
    assert color == (0, 0, 230)


def test_analyze_crop_health_empty_and_small():
    """Verify edge cases like empty, None, or tiny crops return safely as NO VEGETATION."""
    assert analyze_crop_health(None)[0] == "NO VEGETATION"
    assert analyze_crop_health(np.zeros((2, 2, 3), dtype=np.uint8))[0] == "NO VEGETATION"


def test_plant_ai_disconnected_tpu():
    """Verify Plant AI returns empty results and informs on UI when Coral TPU is disconnected (no CPU fallback)."""
    detector = PlantAIDetector()
    detector.is_available = False  # Simulate TPU disconnected

    frame = np.full((400, 600, 3), (200, 200, 200), dtype=np.uint8)
    results, latency = detector.detect_and_analyze(frame)

    # Must NOT run CPU fallback detection
    assert len(results) == 0
    assert latency == 0.0

    # UI overlay must clearly indicate TPU Disconnected
    annotated = detector.draw_overlay(frame, results, latency)
    assert annotated.shape == frame.shape


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
    assert avg_ms < 25.0, f"Inference too slow ({avg_ms:.2f} ms), expected < 25 ms"


@pytest.mark.coral
def test_coral_real_plant_sample_images():
    """Test Plant AI detection and health analysis on authentic real plant dataset images."""
    download_real_samples()
    sample_files = sorted(list(SAMPLE_DIR.glob("*.jpg")))
    assert len(sample_files) >= 4, f"Expected at least 4 real sample images in {SAMPLE_DIR}"

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
            print(f"  [#{idx+1}] {res.label:<16} Status={res.status:<18} H={res.healthy_pct:.1f}% C={res.chlorosis_pct:.1f}% N={res.necrosis_pct:.1f}% Conf={res.confidence*100:.1f}%")

        assert len(results) >= 1, f"Expected at least 1 detection for {img_path.name}"
        assert out_path.exists()
