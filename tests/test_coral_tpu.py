"""Unit tests for Google Coral USB Edge TPU acceleration and Picamera2 integration."""

from pathlib import Path
import sys
import time
import pytest
import numpy as np

# Module-level marker: skips all tests in this module unless `--coral` is passed to pytest
pytestmark = pytest.mark.coral

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

MODEL_PATH = Path(__file__).resolve().parent / "models" / "mobilenet_v2_1.0_224_quant_edgetpu.tflite"
LABELS_PATH = Path(__file__).resolve().parent / "models" / "imagenet_labels.txt"


def test_edgetpu_delegate_load():
    """Test that LiteRT can locate and load the libedgetpu C delegate library."""
    import ai_edge_litert.interpreter as tflite

    delegate = tflite.load_delegate("libedgetpu.so.1")
    assert delegate is not None, "Failed to load Edge TPU delegate (libedgetpu.so.1)"


def test_edgetpu_model_initialization():
    """Test loading an Edge TPU quantized TFLite model and allocating tensors."""
    import ai_edge_litert.interpreter as tflite

    assert MODEL_PATH.is_file(), f"Model file not found at {MODEL_PATH}"

    delegate = tflite.load_delegate("libedgetpu.so.1")
    interpreter = tflite.Interpreter(model_path=str(MODEL_PATH), experimental_delegates=[delegate])
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    assert len(input_details) == 1
    assert list(input_details[0]["shape"]) == [1, 224, 224, 3]
    assert input_details[0]["dtype"] == np.uint8

    assert len(output_details) == 1
    assert list(output_details[0]["shape"]) == [1, 1001]
    assert output_details[0]["dtype"] == np.uint8


def test_edgetpu_inference_execution():
    """Test running end-to-end inference on the Edge TPU and verify execution speed."""
    import ai_edge_litert.interpreter as tflite

    delegate = tflite.load_delegate("libedgetpu.so.1")
    interpreter = tflite.Interpreter(model_path=str(MODEL_PATH), experimental_delegates=[delegate])
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Create dummy input data
    input_data = np.random.randint(0, 256, size=input_details[0]["shape"], dtype=np.uint8)

    # Warm-up run
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    # Benchmark 20 iterations
    latencies = []
    for _ in range(20):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])
        latencies.append(time.perf_counter() - start)

    avg_ms = (sum(latencies) / len(latencies)) * 1000
    fps = 1000 / avg_ms

    assert output.shape == (1, 1001)
    # Edge TPU inference on MobileNetV2 should be well under 25ms (typically ~4-6ms on RPi 5)
    assert avg_ms < 25.0, f"Edge TPU inference too slow ({avg_ms:.2f} ms), expected < 25 ms"
    print(f"\n[PASS] Edge TPU inference: {avg_ms:.2f} ms ({fps:.1f} FPS)")


def test_picamera2_and_litert_coexistence():
    """Verify that Picamera2 (system package) and LiteRT Edge TPU work simultaneously in Python 3.13."""
    import picamera2
    import ai_edge_litert.interpreter as tflite

    assert hasattr(picamera2, "Picamera2")

    delegate = tflite.load_delegate("libedgetpu.so.1")
    interpreter = tflite.Interpreter(model_path=str(MODEL_PATH), experimental_delegates=[delegate])
    interpreter.allocate_tensors()

    assert interpreter is not None
