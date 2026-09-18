import sys
import ai_edge_litert.interpreter as tflite
from pathlib import Path

model_path = Path(__file__).resolve().parent.parent / "gui" / "models" / "yolo26n_e100.tflite"
print(f"Model path: {model_path} (exists: {model_path.exists()})", flush=True)

try:
    delegate = tflite.load_delegate("libedgetpu.so.1")
    print("Loaded delegate successfully", flush=True)
    interpreter = tflite.Interpreter(model_path=str(model_path), experimental_delegates=[delegate])
    interpreter.allocate_tensors()
    print("Allocated tensors successfully", flush=True)

    in_det = interpreter.get_input_details()
    out_det = interpreter.get_output_details()

    print("=== Input Details ===", flush=True)
    for d in in_det:
        print(f"Name: {d['name']}, Shape: {d['shape']}, Dtype: {d['dtype']}, Quant: {d['quantization']}", flush=True)

    print("=== Output Details ===", flush=True)
    for d in out_det:
        print(f"Name: {d['name']}, Shape: {d['shape']}, Dtype: {d['dtype']}, Quant: {d['quantization']}", flush=True)

except Exception as e:
    print(f"Error: {e}", flush=True)
