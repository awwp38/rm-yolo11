"""
导出 ONNX 模型
用法: python export.py [model_path]
"""
from ultralytics import YOLO
import sys

model_path = sys.argv[1] if len(sys.argv) > 1 else "runs/train/weights/best.pt"
print(f"Exporting: {model_path}")
model = YOLO(model_path)
model.export(format="onnx", imgsz=640, simplify=True, opset=12)
print("Done → best.onnx")
