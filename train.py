"""
YOLOv11n-Pose 装甲板训练
用法: python train.py [data_yaml]
"""
from ultralytics import YOLO
import sys

DATA = sys.argv[1] if len(sys.argv) > 1 else "../rm-aiming-sim/data/dataset/armor.yaml"

model = YOLO("yolo11n-pose.pt")  # 从预训练开始

model.train(
    data=DATA,
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    workers=8,
    patience=10,
    save=True,
    project="runs",
    name="train",
    exist_ok=True,
)
