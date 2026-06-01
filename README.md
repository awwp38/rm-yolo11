# rm-yolo11 — RoboMaster YOLOv11n 装甲板检测

轻量级 Python 项目，使用 YOLOv11n-Pose 做装甲板检测。
目标：<5ms GPU 推理，直接输出 4 个装甲板角点。

## 结构
```
train.py          # 训练脚本
detect.py         # 实时摄像头检测
export.py         # ONNX 导出
config.yaml       # 训练配置
```

## 快速开始
```bash
pip install ultralytics opencv-python torch
python detect.py
```
