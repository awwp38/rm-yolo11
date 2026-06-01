"""
YOLOv11n-Pose 装甲板实时检测
用法: python detect.py [camera_id] [conf]
"""
import cv2, sys, time
from ultralytics import YOLO

# RM 装甲板编号
NAMES = ["B1","B2","B3","B4","B5","BO","BS","R1","R2","R3","R4","R5","RO","RS"]

def main():
    model_path = "runs/train/weights/best.pt"
    camera_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3

    if not __import__('os').path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Run train.py first, or download pretrained: yolo11n-pose.pt")
        model_path = "yolo11n-pose.pt"

    import torch
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Model: {model_path}")
    model = YOLO(model_path, task="pose")
    if device == 0: model.to(device)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Camera {camera_id} not found")
        return

    fps_t = time.time()
    frames = 0

    while True:
        ok, frame = cap.read()
        if not ok: break

        t0 = time.time()
        results = model(frame, conf=conf, imgsz=640, verbose=False, device=device)
        ms = (time.time() - t0) * 1000

        for r in results:
            if r.boxes is None: continue
            for box in r.boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                score = float(box.conf[0])
                name = NAMES[cls_id] if cls_id < len(NAMES) else f"C{cls_id}"
                color = (0,0,255) if name[0]=='R' else (255,0,0)
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                cv2.putText(frame,f"{name} {score:.2f}",(x1,y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX,.5,color,2)

            if r.keypoints is not None:
                for kp in r.keypoints.data[0]:
                    if float(kp[2]) > 0.3:
                        cv2.circle(frame,(int(kp[0]),int(kp[1])),4,(0,0,255),-1)

        frames += 1
        if frames % 30 == 0:
            fps = 30 / (time.time() - fps_t)
            fps_t = time.time()

        cv2.putText(frame,f"{ms:.0f}ms {fps:.0f}FPS",(10,25),
                   cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2)
        cv2.imshow("YOLOv11n Armor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
