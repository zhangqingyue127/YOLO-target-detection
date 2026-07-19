"""使用摄像头进行实时目标检测"""
from pathlib import Path

import cv2
from ultralytics import YOLO


def main():
    weight_path = "runs/detect/yolo_detection/weights/best.pt"
    if not Path(weight_path).exists():
        print(f"未找到模型权重 '{weight_path}'，请先运行 train.py 完成训练。")
        return

    model = YOLO(weight_path)
    cap = cv2.VideoCapture(0)

    print("按 'q' 退出")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, conf=0.25, verbose=False)
        annotated = results[0].plot()

        cv2.imshow("YOLO Real-time Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
