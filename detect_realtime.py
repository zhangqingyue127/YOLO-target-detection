"""实时目标检测 — 支持摄像头(默认) / 视频文件"""
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO


def main():
    weight_path = "runs/exp_v8n_lr001/weights/best.pt"
    if not Path(weight_path).exists():
        print(f"错误: 未找到模型权重 {weight_path}")
        return

    model = YOLO(weight_path)

    # ── 判断输入源 ──
    if len(sys.argv) > 1:
        source = sys.argv[1]
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"错误: 无法打开视频文件 {source}")
            return
        # 自动生成输出文件名
        src = Path(source)
        out_path = src.parent / f"{src.stem}_detected{src.suffix}"
        fps_in = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps_in, (w, h))
        print(f"处理视频: {src.name} ({w}x{h}, {fps_in:.1f}fps)")
        print(f"结果保存: {out_path.name}")
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        writer = None
        print("摄像头模式 — 按 'q' 退出")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, conf=0.25, verbose=False)
        annotated = results[0].plot()

        if writer:
            writer.write(annotated)
        else:
            cv2.imshow("YOLO Real-time Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
