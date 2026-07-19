"""训练过程与检测结果的可视化"""
import cv2
import numpy as np


def draw_boxes(image_bgr, results, class_names, conf_threshold=0.25):
    """在图像上绘制检测框。"""
    image = image_bgr.copy()
    if results.boxes is None:
        return image

    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)

    colors = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 0, 128),
        (128, 128, 0),
        (0, 128, 128),
        (128, 0, 0),
    ]

    for box, conf, cls_id in zip(boxes, confs, classes):
        if conf < conf_threshold:
            continue
        x1, y1, x2, y2 = map(int, box)
        color = colors[cls_id % len(colors)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names.get(cls_id, '?')} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(image, label, (x1 + 2, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image
