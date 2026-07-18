from __future__ import annotations

import shutil
from pathlib import Path


SOURCE_IMAGES = Path(
    r"F:\桌面\学校\学业课程\大三\小学期-实训"
    r"\YOLO目标检测\dataset\offline_frames"
)
SOURCE_LABELS = Path(
    r"F:\桌面\学校\学业课程\大三\小学期-实训"
    r"\YOLO目标检测\dataset\offline_labels"
)
TARGET_DATASET = Path(__file__).parent / "dataset"

VIDEO_SPLITS = {
    "video1": "train",
    "video2": "train",
    "video3": "train",
    "video4": "train",
    "video5": "train",
    "video6": "val",
    "video7": "test",
}
CLASS_COUNT = 10


def validate_label(label_path: Path) -> None:
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_number} 不是 5 列 YOLO 格式")
        class_id = int(parts[0])
        coordinates = [float(value) for value in parts[1:]]
        if not 0 <= class_id < CLASS_COUNT:
            raise ValueError(f"{label_path}:{line_number} 类别越界：{class_id}")
        if not all(0.0 <= value <= 1.0 for value in coordinates):
            raise ValueError(f"{label_path}:{line_number} 坐标未归一化")


def prepare_dataset() -> dict[str, int]:
    if not SOURCE_IMAGES.is_dir() or not SOURCE_LABELS.is_dir():
        raise FileNotFoundError("找不到本地抽帧图片目录或标注目录")

    counts = {"train": 0, "val": 0, "test": 0}
    images = sorted(SOURCE_IMAGES.rglob("*.jpg"))

    for image_path in images:
        video_name = image_path.parent.name
        if video_name not in VIDEO_SPLITS:
            raise ValueError(f"没有为视频目录配置数据划分：{video_name}")

        split = VIDEO_SPLITS[video_name]
        label_path = SOURCE_LABELS / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"图片缺少标签：{image_path}")
        validate_label(label_path)

        image_target = TARGET_DATASET / "images" / split / image_path.name
        label_target = TARGET_DATASET / "labels" / split / label_path.name
        image_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, image_target)
        shutil.copy2(label_path, label_target)
        counts[split] += 1

    if sum(counts.values()) != 131:
        raise RuntimeError(f"预期整理 131 张图片，实际为 {sum(counts.values())} 张")
    return counts


if __name__ == "__main__":
    result = prepare_dataset()
    print(f"整理完成：train={result['train']}，val={result['val']}，test={result['test']}")
