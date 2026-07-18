from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


INPUT_DIR = Path(
    r"F:\桌面\学校\学业课程\大三\小学期-实训"
    r"\YOLO目标检测\dataset\offline_dataset"
)
OUTPUT_DIR = Path(
    r"F:\桌面\学校\学业课程\大三\小学期-实训"
    r"\YOLO目标检测\dataset\offline_frames"
)

FRAME_STEP = 30
JPEG_QUALITY = 95


@dataclass(slots=True)
class VideoResult:
    video_path: Path
    output_dir: Path
    success: bool
    total_frames: int = 0
    fps: float = 0.0
    duration_seconds: float = 0.0
    saved_frames: int = 0
    skipped_frames: int = 0
    error: str = ""


def save_image_unicode(output_path: Path, frame: np.ndarray) -> None:
    """使用 imencode + tofile，在 Windows 中文路径下可靠保存 JPEG。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
    )
    if not success:
        raise RuntimeError(f"JPEG 编码失败：{output_path}")

    encoded.tofile(str(output_path))
    if not output_path.exists() or output_path.stat().st_size <= 0:
        raise OSError(f"图片写入失败或文件为空：{output_path}")


def get_output_dir(video_path: Path) -> Path:
    """保留输入目录层级，确保不同子目录中的同名视频不会互相覆盖。"""
    relative_video = video_path.relative_to(INPUT_DIR)
    return OUTPUT_DIR / relative_video.with_suffix("")


def extract_video_frames(video_path: Path) -> VideoResult:
    output_dir = get_output_dir(video_path)
    result = VideoResult(
        video_path=video_path,
        output_dir=output_dir,
        success=False,
    )
    capture: cv2.VideoCapture | None = None

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError("无法打开视频")

        result.total_frames = max(
            0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        )
        result.fps = max(0.0, float(capture.get(cv2.CAP_PROP_FPS)))
        if result.fps > 0:
            result.duration_seconds = result.total_frames / result.fps

        frame_index = 0
        frames_read = 0
        while True:
            read_ok, frame = capture.read()
            if not read_ok:
                break

            frames_read += 1
            if frame_index % FRAME_STEP == 0:
                filename = f"{video_path.stem}_frame_{frame_index:06d}.jpg"
                output_path = output_dir / filename
                if output_path.exists() and output_path.stat().st_size > 0:
                    result.skipped_frames += 1
                else:
                    save_image_unicode(output_path, frame)
                    result.saved_frames += 1

            frame_index += 1

        if frames_read == 0:
            raise RuntimeError("视频已打开，但未能读取任何帧")

        result.success = True
    except Exception as error:
        result.error = f"{type(error).__name__}: {error}"
    finally:
        if capture is not None:
            capture.release()

    return result


def find_mp4_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在：{input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"输入路径不是目录：{input_dir}")

    return sorted(
        (
            path
            for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() == ".mp4"
        ),
        key=lambda path: str(path).casefold(),
    )


def print_video_result(result: VideoResult) -> None:
    status = "成功" if result.success else "失败"
    print(f"  状态：{status}")
    print(f"  原始总帧数：{result.total_frames}")
    print(f"  FPS：{result.fps:.2f}")
    print(f"  视频时长：{result.duration_seconds:.2f} 秒")
    print(f"  本次实际保存：{result.saved_frames} 张")
    print(f"  已存在并跳过：{result.skipped_frames} 张")
    print(f"  输出目录：{result.output_dir}")
    if result.error:
        print(f"  错误：{result.error}")


def main() -> None:
    if FRAME_STEP <= 0:
        raise ValueError("FRAME_STEP 必须大于 0")
    if not 0 <= JPEG_QUALITY <= 100:
        raise ValueError("JPEG_QUALITY 必须在 0 到 100 之间")

    try:
        video_files = find_mp4_files(INPUT_DIR)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as error:
        print(f"初始化失败：{type(error).__name__}: {error}")
        return

    print(f"找到的视频数量：{len(video_files)}")
    print(f"抽帧间隔：每 {FRAME_STEP} 帧保存一张")
    print(f"JPEG 质量：{JPEG_QUALITY}")
    print()

    results: list[VideoResult] = []
    for index, video_path in enumerate(video_files, start=1):
        print(f"[{index}/{len(video_files)}] 处理：{video_path}")
        result = extract_video_frames(video_path)
        results.append(result)
        print_video_result(result)
        print()

    successful = sum(result.success for result in results)
    failed = len(results) - successful
    total_saved = sum(result.saved_frames for result in results)

    print("=" * 60)
    print("全部视频处理完成")
    print(f"找到的视频数量：{len(video_files)}")
    print(f"成功处理数量：{successful}")
    print(f"失败数量：{failed}")
    print(f"本次总共保存的图片数量：{total_saved}")
    print(f"输出根目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
