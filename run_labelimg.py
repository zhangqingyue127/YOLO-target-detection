"""从项目中启动 LabelImg，并自动加载抽帧图片及预定义类别。"""

from __future__ import annotations

import sys
import types
from pathlib import Path

IMAGE_DIR = Path(
    r"F:\桌面\学校\学业课程\大三\小学期-实训"
    r"\YOLO目标检测\dataset\offline_frames"
)
CLASS_FILE = Path(__file__).with_name("predefined_classes.txt")


def apply_pyqt5_compatibility_patch() -> None:
    """修复旧版 LabelImg 向新版 PyQt5 绘图接口传入 float 的问题。"""
    import libs

    libs_dir = Path(libs.__file__).parent

    # shape.py：完成一个标注框后绘制标签文字时会执行此处。
    shape_path = libs_dir / "shape.py"
    shape_source = shape_path.read_text(encoding="utf-8")
    old_draw_text = "painter.drawText(min_x, min_y, self.label)"
    new_draw_text = "painter.drawText(QPointF(min_x, min_y), self.label)"
    if old_draw_text not in shape_source:
        raise RuntimeError("无法应用 LabelImg shape.py 的 drawText 兼容补丁")
    shape_source = shape_source.replace(old_draw_text, new_draw_text, 1)

    shape_module = types.ModuleType("libs.shape")
    shape_module.__file__ = str(shape_path)
    shape_module.__package__ = "libs"
    sys.modules["libs.shape"] = shape_module
    exec(compile(shape_source, str(shape_path), "exec"), shape_module.__dict__)

    # canvas.py：绘制矩形和鼠标十字辅助线时会执行这些调用。
    canvas_path = libs_dir / "canvas.py"
    source = canvas_path.read_text(encoding="utf-8")
    replacements = {
        "p.drawRect(left_top.x(), left_top.y(), rect_width, rect_height)": (
            "p.drawRect(QRectF(left_top.x(), left_top.y(), "
            "rect_width, rect_height))"
        ),
        (
            "p.drawLine(self.prev_point.x(), 0, self.prev_point.x(), "
            "self.pixmap.height())"
        ): (
            "p.drawLine(QLineF(self.prev_point.x(), 0, "
            "self.prev_point.x(), self.pixmap.height()))"
        ),
        (
            "p.drawLine(0, self.prev_point.y(), self.pixmap.width(), "
            "self.prev_point.y())"
        ): (
            "p.drawLine(QLineF(0, self.prev_point.y(), "
            "self.pixmap.width(), self.prev_point.y()))"
        ),
    }

    for old, new in replacements.items():
        if old not in source:
            raise RuntimeError(f"无法应用 LabelImg 兼容补丁，未找到代码：{old}")
        source = source.replace(old, new, 1)

    module = types.ModuleType("libs.canvas")
    module.__file__ = str(canvas_path)
    module.__package__ = "libs"
    sys.modules["libs.canvas"] = module
    exec(compile(source, str(canvas_path), "exec"), module.__dict__)


if __name__ == "__main__":
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"抽帧目录不存在：{IMAGE_DIR}")
    if not CLASS_FILE.is_file():
        raise FileNotFoundError(f"标签文件不存在：{CLASS_FILE}")

    apply_pyqt5_compatibility_patch()
    from labelImg.labelImg import main

    # LabelImg 命令行格式：labelImg.py 图片目录 类别文件
    sys.argv = ["labelImg", str(IMAGE_DIR), str(CLASS_FILE)]
    raise SystemExit(main())
