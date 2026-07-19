# YOLO 目标检测系统 — 项目实施步骤文档

## 一、项目当前状态

### 已完成部分
- 数据集：7 个线下视频抽帧 → 131 张标注图片（YOLO TXT 格式）
- 数据集划分：按视频划分 train(89) / val(27) / test(15)，避免数据泄漏
- 10 个类别：person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light
- 辅助脚本：`extract_offline_videos.py` / `run_labelimg.py` / `prepare_offline_dataset.py`
- 配置文件：`data.yaml` / `predefined_classes.txt`

### 待完成部分
- 补充公共数据集（COCO/CODA）中 airplane、train、truck、boat 四类样本
- YOLO 训练代码框架
- 模型训练与评估
- 实时检测演示系统

---

## 二、环境配置

### 2.1 创建虚拟环境
```bash
cd D:\school_ai_demo\YOLO-target-detection
python -m venv venv
venv\Scripts\activate
```

### 2.2 安装依赖
先用以下内容覆盖 `requirements.txt`：

```text
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
streamlit>=1.25.0
```

然后安装：
```bash
pip install -r requirements.txt
```

> **GPU 训练**：如果本机有 NVIDIA 显卡，前往 https://pytorch.org 获取对应 CUDA 版本的 PyTorch 安装命令。无 GPU 则使用 CPU 训练（速度较慢但仍可运行）。

### 2.3 验证环境
```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -c "from ultralytics import YOLO; print('Ultralytics OK')"
```

---

## 三、代码框架搭建

在项目根目录下创建以下文件。完整的目录结构：

```text
YOLO-target-detection/
├── data.yaml                    # 数据集配置（已有）
├── dataset/                     # 数据集（已有）
│   ├── images/train|val|test/
│   └── labels/train|val|test/
├── train.py                     # 训练脚本（新建）
├── evaluate.py                  # 评估脚本（新建）
├── detect_realtime.py           # 实时检测（新建）
├── app.py                       # Streamlit 演示界面（新建）
├── utils/
│   └── __init__.py
│   └── visualizer.py            # 可视化工具（新建）
├── requirements.txt             # 依赖（已更新）
├── PROJECT_GUIDE.md             # 本文档
└── runs/                        # 训练输出（自动生成，已在 .gitignore）
```

### 3.1 训练脚本 `train.py`

```python
"""YOLO 模型训练脚本"""
from ultralytics import YOLO

def main():
    # 从预训练权重开始（首次运行会自动下载 yolov8n.pt）
    model = YOLO("yolov8n.pt")

    results = model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        name="yolo_detection",
        patience=15,          # 15 个 epoch 无提升则早停
        lr0=0.01,             # 初始学习率
        lrf=0.01,             # 最终学习率因子（lr0 * lrf = 最终学习率）
        warmup_epochs=3,      # 预热轮次
        optimizer="AdamW",    # 优化器
        augment=True,         # 开启数据增强
        device=0,             # GPU 编号；CPU 训练改为 "cpu"
        workers=4,
        seed=42,
        val=True,
    )
    print(f"训练完成，最佳权重保存在: {results.save_dir}")

if __name__ == "__main__":
    main()
```

### 3.2 评估脚本 `evaluate.py`

```python
"""模型评估：在测试集上计算 mAP、P-R 曲线等指标"""
import sys
from pathlib import Path
from ultralytics import YOLO

def main():
    weight_path = "runs/detect/yolo_detection/weights/best.pt"
    if not Path(weight_path).exists():
        print(f"未找到权重 '{weight_path}'，请先运行 train.py")
        sys.exit(1)

    model = YOLO(weight_path)

    # 在测试集上评估
    metrics = model.val(data="data.yaml", split="test", imgsz=640)

    print(f"[结果摘要]")
    print(f"  mAP@0.5:  {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"  Precision:  {metrics.box.mp:.4f}")
    print(f"  Recall:     {metrics.box.mr:.4f}")

    # 按类别输出详细精度
    names = model.names
    print(f"\n[各类别 AP@0.5]")
    for cls_id, ap in enumerate(metrics.box.ap50):
        print(f"   {names.get(cls_id, cls_id):15s}: {ap:.4f}")

    # 在 3 张测试样本上生成预测图
    test_imgs = sorted(Path("dataset/images/test").glob("*.jpg"))
    save_dir = Path("runs/detect/evaluation")
    save_dir.mkdir(parents=True, exist_ok=True)
    for img_path in test_imgs[:3]:
        results = model(img_path, conf=0.25)
        results[0].save(filename=save_dir / img_path.name)
    print(f"\n样例预测图已保存至: {save_dir}")

if __name__ == "__main__":
    main()
```

### 3.3 可视化工具 `utils/visualizer.py`

```python
"""训练过程与检测结果的可视化"""
import cv2
import numpy as np


def draw_boxes(image_bgr, results, class_names, conf_threshold=0.25):
    """在图像上绘制检测框。"""
    image = image_bgr.copy()
    h, w = image.shape[:2]
    if results.boxes is None:
        return image

    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)

    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 128), (128, 128, 0),
        (0, 128, 128), (128, 0, 0)
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
```

### 3.4 实时检测 `detect_realtime.py`

```python
"""使用摄像头进行实时目标检测"""
import cv2
from ultralytics import YOLO

def main():
    weight_path = "runs/detect/yolo_detection/weights/best.pt"
    model = YOLO(weight_path)
    cap = cv2.VideoCapture(0)  # 0 = 默认摄像头

    print("按 'q' 退出")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, conf=0.25, verbose=False)
        annotated = results[0].plot()  # ultralytics 内置绘制方法

        cv2.imshow("YOLO Real-time Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

### 3.5 Streamlit 演示界面 `app.py`

```python
"""Streamlit Web 演示界面：上传图片/视频并展示检测结果"""
import tempfile
from pathlib import Path
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

WEIGHT_PATH = "runs/detect/yolo_detection/weights/best.pt"

st.set_page_config(page_title="YOLO 目标检测", layout="wide")
st.title("YOLO 目标检测演示系统")

@st.cache_resource
def load_model():
    return YOLO(WEIGHT_PATH)

if not Path(WEIGHT_PATH).exists():
    st.error(f"未找到模型权重 `{WEIGHT_PATH}`，请先完成训练。")
    st.stop()

model = load_model()

tab1, tab2 = st.tabs(["图片检测", "视频检测"])

with tab1:
    uploaded = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])
    conf = st.slider("置信度阈值", 0.1, 1.0, 0.3, 0.05)
    if uploaded:
        image = Image.open(uploaded)
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        results = model(image_bgr, conf=conf)
        annotated = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, caption="检测结果", use_container_width=True)

with tab2:
    uploaded = st.file_uploader("上传视频", type=["mp4", "avi", "mov"])
    conf = st.slider("置信度阈值", 0.1, 1.0, 0.3, 0.05, key="vid_conf")
    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            results = model(frame, conf=conf, verbose=False)
            annotated = results[0].plot()
            stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
        cap.release()

st.markdown("---")
st.caption(f"模型权重: {WEIGHT_PATH} | 类别数: {len(model.names)}")
```

---

## 四、开始训练

### 4.1 数据补充（重要）

当前数据集中 **airplane、train、truck、boat** 四类的线下实例为 0，需要从公共数据集补充。

**补充方案：**

1. 从 [COCO 2017](https://cocodataset.org) 下载 train2017 和 annotations
2. 编写筛选脚本，抽取包含 airplane/train/truck/boat 的图片和标注
3. 将 COCO 标注转为 YOLO 格式（归一化 bbox）
4. 按 7:2:1 并入现有 train/val/test

```python
# 示例：COCO → YOLO 类别映射
# person=0, bicycle=1, car=2, motorcycle=3, airplane=4, bus=5, train=6, truck=7, boat=8, traffic light=9
COCO_TO_YOLO = {
    1: 0, 2: 1, 3: 2, 4: 3,  # person, bicycle, car, motorcycle
    5: 4, 6: 5, 7: 6, 8: 7,   # airplane, bus, train, truck
    9: 8, 10: 9                # boat, traffic light
}
```

> 如果暂时无法补充公共数据，可以先用现有 6 类（person, bicycle, car, motorcycle, bus, traffic light）开始训练，后续再扩展。

### 4.2 执行训练

```bash
python train.py
```

训练过程中会自动：
- 打印每个 epoch 的 Loss 和 mAP
- 在 `runs/detect/yolo_detection/` 下保存最佳模型 `best.pt` 和最后一轮 `last.pt`
- 自动生成训练曲线（results.png）、P-R 曲线、混淆矩阵等

### 4.3 训练输出解读

| 指标 | 含义 | 期望值 |
|---|---|---|
| `train/box_loss` | 边框回归损失，持续下降 | → 0 |
| `train/cls_loss` | 分类损失，持续下降 | → 0 |
| `val/box_loss` | 验证集边框损失 | 低于 1.5 |
| `metrics/mAP50(B)` | mAP@0.5 | > 0.6 |
| `metrics/mAP50-95(B)` | mAP@0.5:0.95 | > 0.4 |

---

## 五、评估实验

### 5.1 在测试集上评估

```bash
python evaluate.py
```

输出内容：
- 整体 mAP@0.5、mAP@0.5:0.95、Precision、Recall
- 每个类别的 AP@0.5（可以发现哪些类容易检测，哪些类困难）
- 测试样本预测图片

### 5.2 实验计划（对应课程文档中的四个实验）

#### 实验一：不同 YOLO 模型比较

修改 `train.py` 中的预训练权重，分别训练并对比：

```python
models = {
    "YOLOv5n":  "yolov5nu.pt",   # YOLOv5 nano
    "YOLOv5s":  "yolov5su.pt",   # YOLOv5 small
    "YOLOv8n":  "yolov8n.pt",   # YOLOv8 nano
    "YOLOv8s":  "yolov8s.pt",   # YOLOv8 small
    "YOLOv8m":  "yolov8m.pt",   # YOLOv8 medium
    "YOLOv11n": "yolo11n.pt",   # YOLOv11 nano
}
```

对每个模型训练 100 epochs，记录 mAP@0.5 和推理速度（FPS）。

#### 实验二：数据增强对比

```python
# 修改 train.py，分别做两组对比
# 组 A：关闭数据增强
model.train(data="data.yaml", epochs=100, augment=False, name="no_augment")

# 组 B：开启数据增强（默认开启 mosaic, flip, hsv 等）
model.train(data="data.yaml", epochs=100, augment=True, name="with_augment")
```

对比两组在 val 上的 mAP 和泛化表现。

#### 实验三：训练轮次影响

```python
for epochs in [30, 50, 100, 150]:
    model.train(data="data.yaml", epochs=epochs, name=f"epochs_{epochs}")
```

绘制 epochs vs mAP 曲线，观察何时收敛。

#### 实验四：不同类别难度分析

在 `evaluate.py` 输出的每类 AP@0.5 中，分析：
- 哪些类别（如 car、person）检测效果好 → 样本多、特征明显
- 哪些类别困难 → 样本少、遮挡多、小目标

---

## 六、调参指南

### 6.1 核心可调参数

| 参数 | 默认值 | 说明 | 调参建议 |
|---|---|---|---|
| `lr0` | 0.01 | 初始学习率 | 过拟合→降低；收敛慢→提高；范围 1e-4 ~ 1e-2 |
| `lrf` | 0.01 | 最终学习率因子 | 余弦退火终点 = lr0 × lrf，通常保持 0.01 |
| `batch` | 16 | 批量大小 | GPU 显存允许则增大（32/64），有助于稳定训练 |
| `epochs` | 100 | 训练轮次 | 配合 patience 早停使用 |
| `patience` | 15 | 早停耐心值 | 越小越早停止，避免过拟合 |
| `imgsz` | 640 | 输入尺寸 | 增大可检测小目标但更慢（如 1280） |
| `optimizer` | AdamW | 优化器 | SGD（需调大 lr）或 AdamW（默认稳妥） |
| `weight_decay` | 5e-4 | 权重衰减 | 过拟合时增大到 1e-3 |
| `dropout` | 0.0 | 分类头 dropout | 严重过拟合时设为 0.1~0.2 |

### 6.2 调参流程

```
Step 1: 用默认参数跑一次 baseline，记录 mAP → 基准线
Step 2: 如果 train_loss 下降但 val_loss 上升 → 过拟合 → 增大 weight_decay / 加 dropout / 减小 epochs
Step 3: 如果 train_loss 也下不去 → 欠拟合 → 增大 epochs / 提高 lr0 / 换更大模型（n→s→m）
Step 4: 小目标检测差 → 增大 imgsz（如 1280）
Step 5: 某特定类别差 → 检查该类样本量是否充足，考虑数据补充
```

### 6.3 常见问题排查

| 问题 | 可能原因 | 解决方案 |
|---|---|---|
| mAP 很低 (< 0.3) | 数据太少 / 标注错误 | 检查标签、补充数据 |
| Loss 为 NaN | 学习率过大 | 降低 lr0 一个数量级 |
| 验证集 mAP 远低于训练集 | 过拟合 | 减小 epochs、增大 weight_decay、增强数据 |
| GPU 内存溢出 (CUDA OOM) | batch 太大 | 减小 batch、减小 imgsz |
| 某类 AP 接近 0 | 该类样本太少或标注不一致 | 检查标注、补充样本 |

---

## 七、实时检测与交付

### 7.1 摄像头实时检测

```bash
python detect_realtime.py
```

### 7.2 Web 演示界面

```bash
streamlit run app.py
```

浏览器打开 `http://localhost:8501` 即可使用。

### 7.3 项目交付清单

| 交付物 | 对应文件 | 状态 |
|---|---|---|
| **数据集** | `dataset/` 目录及 `data.yaml` | 已有 |
| **训练代码** | `train.py` | 新建 |
| **评估代码** | `evaluate.py` | 新建 |
| **可视化工具** | `utils/visualizer.py` | 新建 |
| **实时检测** | `detect_realtime.py` | 新建 |
| **演示系统** | `app.py` | 新建 |
| **项目报告** | `.docx`（3000~5000 字） | 待写 |
| **答辩 PPT** | `.pptx` | 待做 |

---

## 八、快速启动清单

按以下顺序操作即可完成项目：

```bash
# 1. 创建环境
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 2. [可选] 补充 COCO 数据到 dataset/（在 data.yaml 中配置）

# 3. 创建上述 4 个新文件：train.py, evaluate.py, utils/visualizer.py, detect_realtime.py, app.py

# 4. 训练
python train.py

# 5. 评估
python evaluate.py

# 6. 查看训练结果
# 打开 runs/detect/yolo_detection/results.png 看 Loss/mAP 曲线

# 7. 实验（调参、不同模型等）
# 修改 train.py 中的参数或预训练权重，重新训练并对比

# 8. 实时检测
python detect_realtime.py

# 9. 演示界面
streamlit run app.py

# 10. 撰写报告 + PPT（参考 runs/ 中的图表）
```

---

## 九、预期效果

基于 131 张线下图片 (6 类实际有效，4 类需补充)，预期：
- **仅用线下数据训练**（6 类）：YOLOv8n 约 50~100 epochs，mAP@0.5 可达 0.6~0.75
- **补充 COCO 后训练**（10 类）：同等条件下 mAP@0.5 可达 0.5~0.7
- **实时检测**：YOLOv8n 在普通笔记本上可达 20~40 FPS

> 数据量较小（131 张），模型容易过拟合。建议尽快补充 COCO 公共数据，将训练集扩充至 500~1000+ 张，效果和泛化能力会显著提升。
