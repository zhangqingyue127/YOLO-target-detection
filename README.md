# YOLO Target Detection

本项目用于道路交通场景的 YOLO 目标检测，仓库包含线下采集数据的抽帧图片、LabelImg 标注、数据集配置及相关预处理脚本。

## 数据采集与预处理

项目数据来源包括公共数据集 **COCO**、**CODA** 和线下采集数据。考虑到公共数据集的许可证、体积及发布方式，本仓库当前仅收录自行整理的线下采集子集；COCO 与 CODA 请从各自官方渠道获取，并按照项目的 10 类类别体系完成筛选或映射。

线下数据共采集 **7 个 MP4 视频**，使用 `extract_offline_videos.py` 递归读取视频，默认每隔 30 帧抽取一张 JPEG 图片，JPEG 质量为 95，共得到并标注 **131 帧**。标注工具采用 **LabelImg**，标签保存为 YOLO TXT 格式。

类别定义如下：

| ID | 类别 |
|---:|---|
| 0 | person |
| 1 | bicycle |
| 2 | car |
| 3 | motorcycle |
| 4 | airplane |
| 5 | bus |
| 6 | train |
| 7 | truck |
| 8 | boat |
| 9 | traffic light |

线下标注共包含 **1181 个目标实例**。其中 `airplane`、`train`、`truck` 和 `boat` 在线下采集子集中暂无实例，需要通过 COCO/CODA 等公共数据补充。各类别的线下实例数量为：

| 类别 | 实例数 |
|---|---:|
| person | 280 |
| bicycle | 43 |
| car | 387 |
| motorcycle | 184 |
| airplane | 0 |
| bus | 42 |
| train | 0 |
| truck | 0 |
| boat | 0 |
| traffic light | 245 |

## 数据集划分

为避免同一视频的相邻帧同时出现在训练集和验证/测试集中造成数据泄漏，本项目按视频划分，而不是随机打散图片：

| 数据集 | 来源视频 | 图片/标签数量 |
|---|---|---:|
| train | video1–video5 | 89 |
| val | video6 | 27 |
| test | video7 | 15 |
| 合计 | video1–video7 | 131 |

目录结构：

```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

每张图片与标签文件同名，例如：

```text
dataset/images/train/video1_frame_000000.jpg
dataset/labels/train/video1_frame_000000.txt
```

## 脚本说明

- `extract_offline_videos.py`：递归查找 MP4 视频并按固定帧间隔抽帧，兼容 Windows 中文路径和重复运行。
- `run_labelimg.py`：使用预定义类别启动 LabelImg，并包含旧版 LabelImg 与新版 PyQt5 的兼容处理。
- `prepare_offline_dataset.py`：校验 YOLO 标签并按视频划分、整理线下数据集。
- `predefined_classes.txt`：LabelImg 使用的类别列表。
- `data.yaml`：YOLO 数据集配置。

## 环境与使用

安装依赖：

```bash
pip install -r requirements.txt
```

抽取线下视频帧：

```bash
python extract_offline_videos.py
```

启动 LabelImg：

```bash
python run_labelimg.py
```

重新校验并整理线下数据：

```bash
python prepare_offline_dataset.py
```

数据路径常量目前按 Windows 本地采集环境配置；在其他机器运行抽帧或整理脚本前，请修改脚本顶部的输入、输出目录。

## 数据与隐私说明

- 仓库不包含原始 MP4 视频。
- 线下道路场景可能包含行人、车辆及车牌，使用数据前应确认采集授权、隐私处理和适用范围。
- COCO 与 CODA 数据不在本仓库重复分发，请遵守其官方许可证和使用条款。
