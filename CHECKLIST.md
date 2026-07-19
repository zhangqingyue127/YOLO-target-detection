# 项目完成清单

## 第一阶段：环境搭建（Day 1）

- [x] 安装 Python 3.8+
- [x] 安装依赖 `pip install -r requirements.txt`
- [x] 验证 PyTorch 和 CUDA
- [x] 验证 Ultralytics

## 第二阶段：数据处理（Day 1-2）

- [x] 确认 `data.yaml` 中类别数量和名称正确
- [x] 确认 `dataset/` 目录结构完整（images+labels × train/val/test）
- [x] 方案 A：改为 6 类训练，删除 airplane/train/truck/boat，标签 ID 重映射
- [x] 方案 B：从 COCO 补充数据（下载标注 → CDN 按需下载图片 → 转 YOLO 格式 → 合并）

## 第三阶段：模型训练（Day 3-4）

- [x] 仅线下数据 Baseline 训练（YOLOv8n, 89 张, 56 epochs 早停）
- [x] 补充 COCO 后训练（YOLOv8n, 1047 张, 100 epochs）
- [x] 训练曲线和最佳模型已保存

## 第四阶段：模型评估（Day 4-5）

- [x] 运行评估，获取测试集指标
- [x] 记录 mAP@0.5、mAP@0.5:0.95、Precision、Recall
- [x] 查看每个类别的 AP

## 第五阶段：对比实验（Day 5-6）

- [ ] **实验一：不同模型对比**
- [ ] **实验二：数据增强对比**
- [ ] **实验三：训练轮次影响**
- [ ] **实验四：类别难度分析**

## 第六阶段：调参优化（Day 5-6）

- [ ] 根据评估结果判断问题
- [ ] 调整学习率/batch size/imgsz
- [ ] 记录调参结果

## 第七阶段：实时检测系统（Day 7-8）

- [ ] 运行 `python detect_realtime.py`
- [ ] 运行 `streamlit run app.py`

## 第八阶段：文档与交付（Day 9-10）

- [ ] 整理训练截图和检测效果截图
- [ ] 提取实验数据，制表对比
- [ ] 撰写项目报告（3000~5000 字）
- [ ] 制作答辩 PPT

---

## 进度日志

### 2026-07-19 — 环境搭建 + 数据准备

**环境**
- Python 3.12.7, PyTorch 2.5.1+cu121, ultralytics 8.4.101
- GPU: NVIDIA RTX 3060 Laptop (6GB), CUDA 12.7
- 显存占用：训练时 ~2.5GB / 6GB

**方案 A：6 类筛选**
- 原 10 类中 airplane/train/truck/boat 无实例，删除
- 保留 person(0), bicycle(1), car(2), motorcycle(3), bus(4), traffic light(5)
- 修改 data.yaml (nc:6)，131 个标签文件 ID 重映射 (5→4, 9→5)

**方案 B：COCO 数据补充**
- 下载 COCO 2017 标注文件 (instances_train2017.json, instances_val2017.json)
- 筛选 6 类目标图片 1188 张，自 CDN 按需下载
- 统一 resize 到 640×640，按 8:2 划分 train/val 并入 dataset/
- 合并后: train 1047 张, val 272 张, test 15 张（test 保持不变）

### 2026-07-19 — 模型训练

**Baseline（仅线下 89 张训练图）**
- 模型: YOLOv8n, 56 epochs（patience=15 早停，best=epoch 41）
- 训练耗时: ~1.6 分钟 (GPU)
- 测试集 mAP@0.5: **0.396**

| 类别 | AP@0.5 |
|---|---|
| car | 0.699 |
| bus | 0.758 (仅 8 实例) |
| motorcycle | 0.324 |
| person | 0.149 |
| traffic light | 0.049 |
| bicycle | 0.000 (未检测) |

**补充 COCO 后（1047 张训练图）**
- 模型: YOLOv8n, 100 epochs
- 训练耗时: ~17 分钟 (GPU)
- 测试集 mAP@0.5: **0.543** (+37%)
- 测试集 mAP@0.5:0.95: **0.246** (+30%)

| 类别 | 仅线下 | +COCO | 提升 |
|---|---|---|---|
| person | 0.149 | **0.754** | +405% |
| bicycle | 0.000 | 0.049 | 可检测 |
| car | 0.699 | **0.733** | +5% |
| motorcycle | 0.324 | **0.617** | +90% |
| traffic light | 0.049 | **0.561** | +1045% |

**权重路径**
- Baseline: `runs/detect/yolo_detection/weights/best.pt`
- COCO 版: `runs/detect/yolo_detection-3/weights/best.pt`

### 待办
- 对比实验（YOLOv8s、数据增强、epochs 影响）
- 实时检测系统验证
- 项目报告 + PPT
