"""对比实验脚本：批量训练 + 评估 + 汇总"""
import gc
import json
import sys
import time
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).parent
RESULT_FILE = ROOT / "experiment_results.json"

EXPERIMENTS = [
    # ===== 实验一：不同模型对比 =====
    # nano 模型 6GB 显存可跑 batch=16；small 模型需降低 batch 避免 mosaic OOM
    {"name": "exp_v8n",        "model": "yolov8n.pt",  "epochs": 100, "augment": True,  "batch": 16, "desc": "YOLOv8n baseline"},
    {"name": "exp_v8s",        "model": "yolov8s.pt",  "epochs": 100, "augment": True,  "batch": 8,  "desc": "YOLOv8s"},
    {"name": "exp_v11n",       "model": "yolo11n.pt",  "epochs": 100, "augment": True,  "batch": 16, "desc": "YOLOv11n"},
    {"name": "exp_v11s",       "model": "yolo11s.pt",  "epochs": 100, "augment": True,  "batch": 8,  "desc": "YOLOv11s"},
    # ===== 实验二：学习率对比 =====
    {"name": "exp_v8n_lr001",  "model": "yolov8n.pt",  "epochs": 100, "augment": True,  "batch": 16, "lr0": 0.001, "desc": "YOLOv8n lr0=0.001"},
    # ===== 实验三：轮次 =====
    {"name": "exp_v8n_ep50",   "model": "yolov8n.pt",  "epochs": 50,  "augment": True,  "batch": 16, "desc": "YOLOv8n 50 epochs"},
    {"name": "exp_v8n_ep150",  "model": "yolov8n.pt",  "epochs": 150, "augment": True,  "batch": 16, "desc": "YOLOv8n 150 epochs"},
    # ===== 实验四：数据增强 =====
    {"name": "exp_v8n_no_aug", "model": "yolov8n.pt", "epochs": 100, "augment": False, "batch": 16,"desc": "YOLOv8n 关闭增强"},
]


def load_done() -> set:
    """返回已完成实验名（跳过失败的，允许重跑）"""
    if RESULT_FILE.exists():
        return {r["name"] for r in json.loads(RESULT_FILE.read_text(encoding="utf-8")) if "error" not in r}
    return set()


def save_result(result: dict) -> None:
    results = []
    if RESULT_FILE.exists():
        results = json.loads(RESULT_FILE.read_text(encoding="utf-8"))
    results.append(result)
    RESULT_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def train_one(exp: dict) -> dict | None:
    name = exp["name"]
    print(f"\n{'=' * 60}")
    print(f"[{name}] {exp['desc']} | model={exp['model']} epochs={exp['epochs']} augment={exp['augment']} batch={exp.get('batch', 16)}")
    print(f"{'=' * 60}")

    t0 = time.time()
    model = YOLO(exp["model"])

    model.train(
        data="data.yaml",
        epochs=exp["epochs"],
        imgsz=640,
        batch=exp.get("batch", 16),
        name=name,
        patience=0,
        augment=exp["augment"],
        device=0,
        workers=4,
        seed=42,
        val=True,
        verbose=False,
        plots=False,
        lr0=exp.get("lr0", 0.01),
        lrf=0.01,
        warmup_epochs=3,
        optimizer="AdamW",
    )
    train_time = time.time() - t0

    best_path = ROOT / "runs" / "detect" / name / "weights" / "best.pt"
    if not best_path.exists():
        print(f"  错误: 未找到权重 {best_path}")
        return None

    model = YOLO(str(best_path))
    metrics = model.val(data="data.yaml", split="test", imgsz=640, verbose=False)

    model_size = best_path.stat().st_size / 1e6
    inference_ms = round(metrics.speed.get("inference", 0), 1)

    result = {
        "name": name,
        "desc": exp["desc"],
        "model": exp["model"],
        "epochs": exp["epochs"],
        "augment": exp["augment"],
        "batch": exp.get("batch", 16),
        "train_time_s": round(train_time, 0),
        "model_size_mb": round(model_size, 1),
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "inference_ms": inference_ms,
        "per_class_ap50": {
            model.names[i]: round(float(ap), 4)
            for i, ap in enumerate(metrics.box.ap50)
        },
    }
    return result


def print_summary():
    data = json.loads(RESULT_FILE.read_text(encoding="utf-8"))
    print("\n" + "=" * 70)
    print("实验一：不同模型对比")
    print("=" * 70)
    print(f"{'模型':<12} {'size(MB)':<10} {'mAP@0.5':<10} {'mAP@0.5:0.95':<14} {'推理(ms)':<10} {'训练(s)':<10}")
    print("-" * 66)
    for r in data:
        if r["name"] in ("exp_v8n", "exp_v8s", "exp_v11n", "exp_v11s"):
            print(f"{r['model']:<12} {r['model_size_mb']:<10.1f} {r['mAP50']:<10.4f} {r['mAP50_95']:<14.4f} {r['inference_ms']:<10.1f} {r['train_time_s']:<10.0f}")

    print("\n实验二：学习率对比")
    print(f"{'学习率(lr0)':<12} {'mAP@0.5':<10} {'mAP@0.5:0.95':<14}")
    for r in data:
        if r["name"] in ("exp_v8n", "exp_v8n_lr001"):
            tag = "0.01 (默认)" if r["name"] == "exp_v8n" else "0.001"
            print(f"{tag:<12} {r['mAP50']:<10.4f} {r['mAP50_95']:<14.4f}")

    print("\n实验三：训练轮次")
    print(f"{'Epochs':<8} {'mAP@0.5':<10} {'mAP@0.5:0.95':<14}")
    for r in sorted(data, key=lambda x: x.get("epochs", 0)):
        if r["name"].startswith("exp_v8n_ep") and "mAP50" in r:
            print(f"{r['epochs']:<8} {r['mAP50']:<10.4f} {r['mAP50_95']:<14.4f}")

    print("\n实验四：各类别 AP@0.5 (YOLOv8n baseline)")
    for r in data:
        if r["name"] == "exp_v8n" and "per_class_ap50" in r:
            for cls, ap in sorted(r["per_class_ap50"].items(), key=lambda x: -x[1]):
                print(f"  {cls:15s}: {ap:.4f}")
                
    print("\n实验五：数据增强对比")
    print(f"{'实验名称':<15} {'lr0':<6} {'mAP@0.5':<10} {'mAP@0.5:0.95':<14} {'Precision':<10} {'Recall':<10} {'推理(ms)':<10} {'训练(s)':<10}")
    print("-" * 85)
    for r in data:
        if r["name"] in ("exp_v8n", "exp_v8n_no_aug", "exp_v8n_tuned") and "mAP50" in r:
            lr0 = 0.01
            print(f"{r['name']:<15} {lr0:<6.3f} {r['mAP50']:<10.4f} {r['mAP50_95']:<14.4f} {r['precision']:<10.4f} {r['recall']:<10.4f} {r['inference_ms']:<10.1f} {r['train_time_s']:<10.0f}")


def main():
    done = load_done()
    print(f"已有 {len(done)} 组结果\n")

    for exp in EXPERIMENTS:
        if exp["name"] in done:
            print(f"跳过已完成: {exp['name']} ({exp['desc']})")
            continue

        try:
            result = train_one(exp)
            if result:
                save_result(result)
                print(f"  完成: mAP@0.5={result['mAP50']:.4f}, 耗时 {result['train_time_s']:.0f}s")
        except KeyboardInterrupt:
            print("\n\n用户中断，已跑结果已保存到 experiment_results.json")
            print_summary()
            sys.exit(0)
        except Exception as e:
            print(f"  实验失败: {e}")
            save_result({"name": exp["name"], "error": str(e)})
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    print_summary()
    print(f"\n全部完成，结果保存在 {RESULT_FILE}")


if __name__ == "__main__":
    main()
