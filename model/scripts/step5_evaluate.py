"""
STEP 5 — Local Evaluation (run on macOS after training)
=========================================================
Downloads best.pt from Google Drive, evaluates the model on the local
hold-out test set, prints a full metrics table (mAP50, mAP50-95, precision,
recall per class), and saves a visualisation grid of predicted vs. ground-
truth bounding boxes.

Prerequisites
─────────────
• pip install -r requirements_local.txt
• Place  best.pt  in  <project_root>/model/best.pt
  (download it from Drive:  BrainTumor/train/weights/best.pt)
• The original dataset must still be at
  /Users/bondok/Downloads/Dataset/BrainTumorYolov8/

Usage:
    python scripts/step5_evaluate.py [--model PATH] [--n 16] [--conf 0.25]
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ─── Configuration ────────────────────────────────────────────────────────────

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "model" / "best.pt"
DATASET_ROOT  = Path("/Users/bondok/Downloads/Dataset/BrainTumorYolov8")
DATA_YAML     = DATASET_ROOT / "data.yaml"

CLASS_NAMES   = ["glioma", "meningioma", "pituitary"]
# One distinct colour per class (RGB 0-1)
CLASS_COLORS  = [(0.96, 0.26, 0.21), (0.13, 0.59, 0.95), (0.30, 0.69, 0.31)]

OUTPUT_VIZ    = PROJECT_ROOT / "evaluation_results.png"

# ─── Model loader ─────────────────────────────────────────────────────────────

def load_model(model_path: Path):
    """Load a YOLOv8 model from a .pt checkpoint."""
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("[ERROR] ultralytics not installed. Run: pip install ultralytics")

    if not model_path.exists():
        sys.exit(
            f"[ERROR] Model not found: {model_path}\n"
            "  Download  best.pt  from Google Drive and place it at model/best.pt"
        )

    print(f"[MODEL] Loading {model_path} ...")
    return YOLO(str(model_path))


# ─── Validation metrics ───────────────────────────────────────────────────────

def run_validation(model, data_yaml: Path, split: str = "test") -> dict:
    """
    Run model.val() on the specified split and return a structured metrics dict.
    YOLOv8's val() already computes per-class P, R, mAP50, mAP50-95.
    """
    print(f"\n[EVAL] Running validation on '{split}' split ...")
    results = model.val(data=str(data_yaml), split=split, verbose=False)

    metrics = {
        "mAP50":     float(results.box.map50),
        "mAP50-95":  float(results.box.map),
        "precision": float(results.box.mp),   # mean precision
        "recall":    float(results.box.mr),   # mean recall
        "per_class": {},
    }

    # Per-class breakdowns (arrays indexed by class id)
    for i, name in enumerate(CLASS_NAMES):
        metrics["per_class"][name] = {
            "AP50":      float(results.box.ap50[i]) if i < len(results.box.ap50) else 0.0,
            "AP50-95":   float(results.box.ap[i])   if i < len(results.box.ap)   else 0.0,
            "precision": float(results.box.p[i])    if i < len(results.box.p)    else 0.0,
            "recall":    float(results.box.r[i])    if i < len(results.box.r)    else 0.0,
        }

    return metrics


def print_metrics_table(metrics: dict) -> None:
    """Pretty-print the metrics dict as an ASCII table."""
    col = 12
    sep = "─" * (16 + col * 4)

    print("\n" + "=" * (16 + col * 4))
    print("  Evaluation Metrics (Test Set)")
    print("=" * (16 + col * 4))
    print(f"{'Class':<16}{'AP50':>{col}}{'AP50-95':>{col}}{'Precision':>{col}}{'Recall':>{col}}")
    print(sep)

    for name, m in metrics["per_class"].items():
        print(
            f"{name:<16}"
            f"{m['AP50']:>{col}.4f}"
            f"{m['AP50-95']:>{col}.4f}"
            f"{m['precision']:>{col}.4f}"
            f"{m['recall']:>{col}.4f}"
        )

    print(sep)
    print(
        f"{'MEAN':<16}"
        f"{metrics['mAP50']:>{col}.4f}"
        f"{metrics['mAP50-95']:>{col}.4f}"
        f"{metrics['precision']:>{col}.4f}"
        f"{metrics['recall']:>{col}.4f}"
    )
    print("=" * (16 + col * 4))


# ─── Bounding-box visualisation ───────────────────────────────────────────────

def _load_gt_boxes(label_path: Path) -> list[tuple]:
    """
    Parse a YOLO label file and return a list of (class_id, cx, cy, w, h).
    Returns an empty list if the file does not exist (no-tumor image).
    """
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cid, cx, cy, w, h = int(parts[0]), *map(float, parts[1:5])
        boxes.append((cid, cx, cy, w, h))
    return boxes


def _yolo_to_pixel(cx, cy, w, h, img_w, img_h):
    """Convert normalised YOLO box to pixel (x1, y1, x2, y2)."""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return x1, y1, x2, y2


def visualize_predictions(
    model,
    img_dir: Path,
    label_dir: Path,
    n: int = 16,
    conf: float = 0.25,
    output_path: Path = OUTPUT_VIZ,
) -> None:
    """
    Sample `n` images from img_dir, run inference, and plot a grid showing:
      • Solid coloured boxes  = model predictions (with class + confidence)
      • Dashed white boxes    = ground-truth annotations
    Saves the grid to output_path.
    """
    image_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not image_files:
        print(f"[WARN] No images found in {img_dir}")
        return

    sample = random.sample(image_files, min(n, len(image_files)))

    cols = 4
    rows = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.array(axes).flatten()

    for ax in axes:
        ax.axis("off")

    print(f"\n[VIZ]  Running inference on {len(sample)} sample images ...")

    for idx, img_path in enumerate(sample):
        ax = axes[idx]

        # Load image for display
        img = np.array(Image.open(img_path).convert("RGB"))
        img_h, img_w = img.shape[:2]
        ax.imshow(img)
        ax.set_title(img_path.stem[:28], fontsize=7, pad=2)

        # ── Ground-truth boxes (dashed white) ─────────────────────────────
        gt_boxes = _load_gt_boxes(label_dir / f"{img_path.stem}.txt")
        for (cid, cx, cy, bw, bh) in gt_boxes:
            x1, y1, x2, y2 = _yolo_to_pixel(cx, cy, bw, bh, img_w, img_h)
            rect = mpatches.FancyBboxPatch(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.5, edgecolor="white", facecolor="none",
                linestyle="--",
                boxstyle=mpatches.BoxStyle("Square", pad=0),
            )
            ax.add_patch(rect)

        # ── Predicted boxes (solid, class colour) ─────────────────────────
        results = model.predict(str(img_path), conf=conf, verbose=False)[0]
        for box in results.boxes:
            cid  = int(box.cls.item())
            conf_val = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            color = CLASS_COLORS[cid] if cid < len(CLASS_COLORS) else (1, 1, 0)
            rect  = mpatches.FancyBboxPatch(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor="none",
                boxstyle=mpatches.BoxStyle("Square", pad=0),
            )
            ax.add_patch(rect)

            label = f"{CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else cid} {conf_val:.2f}"
            ax.text(
                x1, max(y1 - 4, 0), label,
                fontsize=7, color="white", fontweight="bold",
                bbox=dict(facecolor=color, alpha=0.75, pad=1, edgecolor="none"),
            )

    # ── Legend ─────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor=c, label=n)
        for n, c in zip(CLASS_NAMES, CLASS_COLORS)
    ] + [
        mpatches.Patch(facecolor="none", edgecolor="white",
                       linestyle="--", label="ground truth", linewidth=1.5)
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=len(legend_handles), fontsize=9, framealpha=0.8)

    plt.suptitle("Brain Tumor Detection — Predictions vs Ground Truth",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ]  Saved → {output_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained YOLOv8 model locally")
    parser.add_argument("--model", default=str(DEFAULT_MODEL),  help="Path to best.pt")
    parser.add_argument("--n",     default=16, type=int,        help="Images to visualise")
    parser.add_argument("--conf",  default=0.25, type=float,    help="Confidence threshold")
    args = parser.parse_args()

    print("=" * 60)
    print("  Brain Tumor — Step 5: Local Evaluation")
    print("=" * 60)

    model = load_model(Path(args.model))

    # ── Validation metrics ──────────────────────────────────────────────────
    if not DATA_YAML.exists():
        # Build a minimal data.yaml pointing at the local dataset
        yaml_text = (
            f"path: {DATASET_ROOT}\n"
            "train: train/images\nval: valid/images\ntest: test/images\n"
            "nc: 3\nnames: ['glioma', 'meningioma', 'pituitary']\n"
        )
        tmp_yaml = PROJECT_ROOT / "data_eval.yaml"
        tmp_yaml.write_text(yaml_text)
        yaml_path = tmp_yaml
    else:
        yaml_path = DATA_YAML

    metrics = run_validation(model, yaml_path)
    print_metrics_table(metrics)

    # ── Visualisation ───────────────────────────────────────────────────────
    test_img_dir = DATASET_ROOT / "test" / "images"
    test_lbl_dir = DATASET_ROOT / "test" / "labels"

    if test_img_dir.exists():
        visualize_predictions(
            model,
            img_dir=test_img_dir,
            label_dir=test_lbl_dir,
            n=args.n,
            conf=args.conf,
        )
    else:
        print(f"[WARN] Test images not found at {test_img_dir} — skipping visualisation")

    print("\n✓  Evaluation complete.")


if __name__ == "__main__":
    main()
