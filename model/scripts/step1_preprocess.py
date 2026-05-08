"""
STEP 1 — Local Preprocessing (run on macOS)
============================================
Reads the BrainTumorYolov8 dataset, validates every image and label file,
auto-converts polygon annotations to bounding-box format when needed,
prints a class-distribution table, and packages everything into a ZIP
ready to upload to Google Drive.

Usage:
    python scripts/step1_preprocess.py

Output:
    brain_tumor_dataset.zip   (created in the project root)
"""

import os
import sys
import shutil
import zipfile
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# ─── Configuration ────────────────────────────────────────────────────────────

# Raw dataset root (the folder that contains train/, valid/, test/)
DATASET_ROOT = Path("/Users/bondok/Downloads/Dataset/BrainTumorYolov8")

# Where to write the final ZIP (project root)
OUTPUT_ZIP = Path(__file__).resolve().parent.parent / "brain_tumor_dataset.zip"

# The three canonical splits and where images / labels live inside each
SPLITS = {
    "train": ("train/images", "train/labels"),
    "valid": ("valid/images", "valid/labels"),
    "test":  ("test/images",  "test/labels"),
}

CLASS_NAMES = ["glioma", "meningioma", "pituitary"]
NC = len(CLASS_NAMES)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def polygon_to_bbox(values: list[float]) -> tuple[float, float, float, float]:
    """
    Convert a flat list of normalised polygon vertices [x1,y1,x2,y2,...] to a
    YOLO bounding-box tuple (cx, cy, w, h), all still normalised 0-1.
    """
    xs = values[0::2]   # every even index
    ys = values[1::2]   # every odd index
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w  = x_max - x_min
    h  = y_max - y_min
    return cx, cy, w, h


def convert_label_file(src: Path, dst: Path) -> dict[int, int]:
    """
    Read a YOLO label file, detect format (bbox vs polygon), convert polygons
    to bboxes, and write the result.  Returns a dict {class_id: count}.
    """
    counts: dict[int, int] = defaultdict(int)
    dst.parent.mkdir(parents=True, exist_ok=True)

    with src.open() as f:
        raw_lines = [l.strip() for l in f if l.strip()]

    out_lines: list[str] = []
    for line in raw_lines:
        parts = line.split()
        class_id = int(parts[0])
        coords = [float(v) for v in parts[1:]]

        if len(coords) == 4:
            # Already in bbox format: cx cy w h
            cx, cy, w, h = coords
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            # Polygon segmentation format: x1 y1 x2 y2 ...
            cx, cy, w, h = polygon_to_bbox(coords)
        else:
            print(f"  [WARN] Unexpected coord count ({len(coords)}) in {src} — skipping line")
            continue

        # Clamp to [0, 1] to guard against tiny floating-point overflows
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        w  = min(max(w,  0.0), 1.0)
        h  = min(max(h,  0.0), 1.0)

        out_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        counts[class_id] += 1

    dst.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
    return counts


def validate_image(img_path: Path) -> tuple[bool, str]:
    """Return (ok, message). Opens the image with PIL to catch corrupt files."""
    try:
        with Image.open(img_path) as img:
            img.verify()          # checks file integrity without fully decoding
        return True, ""
    except Exception as exc:
        return False, str(exc)


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  Brain Tumor — Step 1: Preprocess & Package")
    print("=" * 60)

    if not DATASET_ROOT.exists():
        sys.exit(f"[ERROR] Dataset not found at {DATASET_ROOT}")

    # Accumulate per-split statistics for the distribution table
    # stats[split][class_id] = annotation_count
    stats: dict[str, dict[int, int]] = {s: defaultdict(int) for s in SPLITS}
    image_counts: dict[str, int] = {}
    bad_images: list[str] = []

    # Use a temporary staging directory so we never modify the source
    with tempfile.TemporaryDirectory() as staging_root:
        staging = Path(staging_root) / "brain_tumor_dataset"

        for split_name, (img_rel, lbl_rel) in SPLITS.items():
            img_dir = DATASET_ROOT / img_rel
            lbl_dir = DATASET_ROOT / lbl_rel

            if not img_dir.exists():
                print(f"[WARN] Missing split directory: {img_dir} — skipping")
                continue

            image_files = sorted(img_dir.glob("*.*"))
            image_counts[split_name] = len(image_files)

            print(f"\n[{split_name}]  {len(image_files)} images  →  validating + copying ...")

            dst_img_dir = staging / img_rel
            dst_lbl_dir = staging / lbl_rel
            dst_img_dir.mkdir(parents=True, exist_ok=True)
            dst_lbl_dir.mkdir(parents=True, exist_ok=True)

            for img_path in tqdm(image_files, unit="img", leave=False):
                # ── Validate image ────────────────────────────────────────────
                ok, msg = validate_image(img_path)
                if not ok:
                    bad_images.append(f"{img_path}: {msg}")
                    continue

                shutil.copy2(img_path, dst_img_dir / img_path.name)

                # ── Convert label ─────────────────────────────────────────────
                stem = img_path.stem
                lbl_src = lbl_dir / f"{stem}.txt"
                lbl_dst = dst_lbl_dir / f"{stem}.txt"

                if lbl_src.exists():
                    counts = convert_label_file(lbl_src, lbl_dst)
                    for cid, cnt in counts.items():
                        stats[split_name][cid] += cnt
                else:
                    # Image with no annotations = background / no-tumor image
                    lbl_dst.write_text("")

        # ── Write data.yaml ───────────────────────────────────────────────────
        data_yaml_path = staging / "data.yaml"
        names_str = "[" + ", ".join(f"'{n}'" for n in CLASS_NAMES) + "]"
        data_yaml_path.write_text(
            f"# YOLOv8 dataset configuration\n"
            f"path: .            # dataset root (relative to this file or absolute)\n"
            f"train: train/images\n"
            f"val:   valid/images\n"
            f"test:  test/images\n\n"
            f"nc: {NC}           # number of classes\n"
            f"names: {names_str}\n"
        )

        # ── Zip staging directory ─────────────────────────────────────────────
        print(f"\nPackaging into {OUTPUT_ZIP} ...")
        with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for file in tqdm(list(staging.rglob("*")), unit="file", leave=False):
                if file.is_file():
                    zf.write(file, file.relative_to(staging_root))

    # ─── Print class distribution table ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Class Distribution")
    print("=" * 60)
    col_w = 14
    header = f"{'Class':<14}" + "".join(f"{s:>{col_w}}" for s in SPLITS)
    print(header)
    print("-" * (14 + col_w * len(SPLITS)))
    for cid, name in enumerate(CLASS_NAMES):
        row = f"{name:<14}"
        for split_name in SPLITS:
            row += f"{stats[split_name][cid]:>{col_w}}"
        print(row)
    print("-" * (14 + col_w * len(SPLITS)))
    totals = f"{'TOTAL (imgs)':<14}"
    for split_name in SPLITS:
        totals += f"{image_counts.get(split_name, 0):>{col_w}}"
    print(totals)

    if bad_images:
        print(f"\n[WARN] {len(bad_images)} corrupt/unreadable image(s) were skipped:")
        for msg in bad_images[:10]:
            print(f"  {msg}")

    zip_mb = OUTPUT_ZIP.stat().st_size / 1_048_576
    print(f"\n✓  Output: {OUTPUT_ZIP}  ({zip_mb:.1f} MB)")
    print("   Upload this file to Google Drive, then run step2_upload.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
