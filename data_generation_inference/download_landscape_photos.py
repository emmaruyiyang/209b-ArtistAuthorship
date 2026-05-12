# This code is assist by Claude Code

"""Download realistic landscape photographs for Pipeline B (ControlNet) input.

Pulls N samples from a HuggingFace image dataset and saves them as JPEGs under
--out-dir. Default dataset is a public landscape-photo collection; override with
--hf-dataset to point at any HF image dataset (pass --image-column if the image
field is named something other than 'image').

Examples:
  python download_landscape_photos.py --n 30
  python download_landscape_photos.py --hf-dataset jonathan-roberts1/Places205 --n 50
  python download_landscape_photos.py --hf-dataset some/dataset --image-column img
"""

import argparse
import logging
import sys
from pathlib import Path

from PIL import Image

log = logging.getLogger("download_landscape")

DEFAULT_OUT_DIR = Path(__file__).parent / "data" / "realistic_inputs"
DEFAULT_DATASET = "jonathan-roberts1/Places205"
DEFAULT_SPLIT = "train"
DEFAULT_IMAGE_COLUMN = "image"
DEFAULT_N = 30
LONG_SIDE = 1024


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--n", type=int, default=DEFAULT_N)
    p.add_argument("--hf-dataset", type=str, default=DEFAULT_DATASET,
                   help=f"HF dataset id (default: {DEFAULT_DATASET})")
    p.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    p.add_argument("--image-column", type=str, default=DEFAULT_IMAGE_COLUMN,
                   help="Column name containing the PIL image (default: image)")
    p.add_argument("--filter-keyword", type=str, default=None,
                   help="If set and dataset has a 'label'/'category' field, "
                        "only keep samples whose label string contains this keyword "
                        "(case-insensitive). Useful to bias toward landscapes.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def resize_long_side(img: Image.Image, long_side: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= long_side:
        return img
    if w >= h:
        new_w = long_side
        new_h = int(h * long_side / w)
    else:
        new_h = long_side
        new_w = int(w * long_side / h)
    return img.resize((new_w, new_h), Image.LANCZOS)


def main():
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("`datasets` not installed. Run: pip install datasets")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Streaming %s [split=%s]...", args.hf_dataset, args.split)
    try:
        ds = load_dataset(args.hf_dataset, split=args.split, streaming=True)
    except Exception as e:
        sys.exit(
            f"Failed to load dataset '{args.hf_dataset}'. "
            f"Try a different --hf-dataset (e.g. 'tanganke/landscape', "
            f"'huggan/few-shot-art-painting'). Error: {e}"
        )

    saved = 0
    skipped = 0
    label_features = getattr(ds, "features", None)
    auto_image_col = None  # resolved on first sample
    filter_warned = False
    for sample in ds:
        if saved >= args.n:
            break

        if auto_image_col is None:
            if args.image_column in sample and isinstance(sample[args.image_column], Image.Image):
                auto_image_col = args.image_column
            else:
                for k, v in sample.items():
                    if isinstance(v, Image.Image):
                        auto_image_col = k
                        if k != args.image_column:
                            log.info("auto-detected image column: '%s' (override --image-column='%s' ignored)",
                                     k, args.image_column)
                        break
                if auto_image_col is None:
                    sys.exit(
                        f"No PIL.Image column found in sample. "
                        f"Available keys: {list(sample.keys())}. "
                        f"Pass --image-column explicitly."
                    )

        if args.filter_keyword:
            label_field = None
            label_key = None
            for k in ("label", "category", "scene", "class"):
                if k in sample:
                    label_field = sample[k]
                    label_key = k
                    break
            if label_field is None:
                if not filter_warned:
                    log.warning(
                        "--filter-keyword '%s' set but no label/category/scene/class column "
                        "in dataset (keys: %s); filter is being IGNORED.",
                        args.filter_keyword, list(sample.keys()),
                    )
                    filter_warned = True
            else:
                if isinstance(label_field, int) and label_features is not None:
                    feat = label_features.get(label_key) if hasattr(label_features, "get") else None
                    if feat is not None and hasattr(feat, "int2str"):
                        label_str = feat.int2str(label_field)
                    else:
                        label_str = str(label_field)
                else:
                    label_str = str(label_field)
                if args.filter_keyword.lower() not in label_str.lower():
                    skipped += 1
                    continue

        img = sample[auto_image_col]
        if not isinstance(img, Image.Image):
            log.warning("sample %d image is not a PIL.Image (got %s); skipping",
                        saved, type(img))
            skipped += 1
            continue

        img = img.convert("RGB")
        img = resize_long_side(img, LONG_SIDE)
        out_path = args.out_dir / f"landscape_{saved:03d}.jpg"
        img.save(out_path, "JPEG", quality=92)
        saved += 1

    log.info("Saved %d images to %s (skipped %d)", saved, args.out_dir, skipped)
    if saved == 0:
        sys.exit("No images saved. Check --hf-dataset / --image-column / --filter-keyword.")


if __name__ == "__main__":
    main()
