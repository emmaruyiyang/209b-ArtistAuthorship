# This code is assist by Claude Code

"""Build 4x2 preview collages for each artist in the SDXL output folders."""
from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent / "data"
OUT_ROOT = ROOT / "_collages"
ARTISTS = ["leonardo_da_vinci", "monet", "picasso", "rembrandt", "van_gogh"]

CELL = 256
COLS, ROWS = 4, 2
PAD = 6
TITLE_H = 32


def make_canvas(title: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    w = COLS * CELL + (COLS + 1) * PAD
    h = TITLE_H + ROWS * CELL + (ROWS + 1) * PAD
    canvas = Image.new("RGB", (w, h), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    draw.text((PAD * 2, (TITLE_H - 18) // 2), title, fill=(240, 240, 240), font=font)
    return canvas, draw


def paste_cell(canvas: Image.Image, img_path: Path, col: int, row: int) -> None:
    img = Image.open(img_path).convert("RGB")
    img.thumbnail((CELL, CELL), Image.LANCZOS)
    x = PAD + col * (CELL + PAD) + (CELL - img.width) // 2
    y = TITLE_H + PAD + row * (CELL + PAD) + (CELL - img.height) // 2
    canvas.paste(img, (x, y))


def build_raw_collage(artist_dir: Path, out_path: Path, title: str) -> bool:
    samples = sorted(artist_dir.glob("gen_*.png"))[: COLS * ROWS]
    if len(samples) < COLS * ROWS:
        print(f"WARN: {artist_dir} only has {len(samples)} samples, skipping")
        return False
    canvas, _ = make_canvas(title)
    for i, p in enumerate(samples):
        paste_cell(canvas, p, col=i % COLS, row=i // COLS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"wrote {out_path}")
    return True


def _canny_is_empty(path: Path) -> bool:
    """True if the canny edge map has effectively no edges (all-black)."""
    img = Image.open(path).convert("L")
    return img.getextrema()[1] == 0


def build_canny_collage(artist_dir: Path, out_path: Path, title: str) -> bool:
    canny_dir = artist_dir / "_canny"
    if not canny_dir.exists():
        print(f"WARN: {canny_dir} missing, skipping")
        return False
    all_canny = sorted(canny_dir.glob("landscape_*.png"))
    canny_files: list[Path] = []
    for p in all_canny:
        if _canny_is_empty(p):
            print(f"  skip empty canny: {p.name}")
            continue
        canny_files.append(p)
        if len(canny_files) == COLS:
            break
    if len(canny_files) < COLS:
        print(f"WARN: {canny_dir} only has {len(canny_files)} non-empty canny images, skipping")
        return False
    canvas, _ = make_canvas(title)
    for col, canny_path in enumerate(canny_files):
        stem = canny_path.stem  # landscape_000_jpg
        gen_path = artist_dir / f"gen_{stem}__00.png"
        if not gen_path.exists():
            print(f"WARN: missing {gen_path}, skipping {artist_dir.name}")
            return False
        paste_cell(canvas, canny_path, col=col, row=0)
        paste_cell(canvas, gen_path, col=col, row=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"wrote {out_path}")
    return True


def main() -> None:
    raw_root = ROOT / "generated_raw_sdxl"
    canny_root = ROOT / "generated_canny_raw_sdxl"
    for artist in ARTISTS:
        build_raw_collage(
            raw_root / artist,
            OUT_ROOT / "generated_raw_sdxl" / f"{artist}.png",
            f"generated_raw_sdxl / {artist}",
        )
        build_canny_collage(
            canny_root / artist,
            OUT_ROOT / "generated_canny_raw_sdxl" / f"{artist}.png",
            f"generated_canny_raw_sdxl / {artist}  (top: canny, bottom: generated)",
        )


if __name__ == "__main__":
    main()
