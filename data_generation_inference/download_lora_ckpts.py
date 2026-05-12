# This code is assist by Claude Code

"""Download per-artist SDXL LoRA checkpoints from Google Drive — pick the
largest savepoint per artist *before* downloading anything.

Strategy:
  1. List the Drive folder tree via gdown's skip_download API (no bytes pulled).
  2. For each top-level subfolder (=artist), find its checkpoint-<N> children,
     pick the largest N that contains pytorch_lora_weights.safetensors.
  3. Download only that single file (plus artist_config.json from the root).
  4. Artists with no usable weights are skipped with a WARN; final summary
     prints which artists were kept vs. skipped. Script always exits 0 if at
     least one artist was kept.

Result layout:
  <project>/lora_ckpts/
    <slug>/pytorch_lora_weights.safetensors
    artist_config.json
"""

import argparse
import logging
import re
import sys
from collections import namedtuple
from pathlib import Path

import gdown
from gdown.download_folder import _download_and_parse_google_drive_link

import requests

log = logging.getLogger("download_lora_ckpts")

DEFAULT_DRIVE_URL = (
    "https://drive.google.com/drive/folders/1l9Gn_v0vcHLlWXGybwgBqU7qCOfCxk5Y"
)
DEFAULT_OUT_DIR = Path(__file__).parent / "lora_ckpts"
WEIGHT_NAME = "pytorch_lora_weights.safetensors"
CONFIG_NAMES = ("artist_config.json", "arvtist_config.json")

Picked = namedtuple("Picked", ["slug", "folder_name", "step", "file_id", "size_hint"])


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def list_drive_tree(url: str):
    """Return the root _GoogleDriveFile with full children tree, no downloads."""
    sess = requests.Session()
    ok, root = _download_and_parse_google_drive_link(
        sess=sess, url=url, quiet=True, remaining_ok=True
    )
    if not ok or root is None:
        raise RuntimeError(f"failed to list drive folder: {url}")
    return root


def pick_largest_checkpoint(artist_folder) -> tuple[int, str] | None:
    """Return (step, file_id) of pytorch_lora_weights.safetensors in the largest
    checkpoint-N subfolder, or (-1, file_id) for a root-level weight as fallback.
    Returns None if no weight file is found."""
    candidates: list[tuple[int, str]] = []
    root_weight_id: str | None = None
    for child in artist_folder.children:
        if child.is_folder() and child.name.startswith("checkpoint-"):
            try:
                step = int(child.name.split("-", 1)[1])
            except ValueError:
                continue
            for f in child.children:
                if not f.is_folder() and f.name == WEIGHT_NAME:
                    candidates.append((step, f.id))
        elif not child.is_folder() and child.name == WEIGHT_NAME:
            root_weight_id = child.id
    if candidates:
        candidates.sort(key=lambda t: t[0])
        return candidates[-1]
    if root_weight_id:
        return (-1, root_weight_id)
    return None


def gdown_file(file_id: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    out = gdown.download(url=url, output=str(dest), quiet=False, fuzzy=False)
    if out is None:
        raise RuntimeError(f"gdown failed for id={file_id}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--drive-url", default=DEFAULT_DRIVE_URL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    log.info("listing drive folder (no downloads yet)...")
    root = list_drive_tree(args.drive_url)
    log.info("root: %s, children: %d", root.name, len(root.children))

    config_id: str | None = None
    config_name_chosen: str | None = None
    for ch in root.children:
        if not ch.is_folder() and ch.name in CONFIG_NAMES:
            if config_id is None or ch.name == "artist_config.json":
                config_id = ch.id
                config_name_chosen = ch.name

    kept: list[Picked] = []
    skipped: list[tuple[str, str]] = []

    for ch in root.children:
        if not ch.is_folder():
            continue
        slug = slugify(ch.name)
        pick = pick_largest_checkpoint(ch)
        if pick is None:
            skipped.append((ch.name, "no pytorch_lora_weights.safetensors in any checkpoint-*"))
            continue
        step, file_id = pick
        kept.append(Picked(slug, ch.name, step, file_id, None))

    log.info("plan: keep %d artists, skip %d", len(kept), len(skipped))
    for p in kept:
        step_lbl = f"checkpoint-{p.step}" if p.step >= 0 else "root"
        log.info("  will download: %s [%s] (id=%s)", p.slug, step_lbl, p.file_id)
    for name, reason in skipped:
        log.warning("  WARN skip %s: %s", name, reason)

    if config_id:
        cfg_dst = args.out_dir / "artist_config.json"
        log.info("downloading %s -> %s", config_name_chosen, cfg_dst)
        try:
            gdown_file(config_id, cfg_dst)
        except Exception as e:
            log.warning("failed to download %s: %s", config_name_chosen, e)
    else:
        log.warning("no artist_config.json found in root — will use derive_token() fallback")

    failed_downloads: list[tuple[str, str]] = []
    for p in kept:
        dst = args.out_dir / p.slug / WEIGHT_NAME
        if dst.exists() and dst.stat().st_size >= 1_000_000:
            log.info("already present: %s (%.1f MB), skipping",
                     dst, dst.stat().st_size / 1024 / 1024)
            continue
        log.info("downloading %s -> %s", p.slug, dst)
        try:
            gdown_file(p.file_id, dst)
            if dst.stat().st_size < 1_000_000:
                raise RuntimeError(
                    f"downloaded file <1MB ({dst.stat().st_size} bytes), likely truncated"
                )
        except Exception as e:
            log.error("failed %s: %s", p.slug, e)
            failed_downloads.append((p.folder_name, str(e)))
            if dst.exists():
                dst.unlink()

    print("\n=== SUMMARY ===")
    print(f"output dir: {args.out_dir}")
    print(f"kept ({len(kept) - len(failed_downloads)}):")
    failed_names = {n for n, _ in failed_downloads}
    for p in kept:
        if p.folder_name in failed_names:
            continue
        step_lbl = f"step={p.step}" if p.step >= 0 else "root_weight"
        size = (args.out_dir / p.slug / WEIGHT_NAME).stat().st_size / 1024 / 1024
        print(f"  {p.slug}: {p.folder_name} [{step_lbl}, {size:.1f} MB]")
    if skipped or failed_downloads:
        print(f"\nskipped ({len(skipped) + len(failed_downloads)}):")
        for name, reason in skipped:
            print(f"  {name}: no weights")
        for name, reason in failed_downloads:
            print(f"  {name}: download failed ({reason})")

    return 0 if (len(kept) - len(failed_downloads)) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
