# This code is assist by Claude Code

"""Pipeline B: raw SDXL (no LoRA) + Canny ControlNet from realistic photos.

Mirrors Phase1_Augmentation.ipynb Cell 6b but skips load_lora_weights / fuse_lora.
Reads shared config raw_sdxl_config.json (next to this file by default).

Input: a folder of realistic photos (use download_landscape_photos.py to populate).
Output: per-artist folder of stylized images that preserve the photo's structure.
"""

import argparse
import hashlib
import json
import logging
import random
import sys
from pathlib import Path

import torch
from PIL import Image
from controlnet_aux import CannyDetector
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
)

log = logging.getLogger("rawsdxl_controlnet")

DEFAULT_CONFIG = Path(__file__).parent / "raw_sdxl_config.json"
DEFAULT_OUT_DIR = Path(__file__).parent / "data" / "generated_canny_raw_sdxl"
VALID_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_device_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    log.warning("CUDA not available; falling back to cpu/float32 (will be very slow)")
    return "cpu", torch.float32


def build_pipeline(cfg: dict, mem_opt: str) -> StableDiffusionXLControlNetPipeline:
    device, dtype = resolve_device_dtype()
    vae = AutoencoderKL.from_pretrained(cfg["vae_model"], torch_dtype=dtype)
    controlnet = ControlNetModel.from_pretrained(
        cfg["controlnet_model"], torch_dtype=dtype, use_safetensors=True
    )
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        cfg["base_model"],
        controlnet=controlnet,
        vae=vae,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    if mem_opt == "offload":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
        if mem_opt == "slicing":
            pipe.enable_attention_slicing()
    return pipe


def build_prompt(artist: dict, content: str, use_style_hint: bool) -> str:
    if use_style_hint and artist.get("style_hint"):
        return (
            f"{content}, {artist['style_hint']} in the style of {artist['display']}, "
            "masterpiece, highly detailed"
        )
    return (
        f"{content}, painting in the style of {artist['display']}, "
        "masterpiece, highly detailed"
    )


def deterministic_seed(tag: str, slug: str, src: str, rep: int) -> int:
    h = hashlib.sha256(f"{tag}|{slug}|{src}|{rep}".encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


def select_artists(cfg: dict, slugs: list[str] | None) -> list[dict]:
    by_slug = {a["slug"]: a for a in cfg["artists"]}
    if not slugs:
        return cfg["artists"]
    out = []
    for s in slugs:
        if s not in by_slug:
            log.warning("artist slug '%s' not in config; using fallback display name", s)
            out.append({"slug": s, "display": s.replace("_", " ").title(), "style_hint": ""})
        else:
            out.append(by_slug[s])
    return out


def list_input_images(input_dir: Path) -> list[Path]:
    paths = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in VALID_IMG_EXT)
    return paths


def center_crop_resize(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side)).resize((size, size), Image.LANCZOS)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True,
                   help="Folder of realistic photos (recursively scanned)")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--artists", nargs="+", default=None)
    p.add_argument("--n-per-image", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--guidance", type=float, default=None)
    p.add_argument("--controlnet-scale", type=float, default=None)
    p.add_argument("--canny-low", type=int, default=None)
    p.add_argument("--canny-high", type=int, default=None)
    p.add_argument("--max-refs", type=int, default=None,
                   help="Cap number of reference images (default: use all)")
    p.add_argument("--content-prompt", type=str, default=None,
                   help="Content cue; canny carries the structure. "
                        "Defaults to config.controlnet_content_prompt.")
    p.add_argument("--save-canny", action="store_true",
                   help="Save edge-map previews under {out}/{artist}/_canny/")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--mem-opt", choices=["slicing", "offload", "none"], default="slicing")
    p.add_argument("--use-style-hint", dest="use_style_hint", action="store_true", default=None)
    p.add_argument("--no-style-hint", dest="use_style_hint", action="store_false")
    p.add_argument("--negative-prompt", type=str, default=None)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    if not args.input_dir.is_dir():
        sys.exit(f"--input-dir does not exist: {args.input_dir}")
    ref_paths = list_input_images(args.input_dir)
    if not ref_paths:
        sys.exit(f"no images (.jpg/.jpeg/.png/.webp) found under: {args.input_dir}")
    if args.max_refs:
        ref_paths = ref_paths[: args.max_refs]
    log.info("found %d reference images", len(ref_paths))

    cfg = load_config(args.config)
    n_per_image = args.n_per_image if args.n_per_image is not None else cfg["n_per_image"]
    steps = args.steps if args.steps is not None else cfg["steps"]
    guidance = args.guidance if args.guidance is not None else cfg["guidance_scale"]
    cn_scale_override = args.controlnet_scale  # if user explicitly passed it, override per-artist
    cn_scale_default = cfg["controlnet_scale"]
    canny_low = args.canny_low if args.canny_low is not None else cfg["canny_low"]
    canny_high = args.canny_high if args.canny_high is not None else cfg["canny_high"]
    resolution = cfg["resolution"]
    use_style_hint = args.use_style_hint if args.use_style_hint is not None else cfg["use_style_hint"]
    negative = args.negative_prompt if args.negative_prompt is not None else cfg["negative_prompt"]
    content_prompt = args.content_prompt if args.content_prompt is not None else cfg.get(
        "controlnet_content_prompt", "an outdoor landscape scene, original composition"
    )

    artists = select_artists(cfg, args.artists)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Building pipeline (this downloads ~10GB on first run)...")
    pipe = build_pipeline(cfg, args.mem_opt)
    gen_device = "cpu" if args.mem_opt == "offload" else resolve_device_dtype()[0]

    log.info("Pre-computing Canny edge maps for %d images...", len(ref_paths))
    canny_detector = CannyDetector()
    canny_cache: dict[Path, Image.Image] = {}
    for rp in ref_paths:
        ref = center_crop_resize(Image.open(rp).convert("RGB"), resolution)
        canny_cache[rp] = canny_detector(
            ref, low_threshold=canny_low, high_threshold=canny_high,
            detect_resolution=resolution, image_resolution=resolution,
        )

    rng = random.Random(args.seed) if args.seed is None else None

    for artist in artists:
        slug = artist["slug"]
        artist_dir = args.out_dir / slug
        artist_dir.mkdir(parents=True, exist_ok=True)
        canny_dir = artist_dir / "_canny" if args.save_canny else None
        if canny_dir:
            canny_dir.mkdir(parents=True, exist_ok=True)
        artist_cn_scale = (
            cn_scale_override if cn_scale_override is not None
            else artist.get("controlnet_scale", cn_scale_default)
        )
        meta_path = artist_dir / "meta.jsonl"
        n_written = 0
        with open(meta_path, "w", encoding="utf-8") as meta_f:
            for rp in ref_paths:
                canny_img = canny_cache[rp]
                if canny_dir:
                    canny_img.save(canny_dir / f"{rp.stem}_{rp.suffix[1:]}.png")
                prompt = build_prompt(artist, content_prompt, use_style_hint)
                for rep in range(n_per_image):
                    if args.seed is not None:
                        seed = deterministic_seed("controlnet", slug, rp.name, rep) ^ args.seed
                        seed &= 0x7FFFFFFF
                    else:
                        seed = rng.randint(1, 10_000_000) if rng else random.randint(1, 10_000_000)
                    g = torch.Generator(gen_device).manual_seed(int(seed))
                    img = pipe(
                        prompt=prompt,
                        negative_prompt=negative,
                        image=canny_img,
                        controlnet_conditioning_scale=artist_cn_scale,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        generator=g,
                    ).images[0]
                    save_path = artist_dir / f"gen_{rp.stem}_{rp.suffix[1:]}__{rep:02d}.png"
                    img.save(save_path)
                    meta_f.write(json.dumps({
                        "path": str(save_path),
                        "seed": int(seed),
                        "prompt": prompt,
                        "negative_prompt": negative,
                        "artist_slug": slug,
                        "artist_display": artist["display"],
                        "ref_image": str(rp),
                        "rep": rep,
                        "steps": steps,
                        "guidance": guidance,
                        "controlnet_scale": artist_cn_scale,
                        "canny_low": canny_low,
                        "canny_high": canny_high,
                    }, ensure_ascii=False) + "\n")
                    meta_f.flush()
                    n_written += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("%s: wrote %d images (cn_scale=%.2f) -> %s",
                 slug, n_written, artist_cn_scale, artist_dir)


if __name__ == "__main__":
    main()
