# This code is assist by Claude Code

"""Pipeline B (LoRA): SDXL + Canny ControlNet from realistic photos + per-artist LoRA.

Mirrors Phase1_Augmentation.ipynb Cell 6b *including* load_lora_weights / fuse_lora.
Artists are auto-discovered from the LoRA checkpoint directory (default ./lora_ckpts/).

Per-artist generation: 200 images = 100 reference photos x 2 reps.
Prompt template (matches training instance prompt + ControlNet content cue):
    f"painting in style of {token}, {content_prompt}, masterpiece, detailed painting"
where token = sks_<slug>; structure comes from the Canny edge map of the ref photo.
"""

import argparse
import hashlib
import json
import logging
import re
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

log = logging.getLogger("lora_controlnet")

DEFAULT_LORA_ROOT = Path(__file__).parent / "lora_ckpts"
DEFAULT_INPUT_DIR = Path(__file__).parent / "data" / "realistic_inputs"
DEFAULT_OUT_DIR = Path(__file__).parent / "data" / "generated_lora_canny"
DEFAULT_CONFIG = Path(__file__).parent / "raw_sdxl_config.json"
VALID_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
WEIGHT_NAME = "pytorch_lora_weights.safetensors"


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


def derive_token(slug: str) -> str:
    return f"sks_{re.sub(r'[^a-z0-9]+', '_', slug.lower()).strip('_')}"


def deterministic_seed(tag: str, slug: str, src: str, rep: int) -> int:
    h = hashlib.sha256(f"{tag}|{slug}|{src}|{rep}".encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


def discover_artists(lora_root: Path, only: list[str] | None) -> list[dict]:
    out: list[dict] = []
    if not lora_root.exists():
        sys.exit(f"lora root not found: {lora_root}")
    for sub in sorted(lora_root.iterdir()):
        if not sub.is_dir():
            continue
        weight = sub / WEIGHT_NAME
        if not weight.exists() or weight.stat().st_size < 1_000_000:
            log.warning("skip %s: missing or truncated weight", sub.name)
            continue
        slug = sub.name
        if only and slug not in only:
            continue
        out.append({"slug": slug, "lora_dir": sub, "token": derive_token(slug)})
    return out


def list_input_images(input_dir: Path) -> list[Path]:
    def ok(p: Path) -> bool:
        if p.suffix.lower() not in VALID_IMG_EXT:
            return False
        return not any(part.startswith(".") for part in p.relative_to(input_dir).parts)
    return sorted(p for p in input_dir.rglob("*") if ok(p))


def center_crop_resize(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side)).resize((size, size), Image.LANCZOS)


def existing_done(meta_path: Path) -> set[tuple[str, int]]:
    if not meta_path.exists():
        return set()
    done = set()
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                done.add((d["ref_image"], int(d["rep"])))
            except Exception:
                continue
    return done


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lora-root", type=Path, default=DEFAULT_LORA_ROOT)
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                   help="Folder of realistic photos (recursively scanned)")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--artists", nargs="+", default=None,
                   help="Subset of artist slugs (default: all auto-discovered)")
    p.add_argument("--n-per-image", type=int, default=2,
                   help="Reps per reference photo (default 2 -> 200 imgs/artist with 100 refs)")
    p.add_argument("--max-refs", type=int, default=None,
                   help="Cap reference image count (smoke test)")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guidance", type=float, default=6.5)
    p.add_argument("--controlnet-scale", type=float, default=None,
                   help="Override per-artist scale (default: config root controlnet_scale)")
    p.add_argument("--canny-low", type=int, default=None)
    p.add_argument("--canny-high", type=int, default=None)
    p.add_argument("--content-prompt", type=str, default=None,
                   help="Content cue text (defaults to config.controlnet_content_prompt). "
                        "Canny supplies structure; text only nudges content.")
    p.add_argument("--lora-scale", type=float, default=0.95)
    p.add_argument("--save-canny", action="store_true",
                   help="Save edge-map previews under {out}/{artist}/_canny/")
    p.add_argument("--mem-opt", choices=["slicing", "offload", "none"], default="slicing")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    if not args.input_dir.is_dir():
        sys.exit(f"--input-dir does not exist: {args.input_dir}")
    ref_paths = list_input_images(args.input_dir)
    if not ref_paths:
        sys.exit(f"no images under {args.input_dir}")
    if args.max_refs:
        ref_paths = ref_paths[: args.max_refs]
    log.info("found %d reference images", len(ref_paths))

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    steps = args.steps
    guidance = args.guidance
    cn_scale = args.controlnet_scale if args.controlnet_scale is not None else cfg["controlnet_scale"]
    canny_low = args.canny_low if args.canny_low is not None else cfg["canny_low"]
    canny_high = args.canny_high if args.canny_high is not None else cfg["canny_high"]
    resolution = cfg["resolution"]
    negative = cfg["negative_prompt"]
    content_text = args.content_prompt if args.content_prompt is not None else cfg.get(
        "controlnet_content_prompt", "an outdoor landscape scene, original composition"
    )

    artists = discover_artists(args.lora_root, args.artists)
    if not artists:
        sys.exit(f"no artists under {args.lora_root}")
    log.info("artists to process: %s", [a["slug"] for a in artists])

    args.out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Building SDXL + ControlNet pipeline...")
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

    for artist in artists:
        slug = artist["slug"]
        token = artist["token"]
        artist_dir = args.out_dir / slug
        artist_dir.mkdir(parents=True, exist_ok=True)
        canny_dir = artist_dir / "_canny" if args.save_canny else None
        if canny_dir:
            canny_dir.mkdir(parents=True, exist_ok=True)
        meta_path = artist_dir / "meta.jsonl"
        done = existing_done(meta_path)
        if done:
            log.info("%s: resuming, %d (ref,rep) already done", slug, len(done))

        try:
            pipe.unfuse_lora()
        except Exception:
            pass
        try:
            pipe.unload_lora_weights()
        except Exception:
            pass

        log.info("%s: loading LoRA from %s, token=%s", slug, artist["lora_dir"], token)
        pipe.load_lora_weights(
            str(artist["lora_dir"]),
            weight_name=WEIGHT_NAME,
        )
        pipe.fuse_lora(lora_scale=args.lora_scale)

        full_prompt = (
            f"painting in style of {token}, {content_text}, "
            "masterpiece, detailed painting"
        )

        n_written = 0
        with open(meta_path, "a", encoding="utf-8") as meta_f:
            for rp in ref_paths:
                canny_img = canny_cache[rp]
                if canny_dir:
                    cp = canny_dir / f"{rp.stem}_{rp.suffix[1:]}.png"
                    if not cp.exists():
                        canny_img.save(cp)
                for rep in range(args.n_per_image):
                    if (str(rp), rep) in done:
                        continue
                    seed = deterministic_seed("lora_controlnet_v1", slug, rp.name, rep)
                    g = torch.Generator(gen_device).manual_seed(int(seed))
                    img = pipe(
                        prompt=full_prompt,
                        negative_prompt=negative,
                        image=canny_img,
                        controlnet_conditioning_scale=cn_scale,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        generator=g,
                    ).images[0]
                    save_path = artist_dir / f"gen_{rp.stem}_{rp.suffix[1:]}__{rep:02d}.png"
                    img.save(save_path)
                    meta_f.write(json.dumps({
                        "path": str(save_path),
                        "seed": int(seed),
                        "prompt": full_prompt,
                        "negative_prompt": negative,
                        "artist_slug": slug,
                        "token": token,
                        "ref_image": str(rp),
                        "rep": rep,
                        "steps": steps,
                        "guidance": guidance,
                        "controlnet_scale": cn_scale,
                        "canny_low": canny_low,
                        "canny_high": canny_high,
                        "lora_scale": args.lora_scale,
                    }, ensure_ascii=False) + "\n")
                    meta_f.flush()
                    n_written += 1

        pipe.unfuse_lora()
        pipe.unload_lora_weights()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("%s: wrote %d images (cn_scale=%.2f) -> %s",
                 slug, n_written, cn_scale, artist_dir)


if __name__ == "__main__":
    main()
