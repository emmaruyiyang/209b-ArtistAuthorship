# This code is assist by Claude Code

"""Pipeline A: raw SDXL (no LoRA) text2img augmentation per artist.

Mirrors Phase1_Augmentation.ipynb Cell 6 but skips load_lora_weights / fuse_lora.
Reads shared config raw_sdxl_config.json (next to this file by default).
"""

import argparse
import hashlib
import json
import logging
import random
import sys
from pathlib import Path

import torch
from diffusers import AutoencoderKL, StableDiffusionXLPipeline

log = logging.getLogger("rawsdxl_text2img")

DEFAULT_CONFIG = Path(__file__).parent / "raw_sdxl_config.json"
DEFAULT_OUT_DIR = Path(__file__).parent / "data" / "generated_raw_sdxl"


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_device_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    log.warning("CUDA not available; falling back to cpu/float32 (will be very slow)")
    return "cpu", torch.float32


def build_pipeline(cfg: dict, mem_opt: str) -> StableDiffusionXLPipeline:
    device, dtype = resolve_device_dtype()
    vae = AutoencoderKL.from_pretrained(cfg["vae_model"], torch_dtype=dtype)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        cfg["base_model"],
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


def deterministic_seed(tag: str, slug: str, content: str, rep: int) -> int:
    h = hashlib.sha256(f"{tag}|{slug}|{content}|{rep}".encode("utf-8")).digest()
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--artists", nargs="+", default=None,
                   help="Subset of artist slugs (default: all from config)")
    p.add_argument("--n-per-prompt", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--guidance", type=float, default=None)
    p.add_argument("--seed", type=int, default=None,
                   help="If set, use deterministic per-image seeds derived from this seed")
    p.add_argument("--mem-opt", choices=["slicing", "offload", "none"], default="slicing")
    p.add_argument("--use-style-hint", dest="use_style_hint", action="store_true", default=None)
    p.add_argument("--no-style-hint", dest="use_style_hint", action="store_false")
    p.add_argument("--negative-prompt", type=str, default=None)
    p.add_argument("--prompts-file", type=Path, default=None,
                   help="Newline-delimited content prompts; overrides config")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    n_per_prompt = args.n_per_prompt if args.n_per_prompt is not None else cfg["n_per_prompt"]
    steps = args.steps if args.steps is not None else cfg["steps"]
    guidance = args.guidance if args.guidance is not None else cfg["guidance_scale"]
    use_style_hint = args.use_style_hint if args.use_style_hint is not None else cfg["use_style_hint"]
    negative = args.negative_prompt if args.negative_prompt is not None else cfg["negative_prompt"]

    if args.prompts_file:
        content_prompts = [
            line.strip() for line in args.prompts_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        content_prompts = cfg["content_prompts"]

    artists = select_artists(cfg, args.artists)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Building pipeline (this downloads ~7GB on first run)...")
    pipe = build_pipeline(cfg, args.mem_opt)
    gen_device = "cpu" if args.mem_opt == "offload" else resolve_device_dtype()[0]

    rng = random.Random(args.seed) if args.seed is None else None

    for artist in artists:
        slug = artist["slug"]
        artist_dir = args.out_dir / slug
        artist_dir.mkdir(parents=True, exist_ok=True)
        meta_path = artist_dir / "meta.jsonl"
        idx = 0
        with open(meta_path, "w", encoding="utf-8") as meta_f:
            for content in content_prompts:
                prompt = build_prompt(artist, content, use_style_hint)
                for rep in range(n_per_prompt):
                    if args.seed is not None:
                        seed = deterministic_seed("text2img", slug, content, rep) ^ args.seed
                        seed &= 0x7FFFFFFF
                    else:
                        seed = rng.randint(1, 10_000_000) if rng else random.randint(1, 10_000_000)
                    g = torch.Generator(gen_device).manual_seed(int(seed))
                    img = pipe(
                        prompt=prompt,
                        negative_prompt=negative,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        generator=g,
                    ).images[0]
                    save_path = artist_dir / f"gen_{idx:05d}.png"
                    img.save(save_path)
                    meta_f.write(json.dumps({
                        "path": str(save_path),
                        "seed": int(seed),
                        "prompt": prompt,
                        "negative_prompt": negative,
                        "artist_slug": slug,
                        "artist_display": artist["display"],
                        "content_prompt": content,
                        "rep": rep,
                        "steps": steps,
                        "guidance": guidance,
                    }, ensure_ascii=False) + "\n")
                    meta_f.flush()
                    idx += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("%s: wrote %d images -> %s", slug, idx, artist_dir)


if __name__ == "__main__":
    main()
