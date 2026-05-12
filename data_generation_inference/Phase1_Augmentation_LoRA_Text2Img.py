# This code is assist by Claude Code

"""Pipeline A (LoRA): SDXL text2img with per-artist DreamBooth-LoRA adapter.

Mirrors Phase1_Augmentation.ipynb Cell 6 *including* load_lora_weights / fuse_lora.
Artists are auto-discovered from the local LoRA checkpoint directory
(default: ./lora_ckpts/) — any subfolder containing pytorch_lora_weights.safetensors
is a usable artist.

Per-artist generation: 200 images = 2 reps x 100 hardcoded content prompts.
Prompt template (matches training instance/validation prompt):
    f"painting in style of {token}, {content}, masterpiece, detailed painting"
where token = sks_<lowercased_artist_name_with_underscores>.
"""

import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path

import torch
from diffusers import AutoencoderKL, StableDiffusionXLPipeline

log = logging.getLogger("lora_text2img")

DEFAULT_LORA_ROOT = Path(__file__).parent / "lora_ckpts"
DEFAULT_OUT_DIR = Path(__file__).parent / "data" / "generated_lora"
DEFAULT_CONFIG = Path(__file__).parent / "raw_sdxl_config.json"  # for negative_prompt etc.
WEIGHT_NAME = "pytorch_lora_weights.safetensors"

CONTENT_PROMPTS: list[str] = [
    # --- urban / street (12) ---
    "a rainy city street at night with reflections on wet pavement",
    "a narrow cobblestone alley in an old european town",
    "a busy avenue at dusk with carriages and pedestrians",
    "a quiet provincial main street in the early morning",
    "a port town pier at sunset with fishing boats moored",
    "a railway station platform under gaslight",
    "a market square with awnings and stalls",
    "a snow-covered village street in winter twilight",
    "a tram crossing a bridge in a small city",
    "a stone bridge over a river in an old town",
    "a baroque cathedral facade at midday",
    "a wooden boardwalk along a seaside promenade",
    # --- nature: mountains / forests / fields (16) ---
    "a mountain valley at sunrise with mist clinging to the slopes",
    "a snow-capped alpine peak under a blue sky",
    "a pine forest path with shafts of late afternoon light",
    "a deciduous forest in full autumn color",
    "a meadow of wildflowers under summer clouds",
    "a wheat field rippling in the wind at harvest",
    "a haystack catching low evening light in a field",
    "a winding country road through rolling hills",
    "a birch grove with white trunks under spring sun",
    "a windswept moor with heather under grey clouds",
    "a hillside vineyard with rows of vines and a stone farmhouse",
    "a lavender field stretching toward distant hills",
    "a pond in a quiet woodland clearing",
    "a poppy field at noon in summer",
    "a foggy bog at dawn with reeds in the foreground",
    "a winter forest after a fresh snowfall",
    # --- water: rivers / seas / lakes (12) ---
    "a lake at dawn with mist on the water",
    "a turbulent sea under a stormy sky",
    "a calm bay with sailboats reflected on the water",
    "a rocky coastline with crashing waves",
    "a river bend curving through a wooded valley",
    "a waterfall plunging into a forest pool",
    "a small fishing village on a calm harbor",
    "a row of moored boats on a quiet riverbank",
    "a moonlit canal in a sleeping town",
    "a wide delta at low tide with reflective mudflats",
    "a frozen lake with skaters and pine trees",
    "a tropical lagoon at midday with palms on the shore",
    # --- skies / weather (6) ---
    "a dramatic sunset over an open plain",
    "towering thunderheads building over distant mountains",
    "a rainbow arching across a rain-soaked field",
    "a comet streaking across a starry rural sky",
    "a brilliant aurora over an arctic landscape",
    "a heavy snowfall over a country lane",
    # --- people / figures (14) ---
    "a peasant family gathered around an evening hearth",
    "a young woman reading by a sunlit window",
    "a portrait of an old fisherman with weathered hands",
    "a child playing with a wooden top on a kitchen floor",
    "a couple dancing at a village festival",
    "a self-portrait of an artist holding a palette",
    "a mother nursing her infant by candlelight",
    "a farmer leading a workhorse through a field",
    "a group of women gleaning in a wheat field at dusk",
    "a portrait of a scholar at a cluttered desk",
    "a soldier resting under a tree on a long march",
    "a dancer adjusting her shoe in a quiet studio",
    "a baker at work in a stone-walled bakery before dawn",
    "a pilgrim walking a dusty road at sunset",
    # --- interiors (10) ---
    "an empty bedroom with morning light streaming through gauzy curtains",
    "a dim tavern interior with figures around a wooden table",
    "a candlelit study with stacked books and an open ledger",
    "a music room with a piano and an open window",
    "an artist's studio with canvases stacked against a wall",
    "a peasant kitchen with hanging copper pots and a stone hearth",
    "a grand dining hall set for a feast",
    "a chapel interior with light falling on the altar",
    "a cluttered apothecary with jars and dried herbs",
    "a workshop full of carpentry tools and wood shavings",
    # --- still life (10) ---
    "a still life of sunflowers in a ceramic vase on a wooden table",
    "a still life of apples and pears on a draped cloth",
    "a still life of a glass of wine and a half-eaten loaf",
    "a still life of irises in a tall earthenware jug",
    "a still life of a skull, a candle, and an open book",
    "a still life of oysters, lemons, and a silver pitcher",
    "a still life of musical instruments on a draped table",
    "a still life of a basket of mushrooms beside a brass kettle",
    "a still life of game birds hanging from a hook",
    "a still life of summer fruit spilling from a wicker basket",
    # --- animals (8) ---
    "a herd of cattle grazing in a misty meadow at sunrise",
    "a flock of sheep crossing a dirt road in late afternoon",
    "a horse standing in a stable doorway",
    "two dogs playing in a snowy yard",
    "a barn cat asleep on a wooden bench in sunlight",
    "a pair of swans on a reed-lined pond",
    "a wolf at the edge of a winter forest at dusk",
    "a fox stepping through tall grass at twilight",
    # --- religious / mythological / historical (8) ---
    "an angel descending into a moonlit garden",
    "a saint praying in a rocky desert at dawn",
    "a knight on horseback at the edge of a forest",
    "a mother and child in a peaceful pastoral landscape",
    "a procession of pilgrims crossing a stone bridge",
    "a hermit reading by lamplight in a cave entrance",
    "a king receiving a messenger in a torchlit hall",
    "a mythic battle scene with banners on a windy hill",
    # --- nightscapes / festivals (4) ---
    "a midsummer bonfire on a hilltop with dancers in silhouette",
    "fireworks bursting over a quiet harbor at night",
    "a lantern-lit village square during a winter festival",
    "a long table feast under strings of lights in a summer garden",
]
assert len(CONTENT_PROMPTS) == 100, f"expected 100 prompts, got {len(CONTENT_PROMPTS)}"


def resolve_device_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    log.warning("CUDA not available; falling back to cpu/float32 (will be very slow)")
    return "cpu", torch.float32


def build_pipeline(base_model: str, vae_model: str, mem_opt: str) -> StableDiffusionXLPipeline:
    device, dtype = resolve_device_dtype()
    vae = AutoencoderKL.from_pretrained(vae_model, torch_dtype=dtype)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model,
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


def deterministic_seed(tag: str, slug: str, content: str, rep: int) -> int:
    h = hashlib.sha256(f"{tag}|{slug}|{content}|{rep}".encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


def discover_artists(lora_root: Path, only: list[str] | None) -> list[dict]:
    """Return list of {slug, lora_dir, token} for each artist subfolder containing
    pytorch_lora_weights.safetensors. Optionally filter by --artists slugs."""
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
    if only:
        found = {a["slug"] for a in out}
        for s in only:
            if s not in found:
                log.warning("requested artist '%s' not in %s", s, lora_root)
    return out


def existing_done(meta_path: Path) -> set[tuple[str, int]]:
    if not meta_path.exists():
        return set()
    done = set()
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                done.add((d["content_prompt"], int(d["rep"])))
            except Exception:
                continue
    return done


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lora-root", type=Path, default=DEFAULT_LORA_ROOT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                   help="raw_sdxl_config.json (only used for negative_prompt + base/vae model ids)")
    p.add_argument("--artists", nargs="+", default=None,
                   help="Subset of artist slugs (default: all auto-discovered)")
    p.add_argument("--n-per-prompt", type=int, default=2,
                   help="Reps per content prompt (default 2 -> 200 imgs/artist)")
    p.add_argument("--max-prompts", type=int, default=None,
                   help="Cap number of content prompts (smoke test)")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guidance", type=float, default=6.5)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--lora-scale", type=float, default=0.95,
                   help="fuse_lora scale (matches training Cell 6 default 0.95)")
    p.add_argument("--mem-opt", choices=["slicing", "offload", "none"], default="none")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    base_model = cfg["base_model"]
    vae_model = cfg["vae_model"]
    negative = cfg["negative_prompt"]

    content_prompts = CONTENT_PROMPTS
    if args.max_prompts:
        content_prompts = content_prompts[: args.max_prompts]

    artists = discover_artists(args.lora_root, args.artists)
    if not artists:
        sys.exit(f"no artists found under {args.lora_root}")
    log.info("artists to process: %s", [a["slug"] for a in artists])

    args.out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Building SDXL pipeline...")
    pipe = build_pipeline(base_model, vae_model, args.mem_opt)
    gen_device = "cpu" if args.mem_opt == "offload" else resolve_device_dtype()[0]

    for artist in artists:
        slug = artist["slug"]
        token = artist["token"]
        artist_dir = args.out_dir / slug
        artist_dir.mkdir(parents=True, exist_ok=True)
        meta_path = artist_dir / "meta.jsonl"
        done = existing_done(meta_path)
        if done:
            log.info("%s: resuming, %d (content,rep) already done", slug, len(done))

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

        idx = 0
        with open(meta_path, "a", encoding="utf-8") as meta_f:
            for content in content_prompts:
                full_prompt = (
                    f"painting in style of {token}, {content}, "
                    "masterpiece, detailed painting"
                )
                for rep in range(args.n_per_prompt):
                    if (content, rep) in done:
                        idx += 1
                        continue
                    seed = deterministic_seed("lora_text2img_v1", slug, content, rep)
                    g = torch.Generator(gen_device).manual_seed(int(seed))
                    img = pipe(
                        prompt=full_prompt,
                        negative_prompt=negative,
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance,
                        height=args.height,
                        width=args.width,
                        generator=g,
                    ).images[0]
                    save_path = artist_dir / f"gen_{idx:05d}.png"
                    img.save(save_path)
                    meta_f.write(json.dumps({
                        "path": str(save_path),
                        "seed": int(seed),
                        "prompt": full_prompt,
                        "negative_prompt": negative,
                        "artist_slug": slug,
                        "token": token,
                        "content_prompt": content,
                        "rep": rep,
                        "steps": args.steps,
                        "guidance": args.guidance,
                        "lora_scale": args.lora_scale,
                        "height": args.height,
                        "width": args.width,
                    }, ensure_ascii=False) + "\n")
                    meta_f.flush()
                    idx += 1

        pipe.unfuse_lora()
        pipe.unload_lora_weights()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("%s: total imgs in %s = %d", slug, artist_dir, idx)


if __name__ == "__main__":
    main()
