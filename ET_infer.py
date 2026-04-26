# pip install bitsandbytes
# pip install --upgrade transformers
# pip install accelerate
# pip install -e /workspace/data/Ashutosh/diffusers-main
# pip install --upgrade huggingface-hub
# pip install soxs
# pip install peft
#
# Example run:
#   python generate_images.py \
#       --model_name sd15 \
#       --prompts_pt /path/to/prompts.pt \
#       --root_output_dir /path/to/output \
#       --images_per_prompt 1 \
#       --num_inference_steps 25 \
#       --guidance_scale 7.5 \
#       --seed_base 42

import os
import re
import pickle
from pathlib import Path

import torch
from PIL import Image
import argparse

# -----------------------
# Argument Parsing
# -----------------------
parser = argparse.ArgumentParser(description="Image Generation Script for Multiple Diffusion Models")

# Model selection
parser.add_argument(
    "--model_name",
    type=str,
    required=True,
    choices=["sd15", "sdxl", "sana15"],
    help="Model to use for generation. Choices: sd15 | sdxl | sana15"
)

# Path arguments
parser.add_argument(
    "--prompts_pt",
    type=str,
    required=True,
    help="Path to the saved prompts .pt file"
)
parser.add_argument(
    "--root_output_dir",
    type=str,
    required=True,
    help="Root directory where generated images will be saved"
)

# Generation hyper-parameters (all optional with sensible defaults)
parser.add_argument(
    "--images_per_prompt",
    type=int,
    default=1,
    help="Number of images to generate per prompt (default: 1)"
)
parser.add_argument(
    "--num_inference_steps",
    type=int,
    default=25,
    help="Number of denoising steps (default: 25)"
)
parser.add_argument(
    "--guidance_scale",
    type=float,
    default=7.5,
    help="Classifier-free guidance scale (default: 7.5)"
)
parser.add_argument(
    "--seed_base",
    type=int,
    default=42,
    help="Base seed for deterministic generation. Pass -1 for fully random (default: 42)"
)

args = parser.parse_args()

# Normalise seed: treat -1 as None (fully random)
SEED_BASE = args.seed_base if args.seed_base >= 0 else None

# -----------------------
# Model Loading
# -----------------------
if args.model_name == "sd15":
    from diffusers import StableDiffusionPipeline
    MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"  # or "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        safety_checker=None
    )
    pipe = pipe.to("cuda")
    print("Stable Diffusion v1.5 loaded successfully!")

elif args.model_name == "sdxl":
    from diffusers import StableDiffusionXLPipeline
    MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")
    print("Stable Diffusion XL loaded successfully!")
    

elif args.model_name == "sana15":
    from diffusers import SanaPipeline
    pipe = SanaPipeline.frm_pretrained( "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers", torch_dtype = torch.bfloat16)
    pipe = pipe.to("cuda")
    pipe.text_encoder = pipe.text_encoder.to(torch.bfloat16)
    pipe.vae = pipe.vae.to(torch.bfloat16) 
    print("SANA 1.5 loaded successfully!")

# -----------------------
# Helpers
# -----------------------
def sanitize_name(name: str) -> str:
    """Make a string safe for file/folder names."""
    name = name.strip()
    name = re.sub(r"\s", "_", name)               # spaces -> underscores
    name = re.sub(r"[^A-Za-z0-9_.-]", "", name)  # remove unsafe chars
    return name


def parse_artist_artwork(prompt: str):
    """
    Extract (artist, artwork) from the prompt. Supports:
      - 'The <artwork> in the style of <artist>'
      - '<artwork> in <artist> style'
      - '<artwork> by <artist>'
    Returns (artist, artwork). Falls back to ('Unknown', prompt) if unmatched.
    """
    s = prompt.strip()

    # Pattern 1: The <artwork> in the style of <artist>
    m = re.match(
        r"^\s*(?:The\s+)?(?P<artwork>.+?)\s+in\s+the\s+style\s+of\s+(?P<artist>.+?)\s*$",
        s, flags=re.IGNORECASE
    )
    if m:
        return (m.group("artist").strip(), m.group("artwork").strip())

    # Pattern 2: <artwork> in <artist> style
    m = re.match(
        r"^\s*(?:The\s+)?(?P<artwork>.+?)\s+in\s+(?P<artist>.+?)\s+style\s*$",
        s, flags=re.IGNORECASE
    )
    if m:
        return (m.group("artist").strip(), m.group("artwork").strip())

    # Pattern 3: <artwork> by <artist>
    m = re.match(
        r"^\s*(?:The\s+)?(?P<artwork>.+?)\s+by\s+(?P<artist>.+?)\s*$",
        s, flags=re.IGNORECASE
    )
    if m:
        return (m.group("artist").strip(), m.group("artwork").strip())

    # Fallback
    return ("Unknown", s)


def load_prompts_list(pt_path: str):
    """
    Load list of prompts from a .pt file. Tries torch.load first, then pickle fallback.
    """
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Prompts file not found: {pt_path}")
    try:
        return torch.load(pt_path)
    except Exception:
        pass
    with open(pt_path, "rb") as f:
        return pickle.load(f)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


# -----------------------
# Main Generation Function
# -----------------------
def generate_from_pt(
    pt_path: str,
    pipe,
    root_output_dir: str,
    images_per_prompt: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed_base: int | None,
):
    """
    Loads prompts from 'pt_path', then for each prompt generates 'images_per_prompt'
    images and saves them under:
        root_output_dir/<artist>/<artist>_<artwork>_<sample>.png
    """
    prompts = load_prompts_list(pt_path)

    # Deduplicate while preserving order
    seen = set()
    unique_prompts = []
    for p in prompts:
        if p not in seen:
            unique_prompts.append(p)
            seen.add(p)

    root_path = Path(root_output_dir)
    ensure_dir(root_path)

    for idx, prompt in enumerate(unique_prompts, start=1):
        artist, artwork = parse_artist_artwork(prompt)
        safe_artist  = sanitize_name(artist if artist else "Unknown")
        safe_artwork = sanitize_name(artwork)

        artist_dir = root_path / safe_artist
        ensure_dir(artist_dir)

        print(f"\n[{idx}/{len(unique_prompts)}] Generating: '{artwork}' in the style of '{artist}'")
        print(f"Prompt: {prompt}")

        # Deterministic generators per image (optional)
        generator = None
        if seed_base is not None:
            generator = [
                torch.Generator(device="cuda").manual_seed(seed_base + idx * 1000 + k)
                for k in range(images_per_prompt)
            ]

        result = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=images_per_prompt,
            generator=generator
        )
        images = result.images  # list of PIL.Image

        for k, image in enumerate(images, start=1):
            filename = artist_dir / f"{safe_artist}_{safe_artwork}_{k:02d}.png"
            image.save(str(filename))
            print(f"  -> Saved: {filename}")


# -----------------------
# Entry Point
# -----------------------
if __name__ == "__main__":
    generate_from_pt(
        pt_path=args.prompts_pt,
        pipe=pipe,
        root_output_dir=args.root_output_dir,
        images_per_prompt=args.images_per_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed_base=SEED_BASE,
    )
