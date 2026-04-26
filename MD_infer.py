# Example run:
#   python MD_infer.py \
#       --model_name sd15 \
#       --out_json /path/to/clip_prompts_renamed.json \
#       --suffix_pt /path/to/suffix.pt \
#       --output_dir /path/to/output \
#       --pair_dict_save_path /path/to/save_dict.pt \
#       --images_per_prompt 1 \
#       --num_inference_steps 25 \
#       --guidance_scale 7.5

import torch
import json
import re
import random
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image

# -----------------------
# Argument Parsing
# -----------------------
parser = argparse.ArgumentParser(description="Motif Duels")

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
    "--out_json",
    type=str,
    required=True,
    help="Path to the JSON file mapping artwork keys to description lists"
)
parser.add_argument(
    "--suffix_pt",
    type=str,
    required=True,
    help="Path to the .pt file containing artwork_key -> style suffix string"
)
parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Directory where generated images will be saved"
)
parser.add_argument(
    "--pair_dict_save_path",
    type=str,
    required=True,
    help="Path to save the pair-to-prompt-directory mapping (.pt file)"
)

# Generation hyper-parameters
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

args = parser.parse_args()

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
# Load Inputs
# -----------------------
with open(args.out_json, "r", encoding="utf-8") as f:
    mapping = json.load(f)  # e.g., mapping['A16'] -> list of 16 descriptions

suffix_dict = torch.load(args.suffix_pt)  # artwork_key -> style suffix string
key_list = list(suffix_dict.keys())

# -----------------------
# Helpers
# -----------------------
def sanitize_name(name: str) -> str:
    """Make a string safe for file/folder names."""
    name = str(name).strip()
    name = re.sub(r"\s", "_", name)               # spaces -> underscores
    name = re.sub(r"[^A-Za-z0-9_.-]", "", name)  # remove unsafe chars
    return name


def format_key_for_filename(key) -> str:
    """
    Build a deterministic, safe filename component from either:
      - tuple keys: (art, fix_artwork) -> 'art__FIXED_fix_artwork'
      - string keys: use sanitized string
    """
    if isinstance(key, tuple) and len(key) == 2:
        return sanitize_name(f"{key[0]}__FIXED_{key[1]}")
    return sanitize_name(str(key))


def first_sentence(text: str) -> str:
    """Return the first sentence. Splits on the first period followed by space or end."""
    s = str(text).strip()
    parts = re.split(r"\.(?:\s|$)", s, maxsplit=1)
    return parts[0].strip()


# -----------------------
# Global Motif Cache
# -----------------------
GLOBAL_SELECTED_BY_ART: Dict[str, List[str]] = {}


def preselect_five_motifs_per_artwork(mapping: dict,
                                      suffix_dict: dict,
                                      seed: int | None = None) -> Dict[str, List[str]]:
    """
    Preselect exactly five motifs per overlapping artwork once and store in global cache.

    Selection rule per artwork:
      - if len >= 5: random.sample(..., 5) without replacement
      - if len < 5:  random.choices(..., k=5) with replacement
    """
    if seed is not None:
        random.seed(seed)

    overlap_keys = [k for k in suffix_dict.keys() if k in mapping]
    if not overlap_keys:
        print("[ERROR] No overlapping keys between mapping and suffix_dict.")
        GLOBAL_SELECTED_BY_ART.clear()
        return {}

    selected_by_art: Dict[str, List[str]] = {}
    for art in overlap_keys:
        all_pmt = mapping.get(art, [])
        if not all_pmt:
            print(f"[WARN] No motifs for artwork '{art}'. Excluding from selection.")
            continue
        if len(all_pmt) >= 5:
            selected_by_art[art] = random.sample(all_pmt, 5)
        else:
            selected_by_art[art] = random.choices(all_pmt, k=5)

    GLOBAL_SELECTED_BY_ART.clear()
    GLOBAL_SELECTED_BY_ART.update(selected_by_art)

    print(f"[INFO] Preselected motifs for {len(selected_by_art)} artworks "
          f"(out of {len(overlap_keys)} overlapping keys).")

    return selected_by_art


def build_prompt_dict(mapping: dict,
                      suffix_dict: dict,
                      seed: int | None = None) -> dict:
    """
    Build composed prompts using globally preselected motifs for each artwork.
    Composed as: first_sentence(motif) + fixed_suffix
    Keyed as (other_art, fixed_artwork).
    """
    if seed is not None:
        random.seed(seed)

    if not GLOBAL_SELECTED_BY_ART:
        preselect_five_motifs_per_artwork(mapping, suffix_dict, seed)

    overlap_keys = [k for k in suffix_dict.keys() if k in mapping]
    if not overlap_keys:
        print("[ERROR] No overlapping keys between mapping and suffix_dict.")
        return {}

    prompt_dict: Dict[Tuple[str, str], List[str]] = {}

    for fixed_artwork in overlap_keys:
        fixed_suffix = str(suffix_dict[fixed_artwork])
        if not fixed_suffix.startswith(" "):
            fixed_suffix = " " + fixed_suffix

        for other_art in overlap_keys:
            if other_art == fixed_artwork:
                continue
            select_pmt = GLOBAL_SELECTED_BY_ART.get(other_art)
            if not select_pmt:
                continue
            composed = [first_sentence(m) + fixed_suffix for m in select_pmt]
            prompt_dict[(other_art, fixed_artwork)] = composed

    total_prompts = sum(len(v) for v in prompt_dict.values())
    num_selected = len(GLOBAL_SELECTED_BY_ART)

    valid_pairs = 0
    for fixed in overlap_keys:
        valid_pairs += num_selected - (1 if fixed in GLOBAL_SELECTED_BY_ART else 0)

    ideal_prompts = valid_pairs * 5
    missing = ideal_prompts - total_prompts

    print(f"[INFO] Built {total_prompts} prompts across {len(prompt_dict)} pairs "
          f"using {len(overlap_keys)} fixed artworks.")
    print(f"[INFO] Ideal target = {ideal_prompts}. "
          f"{'On target.' if missing == 0 else f'Shortfall = {missing} due to missing or empty motif lists.'}")

    return prompt_dict


# -----------------------
# Main Generation Function
# -----------------------
def generate_and_save_all(artwork_prompts: dict,
                          pipe,
                          base_dir: str,
                          pair_dict_save_path: str,
                          images_per_prompt: int,
                          num_inference_steps: int,
                          guidance_scale: float):
    """
    For each dict key and prompt, save images under a SINGLE folder (base_dir).
    Files are named: <key>_prompt_<i>.png
    If images_per_prompt > 1: <key>_prompt_<i>_img_<j>.png

    Records the mapping:
      {(artwork_1, artwork_2): [[prompt, <file_path>], ...]}
    Saves via torch.save() to pair_dict_save_path.
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    pair_to_prompt_dir_dict = {}

    for key, prompts in artwork_prompts.items():
        safe_key = format_key_for_filename(key)

        pair_key = None
        if isinstance(key, tuple) and len(key) == 2:
            pair_key = (str(key[0]), str(key[1]))
            if pair_key not in pair_to_prompt_dir_dict:
                pair_to_prompt_dir_dict[pair_key] = []

        for p_idx, prompt in enumerate(prompts, start=1):
            print(f"\n=== Generating for key '{key}' ===")
            print(f"Prompt {p_idx}/{len(prompts)}: {prompt}")

            for img_num in range(images_per_prompt):
                print(f"  -> Image {img_num + 1}/{images_per_prompt}")

                result = pipe(
                    prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale
                )
                image = result.images[0]

                if images_per_prompt == 1:
                    filename = base_path / f"{safe_key}_prompt_{p_idx}.png"
                else:
                    filename = base_path / f"{safe_key}_prompt_{p_idx}_img_{img_num + 1}.png"

                image.save(str(filename))
                print(f"    Saved: {filename}")

                if pair_key is not None:
                    pair_to_prompt_dir_dict[pair_key].append([prompt, str(filename)])

    # Save the mapping dict
    save_path = Path(pair_dict_save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pair_to_prompt_dir_dict, str(save_path))
    print(f"\nPair-to-prompt-directory dict saved to: {save_path}")


# -----------------------
# Entry Point
# -----------------------
if __name__ == "__main__":
    # 1. Preselect motifs once for reproducibility
    selected = preselect_five_motifs_per_artwork(mapping, suffix_dict, seed=42)
    torch.save(selected, args.pair_dict_save_path.replace(".pt", "_prefix.pt"))

    # 2. Build prompts using the fixed global selections
    prompt_dict = build_prompt_dict(mapping, suffix_dict)

    if not prompt_dict:
        print("[ERROR] No prompts generated. Check mapping/suffix inputs.")
    else:
        generate_and_save_all(
            artwork_prompts=prompt_dict,
            pipe=pipe,
            base_dir=args.output_dir,
            pair_dict_save_path=args.pair_dict_save_path,
            images_per_prompt=args.images_per_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        )
