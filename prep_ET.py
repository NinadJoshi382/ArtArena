"""
prep_ET.py

Scans a dataset directory of images named <artwork>_<artist>.<ext>,
builds a list of prompts ("<artwork> in the style of <artist>"),
saves them to a .pt file, and writes a JSON mapping of
    { original_path: [generated_path_sample_0, generated_path_sample_1, ...] }

Usage:
    python prep_ET.py \
        --dataset_dir   /path/to/dataset \
        --prompt_save   /path/to/prompts.pt \
        --mapping_json_save /path/to/mapping.json \
        --root_output_dir   /path/to/output \
        [--num_imgs 1] \
        [--name_split last]

Generated filename index convention: 1-based, zero-padded to 2 digits.
    e.g. num_imgs=2  →  artist_artwork_01.png, artist_artwork_02.png
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Supported image extensions
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def parse_artwork_artist(stem: str, split_on: str = "last") -> tuple[str, str]:
    """
    Parse a filename stem of the form <artwork>_<artist>.

    Because both <artwork> and <artist> may themselves contain underscores,
    the split strategy is configurable:

        "last"  – everything before the last underscore is the artwork name,
                  the last token is the artist name.
                  e.g. "starry_night_van_gogh" → artwork="starry_night", artist="van_gogh"
                  Not ideal when the artist name has multiple words joined by underscores.

        "first" – the first token is the artwork name, everything else is the artist.
                  e.g. "starry_night_van_gogh" → artwork="starry", artist="night_van_gogh"

    In practice you should choose the convention that matches how your dataset
    files are actually named.  Override via --name_split.

    Returns (artwork, artist) where spaces replace underscores for readability.
    """
    if "_" not in stem:
        # Cannot split – treat entire name as artwork, artist unknown
        return stem.replace("_", " "), "unknown"

    if split_on == "first":
        idx = stem.index("_")
        artwork_raw = stem[:idx]
        artist_raw  = stem[idx + 1:]
    else:  # "last" (default)
        idx = stem.rindex("_")
        artwork_raw = stem[:idx]
        artist_raw  = stem[idx + 1:]

    artwork = artwork_raw.replace("_", " ").strip()
    artist  = artist_raw.replace("_", " ").strip()
    return artwork, artist


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def collect_images(dataset_dir: Path) -> list[Path]:
    """Return all image files in dataset_dir (non-recursive, sorted)."""
    images = sorted(
        p for p in dataset_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return images


def build_prompt(artwork: str, artist: str) -> str:
    return f"{artwork} in the style of {artist}"


def generated_paths_for(
    artist: str,
    artwork: str,
    root_output_dir: Path,
    num_imgs: int,
) -> list[str]:
    """
    Mirror the naming convention used by generate_from_pt:
        root_output_dir/<artist>_<artwork>_<idx>.png
    where spaces in names are replaced with underscores and <idx> is
    1-based and zero-padded to 2 digits (01, 02, …) to match the files
    actually written to disk (e.g. A_1_01.png, A_1_02.png).
    """
    artist_slug  = artist.replace(" ", "_")
    artwork_slug = artwork.replace(" ", "_")
    return [
        str(root_output_dir / f"{artist_slug}_{artwork_slug}_{i + 1:02d}.png")
        for i in range(num_imgs)
    ]


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def save_prompts_pt(prompts: list[str], pt_path: Path) -> None:
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prompts, str(pt_path))
    print(f"[✓] Saved {len(prompts)} prompt(s) → {pt_path}")


def save_mapping_json(mapping: dict[str, list[str]], json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f"[✓] Saved mapping ({len(mapping)} entries) → {json_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare prompts .pt and JSON mapping from an artwork dataset."
    )
    parser.add_argument(
        "--dataset_dir",
        required=True,
        help="Directory containing images named <artwork>_<artist>.<ext>",
    )
    parser.add_argument(
        "--prompt_save",
        required=True,
        help='Path where the prompts .pt file will be saved, e.g. workspace/.../prompts.pt',
    )
    parser.add_argument(
        "--mapping_json_save",
        required=True,
        help="Path where the JSON mapping {original_path: [generated_path, ...]} will be saved",
    )
    parser.add_argument(
        "--root_output_dir",
        required=True,
        help="Root directory where generate_from_pt will write the generated images",
    )
    parser.add_argument(
        "--num_imgs",
        type=int,
        default=1,
        help="Number of generated images per prompt (default: 1). "
             "Must match the value you pass to generate_from_pt.",
    )
    parser.add_argument(
        "--name_split",
        choices=["first", "last"],
        default="last",
        help=(
            "How to split <artwork>_<artist> filenames. "
            "'last'  → artwork=everything-before-last-underscore, artist=last-token (default). "
            "'first' → artwork=first-token, artist=everything-after-first-underscore."
        ),
    )

    args = parser.parse_args()

    dataset_dir     = Path(args.dataset_dir).resolve()
    prompt_save     = Path(args.prompt_save).resolve()
    mapping_json_save = Path(args.mapping_json_save).resolve()
    root_output_dir = Path(args.root_output_dir).resolve()

    # ── Validate dataset directory ──────────────────────────────────────────
    if not dataset_dir.exists():
        sys.exit(f"[ERROR] dataset_dir does not exist: {dataset_dir}")
    if not dataset_dir.is_dir():
        sys.exit(f"[ERROR] dataset_dir is not a directory: {dataset_dir}")

    # ── Collect images ───────────────────────────────────────────────────────
    images = collect_images(dataset_dir)
    if not images:
        sys.exit(f"[ERROR] No supported image files found in: {dataset_dir}")

    print(f"[i] Found {len(images)} image(s) in {dataset_dir}")

    # ── Build prompts and mapping ────────────────────────────────────────────
    prompts: list[str]                  = []
    mapping: dict[str, list[str]]       = {}   # original_path → [generated_path, ...]
    skipped: list[str]                  = []

    for img_path in images:
        stem = img_path.stem  # filename without extension

        artwork, artist = parse_artwork_artist(stem, split_on=args.name_split)

        if not artwork or not artist or artist == "unknown":
            print(f"  [!] Skipping '{img_path.name}': could not parse artwork/artist.")
            skipped.append(str(img_path))
            continue

        prompt = build_prompt(artwork, artist)
        gen_paths = generated_paths_for(
            artist, artwork, root_output_dir, args.num_imgs
        )

        prompts.append(prompt)
        mapping[str(img_path)] = gen_paths

        print(f"  → '{img_path.name}'  |  prompt: \"{prompt}\"")
        if args.num_imgs == 1:
            print(f"     generated: {gen_paths[0]}")
        else:
            for gp in gen_paths:
                print(f"     generated: {gp}")

    if not prompts:
        sys.exit("[ERROR] No valid image/prompt pairs could be built. Nothing saved.")

    if skipped:
        print(f"\n[!] Skipped {len(skipped)} file(s) due to parse failures.")

    # ── Persist ──────────────────────────────────────────────────────────────
    print()
    save_prompts_pt(prompts, prompt_save)
    save_mapping_json(mapping, mapping_json_save)

    print("\n[Done]")
    print(f"  Prompts file : {prompt_save}")
    print(f"  Mapping file : {mapping_json_save}")
    print(f"  Total prompts: {len(prompts)}")


if __name__ == "__main__":
    main()
