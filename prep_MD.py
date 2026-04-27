"""
prep_MD.py

Prepares the three utility files needed for Motif Duels:

    MD_utils_dir/
        suffix.pt       – {"A1": "in the style of <artwork> by <artist>.", ...}
        simple_maps.pt  – {"A1": "/path/to/original_artwork.jpg", ...}
        clean_maps.pt   – {"A1": "<artwork> by <artist>", ...}

Usage:
    python prep_MD.py \
        --motif_json     /workspace/.../Motifs.json \
        --top20_dir      /workspace/.../top20_original \
        --MD_utils_dir   /workspace/.../MD_utils
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_artist_artwork_from_path(filepath: str) -> tuple[str, str]:
    """
    Expects filenames of the form  .../<something>__<artist>_<artwork>.jpg
    where the meaningful part follows the last '__' separator.

    Returns (artist, artwork) with underscores replaced by spaces.
    """
    img_name = filepath.split("/")[-1]          # basename
    core     = img_name.split("__")[-1]         # strip any prefix up to '__'
    core     = core.replace(".jpg", "").replace(".jpeg", "").replace(".png", "")
    artist_raw, artwork_raw = core.rsplit("_", 1)
    artist  = artist_raw.replace("_", " ").strip()
    artwork = artwork_raw.replace("_", " ").strip()
    return artist, artwork


# ---------------------------------------------------------------------------
# Build maps
# ---------------------------------------------------------------------------

def build_simple_map(top20_dir: Path) -> Dict[str, str]:
    """
    {"A1": "/abs/path/to/img1.jpg", "A2": ..., ...}
    Sorted so the key assignment is deterministic.
    """
    pattern = str(top20_dir / "*.jpg")
    files   = sorted(glob.glob(pattern))

    if not files:
        sys.exit(f"[ERROR] No .jpg files found in: {top20_dir}")

    map_org = {f"A{i + 1}": path for i, path in enumerate(files)}
    print(f"[i] simple_maps: {len(map_org)} entries")
    return map_org


def build_clean_map(map_org: Dict[str, str]) -> Dict[str, str]:
    """
    {"A1": "<artwork> by <artist>", ...}
    """
    clean_mapping: Dict[str, str] = {}
    for k, filepath in map_org.items():
        artist, artwork = parse_artist_artwork_from_path(filepath)
        clean_mapping[k] = f"{artwork} by {artist}"
        print(f"  clean_maps  [{k}] → {clean_mapping[k]}")
    return clean_mapping


def build_suffix_dict(map_org: Dict[str, str]) -> Dict[str, str]:
    """
    {"A1": "in the style of <artwork> by <artist>.", ...}
    Uses the same key set as map_org so all three dicts are aligned.
    """
    suffix_dict: Dict[str, str] = {}
    for k, filepath in map_org.items():
        artist, artwork = parse_artist_artwork_from_path(filepath)
        prompt = f"in the style of {artwork} by {artist}."
        suffix_dict[k] = prompt
        print(f"  suffix_dict [{k}] → {prompt}")
    return suffix_dict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare suffix.pt, simple_maps.pt and clean_maps.pt for Motif Duels."
    )

    # ── Inputs ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--motif_json",
        required=True,
        help="Path to Motifs.json (LLM-extracted motif dict: {key: [motif1, motif2, ...]})",
    )
    parser.add_argument(
        "--top20_dir",
        required=True,
        help="Directory containing the top-20 original artwork .jpg files "
             "(ET_eval output). Glob pattern used: <top20_dir>/*.jpg",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--MD_utils_dir",
        required=True,
        help="Output directory. Will be created if it does not exist. "
             "Saves: suffix.pt, simple_maps.pt, clean_maps.pt",
    )

    args = parser.parse_args()

    motif_json_path = Path(args.motif_json).resolve()
    top20_dir       = Path(args.top20_dir).resolve()
    MD_utils_dir    = Path(args.MD_utils_dir).resolve()

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not motif_json_path.exists():
        sys.exit(f"[ERROR] motif_json not found: {motif_json_path}")
    if not top20_dir.is_dir():
        sys.exit(f"[ERROR] top20_dir is not a directory: {top20_dir}")

    ensure_dir(MD_utils_dir)

    # ── Load motif JSON ───────────────────────────────────────────────────────
    with open(motif_json_path, "r", encoding="utf-8") as f:
        motif_mapping = json.load(f)

    print(f"[i] Loaded motif JSON: {len(motif_mapping)} keys from {motif_json_path}")

    # ── Build the three dicts ─────────────────────────────────────────────────
    print("\n── Building simple_maps ──")
    map_org = build_simple_map(top20_dir)

    print("\n── Building clean_maps ──")
    clean_mapping = build_clean_map(map_org)

    print("\n── Building suffix_dict ──")
    suffix_dict = build_suffix_dict(map_org)

    # ── Save ──────────────────────────────────────────────────────────────────
    suffix_path      = MD_utils_dir / "suffix.pt"
    simple_maps_path = MD_utils_dir / "simple_maps.pt"
    clean_maps_path  = MD_utils_dir / "clean_maps.pt"

    torch.save(suffix_dict,   str(suffix_path))
    torch.save(map_org,       str(simple_maps_path))
    torch.save(clean_mapping, str(clean_maps_path))

    print(f"\n[✓] suffix.pt      → {suffix_path}")
    print(f"[✓] simple_maps.pt → {simple_maps_path}")
    print(f"[✓] clean_maps.pt  → {clean_maps_path}")
    print("\n[Done]")


if __name__ == "__main__":
    main()
