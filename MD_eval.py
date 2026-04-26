"""
Tournament evaluation using CLIP cosine similarity, CSD style similarity, or LPIPS.

Outputs one CSV per contender (artwork ID), named <contender_id>.csv,
containing all head-to-head matches where the contender is artwork_2
and the opponent is artwork_1 in t1 keys.

Columns per row (CLIP / CSD):
    - contender_id
    - opponent_id
    - match_id
    - round_idx               (1-based within that match)
    - prompt
    - gen_img_path
    - score_to_contender      (higher is better for CLIP/CSD)
    - score_to_opponent       (higher is better for CLIP/CSD)
    - round_winner            ("contender" / "opponent" / "tie")
    - total_rounds_for_match
    - contender_win_count_for_match
    - match_winner_for_match  ("contender" if win_count > 3 else "opponent" or "no_decision")

Columns per row (LPIPS):
    - contender_id
    - opponent_id
    - match_id
    - round_idx               (1-based within that match)
    - prompt
    - gen_img_path
    - lpips_to_contender      (lower is better)
    - lpips_to_opponent       (lower is better)
    - round_winner            ("contender" / "opponent" / "tie")
    - total_rounds_for_match
    - contender_win_count_for_match
    - match_winner_for_match  ("contender" if win_count > 3 else "opponent" or "no_decision")

Rules (all metrics):
    - CLIP/CSD: higher similarity wins a round.
    - LPIPS:    lower distance wins a round.
    - If contender's win_count > 3, contender wins the match.
    - Ties do not increment either win count.
    - Invalid rounds (load failure or NaN) are skipped.

Usage examples:
    python tournament_eval.py --metric clip \\
        --t1_path /path/to/Sets_prompt_dir_dict_learning.pt \\
        --artwork_map_path /path/to/simple_maps.pt \\
        --output_dir /path/to/output

    python tournament_eval.py --metric csd \\
        --t1_path /path/to/Sets_prompt_dir_dict_learning.pt \\
        --artwork_map_path /path/to/simple_maps.pt \\
        --output_dir /path/to/output \\
        --csd_checkpoint /path/to/checkpoint.pth

    python tournament_eval.py --metric lpips \\
        --t1_path /path/to/Sets_prompt_dir_dict_learning.pt \\
        --artwork_map_path /path/to/simple_maps.pt \\
        --output_dir /path/to/output \\
        --alexnet_weights /path/to/alexnet-owt-7be5be79.pth
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
import pandas as pd

# ========================
# Argument parsing
# ========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tournament evaluation with selectable proximity metric."
    )

    # ---- Generic args ----
    parser.add_argument(
        "--metric",
        type=str,
        choices=["clip", "csd", "lpips"],
        required=True,
        help="Proximity metric to use: 'clip', 'csd', or 'lpips'.",
    )
    parser.add_argument(
        "--t1_path",
        type=str,
        required=True,
        help="Path to the .pt file containing t1 (Sets_prompt_dir_dict_*.pt).",
    )
    parser.add_argument(
        "--artwork_map_path",
        type=str,
        required=True,
        help="Path to the .pt file containing artwork_to_original (simple_maps.pt).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root directory where per-contender CSVs will be saved.",
    )

    # ---- CLIP-specific args ----
    clip_group = parser.add_argument_group("CLIP options")
    clip_group.add_argument(
        "--clip_model_id",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="HuggingFace CLIP model ID (default: openai/clip-vit-base-patch32).",
    )
    clip_group.add_argument(
        "--clip_use_amp",
        action="store_true",
        default=True,
        help="Use automatic mixed precision for CLIP inference (default: True).",
    )

    # ---- CSD-specific args ----
    csd_group = parser.add_argument_group("CSD options")
    csd_group.add_argument(
        "--csd_checkpoint",
        type=str,
        default=None,
        help="Path to the trained CSD model checkpoint (.pth).",
    )

    # ---- LPIPS-specific args ----
    lpips_group = parser.add_argument_group("LPIPS options")
    lpips_group.add_argument(
        "--alexnet_weights",
        type=str,
        default=None,
        help="Path to local AlexNet weights (alexnet-owt-7be5be79.pth) for offline use.",
    )
    lpips_group.add_argument(
        "--lpips_net",
        type=str,
        default="alex",
        help="LPIPS backbone network (default: 'alex').",
    )

    args = parser.parse_args()

    # Validate metric-specific required args
    if args.metric == "csd" and not args.csd_checkpoint:
        parser.error("--csd_checkpoint is required when --metric is 'csd'.")
    if args.metric == "lpips" and not args.alexnet_weights:
        parser.error("--alexnet_weights is required when --metric is 'lpips'.")

    return args


# ========================
# Shared constants
# ========================
EPS = 1e-6


# ========================
# CLIP helpers
# ========================

def init_clip(clip_model_id: str, device: torch.device):
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as e:
        raise RuntimeError("transformers not installed. Run: pip install transformers") from e

    model = CLIPModel.from_pretrained(clip_model_id)
    processor = CLIPProcessor.from_pretrained(clip_model_id)
    model = model.to(device).eval()
    return model, processor


_clip_embedding_cache: Dict[str, torch.Tensor] = {}


def get_clip_embedding(path: str, model, processor, device: torch.device,
                       use_amp: bool) -> torch.Tensor:
    if path in _clip_embedding_cache:
        return _clip_embedding_cache[path]

    with Image.open(path) as im:
        im = im.convert("RGB")

    inputs = processor(images=im, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                feat = model.get_image_features(pixel_values=pixel_values)
        else:
            feat = model.get_image_features(pixel_values=pixel_values)

    feat = feat / (feat.norm(p=2, dim=-1, keepdim=True) + 1e-8)
    _clip_embedding_cache[path] = feat
    return feat


def compute_clip_similarity(path_a: str, path_b: str, model, processor,
                             device: torch.device, use_amp: bool) -> float:
    try:
        emb_a = get_clip_embedding(path_a, model, processor, device, use_amp)
        emb_b = get_clip_embedding(path_b, model, processor, device, use_amp)
    except Exception as e:
        print(f"[ERROR] CLIP embedding failed:\n  A={path_a}\n  B={path_b}\n  {e}")
        return float("nan")

    emb_a = emb_a / (emb_a.norm(p=2, dim=-1, keepdim=True) + 1e-8)
    emb_b = emb_b / (emb_b.norm(p=2, dim=-1, keepdim=True) + 1e-8)
    return float((emb_a @ emb_b.T).item())


# ========================
# CSD helpers
# ========================

def init_csd(checkpoint_path: str, device: torch.device):
    # model.py must provide CSD_CLIP and convert_state_dict
    from model import CSD_CLIP, convert_state_dict
    from torchvision import transforms
    import torchvision.transforms.functional as TF

    model = CSD_CLIP("vit_large", "default")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = convert_state_dict(checkpoint["model_state_dict"])
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    )
    preprocess = transforms.Compose([
        transforms.Resize(size=224, interpolation=TF.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    return model, preprocess


_csd_embedding_cache: Dict[str, torch.Tensor] = {}


def get_csd_embedding(path: str, model, preprocess, device: torch.device) -> torch.Tensor:
    if path in _csd_embedding_cache:
        return _csd_embedding_cache[path]

    with Image.open(path) as im:
        im = im.convert("RGB")

    img = preprocess(im).unsqueeze(0).to(device)
    with torch.no_grad():
        _, style = model(img)

    style = style / (style.norm(p=2, dim=-1, keepdim=True) + 1e-8)
    _csd_embedding_cache[path] = style
    return style


def compute_csd_similarity(path_a: str, path_b: str, model, preprocess,
                            device: torch.device) -> float:
    try:
        emb_a = get_csd_embedding(path_a, model, preprocess, device)
        emb_b = get_csd_embedding(path_b, model, preprocess, device)
    except Exception as e:
        print(f"[ERROR] CSD embedding failed:\n  A={path_a}\n  B={path_b}\n  {e}")
        return float("nan")

    with torch.no_grad():
        return float((emb_a @ emb_b.T).item())


# ========================
# LPIPS helpers
# ========================

def prepare_alexnet_cache(local_path: str) -> None:
    """Copy local AlexNet weights into TorchVision hub cache to enable offline use."""
    if not os.path.isfile(local_path):
        raise FileNotFoundError(f"Local AlexNet weights not found at: {local_path}")

    torch_home = os.environ.get("TORCH_HOME", os.path.join(Path.home(), ".cache", "torch"))
    checkpoints_dir = os.path.join(torch_home, "hub", "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    dest = os.path.join(checkpoints_dir, "alexnet-owt-7be5be79.pth")

    need_copy = True
    if os.path.exists(dest):
        try:
            need_copy = os.path.getsize(local_path) != os.path.getsize(dest)
        except Exception:
            pass

    if need_copy:
        shutil.copy2(local_path, dest)
        print(f"[CACHE] Seeded TorchVision cache: {dest}")
    else:
        print(f"[CACHE] AlexNet weights already cached: {dest}")


def init_lpips(net: str, device: torch.device):
    try:
        import lpips
    except ImportError:
        raise RuntimeError("lpips not installed. Run: pip install lpips")
    loss_fn = lpips.LPIPS(net=net).to(device).eval()
    return loss_fn


def _get_hw(path: str) -> Tuple[int, int]:
    from PIL import ImageOps
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        w, h = im.size
    return h, w


def _load_tensor_lpips(path: str, device: torch.device,
                        target_size: Tuple[int, int] = None) -> torch.Tensor:
    import torchvision.transforms as T
    import torch.nn.functional as Fnn
    from PIL import ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")

    x = T.ToTensor()(im)       # [0, 1], shape [3, H, W]
    x = x * 2.0 - 1.0          # -> [-1, 1] as LPIPS expects
    x = x.unsqueeze(0).to(device)

    if target_size is not None:
        h_t, w_t = target_size
        try:
            x = Fnn.interpolate(x, size=(h_t, w_t), mode="bicubic",
                                align_corners=False, antialias=True)
        except TypeError:
            x = Fnn.interpolate(x, size=(h_t, w_t), mode="bicubic",
                                align_corners=False)
    return x


def compute_lpips_distance(path_a: str, path_b: str, loss_fn,
                            device: torch.device) -> float:
    try:
        h_a, w_a = _get_hw(path_a)
        h_b, w_b = _get_hw(path_b)
        h_t, w_t = min(h_a, h_b), min(w_a, w_b)

        xa = _load_tensor_lpips(path_a, device, target_size=(h_t, w_t))
        xb = _load_tensor_lpips(path_b, device, target_size=(h_t, w_t))

        with torch.no_grad():
            d = loss_fn(xa, xb)
            return float(d.mean().item() if isinstance(d, torch.Tensor) else d)
    except Exception as e:
        print(f"[ERROR] LPIPS failed:\n  A={path_a}\n  B={path_b}\n  {e}")
        return float("nan")


# ========================
# Round winner logic
# ========================

def decide_round_winner(score_contender: float, score_opponent: float,
                         higher_is_better: bool) -> str:
    """
    Unified round winner decision.
    - higher_is_better=True  -> used for CLIP and CSD (similarity)
    - higher_is_better=False -> used for LPIPS (distance, lower is better)
    Returns 'contender', 'opponent', or 'tie'.
    """
    if not (torch.isfinite(torch.tensor(score_contender)) and
            torch.isfinite(torch.tensor(score_opponent))):
        return "invalid"

    diff = score_contender - score_opponent
    if abs(diff) <= EPS:
        return "tie"

    if higher_is_better:
        return "contender" if diff > 0.0 else "opponent"
    else:
        return "contender" if diff < 0.0 else "opponent"


# ========================
# Main tournament loop
# ========================

def run_tournament(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Metric: {args.metric}")

    # ---- Load shared data ----
    t1 = torch.load(args.t1_path)
    artwork_to_original = torch.load(args.artwork_map_path)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Metric-specific initialisation ----
    if args.metric == "clip":
        clip_model, clip_processor = init_clip(args.clip_model_id, device)
        score_col_contender = "clip_cosine_to_contender"
        score_col_opponent  = "clip_cosine_to_opponent"
        higher_is_better    = True

    elif args.metric == "csd":
        csd_model, csd_preprocess = init_csd(args.csd_checkpoint, device)
        score_col_contender = "csd_style_to_contender"
        score_col_opponent  = "csd_style_to_opponent"
        higher_is_better    = True

    elif args.metric == "lpips":
        prepare_alexnet_cache(args.alexnet_weights)
        lpips_fn = init_lpips(args.lpips_net, device)
        score_col_contender = "lpips_to_contender"
        score_col_opponent  = "lpips_to_opponent"
        higher_is_better    = False

    # ---- Per-contender loop ----
    contenders: List[str] = list(artwork_to_original.keys())

    for contender_id in contenders:
        contender_path = artwork_to_original.get(contender_id)
        if not contender_path or not isinstance(contender_path, str):
            print(f"[WARN] Missing original path for contender: {contender_id}; skipping.")
            continue

        # Pre-warm embedding cache where applicable
        if args.metric == "clip":
            try:
                get_clip_embedding(contender_path, clip_model, clip_processor,
                                   device, args.clip_use_amp)
            except Exception as e:
                print(f"[WARN] Failed to cache CLIP embedding for contender {contender_id}: {e}")
                continue

        elif args.metric == "csd":
            try:
                get_csd_embedding(contender_path, csd_model, csd_preprocess, device)
            except Exception as e:
                print(f"[WARN] Failed to cache CSD embedding for contender {contender_id}: {e}")
                continue

        # Collect matches where second element == contender
        match_keys = [
            k for k in t1.keys()
            if isinstance(k, tuple) and len(k) == 2 and k[1] == contender_id
        ]

        all_rows: List[Dict] = []

        for match_key in match_keys:
            opponent_id   = match_key[0]
            opponent_path = artwork_to_original.get(opponent_id)
            if not opponent_path or not isinstance(opponent_path, str):
                print(f"[WARN] Missing original path for opponent: {opponent_id}; "
                      f"skipping match {match_key}.")
                continue

            # Pre-warm opponent embedding cache where applicable
            if args.metric == "clip":
                try:
                    get_clip_embedding(opponent_path, clip_model, clip_processor,
                                       device, args.clip_use_amp)
                except Exception as e:
                    print(f"[WARN] Failed to cache CLIP embedding for opponent {opponent_id}: {e}")
                    continue

            elif args.metric == "csd":
                try:
                    get_csd_embedding(opponent_path, csd_model, csd_preprocess, device)
                except Exception as e:
                    print(f"[WARN] Failed to cache CSD embedding for opponent {opponent_id}: {e}")
                    continue

            entries = t1.get(match_key, [])
            if not isinstance(entries, (list, tuple)):
                print(f"[WARN] Unexpected entries type for match {match_key}: "
                      f"{type(entries)}; skipping.")
                continue

            win_count        = 0
            round_rows_temp  = []
            round_idx        = 0

            for entry in entries:
                if not (isinstance(entry, (list, tuple)) and len(entry) >= 2):
                    print(f"[WARN] Malformed entry for match {match_key}: {entry}; "
                          "skipping round.")
                    continue

                prompt       = entry[0]
                gen_img_path = entry[-1]

                # ---- Compute similarity / distance ----
                if args.metric == "clip":
                    s_cont = compute_clip_similarity(
                        gen_img_path, contender_path,
                        clip_model, clip_processor, device, args.clip_use_amp
                    )
                    s_opp = compute_clip_similarity(
                        gen_img_path, opponent_path,
                        clip_model, clip_processor, device, args.clip_use_amp
                    )

                elif args.metric == "csd":
                    s_cont = compute_csd_similarity(
                        gen_img_path, contender_path,
                        csd_model, csd_preprocess, device
                    )
                    s_opp = compute_csd_similarity(
                        gen_img_path, opponent_path,
                        csd_model, csd_preprocess, device
                    )

                elif args.metric == "lpips":
                    s_cont = compute_lpips_distance(
                        gen_img_path, contender_path, lpips_fn, device
                    )
                    s_opp = compute_lpips_distance(
                        gen_img_path, opponent_path, lpips_fn, device
                    )

                # Skip invalid rounds
                if not (torch.isfinite(torch.tensor(s_cont)) and
                        torch.isfinite(torch.tensor(s_opp))):
                    print(f"[WARN] Invalid metric values for round. "
                          f"gen={gen_img_path}; skipping.")
                    continue

                round_idx += 1
                winner = decide_round_winner(s_cont, s_opp, higher_is_better)
                if winner == "contender":
                    win_count += 1

                row = {
                    "contender_id":       contender_id,
                    "opponent_id":        opponent_id,
                    "match_id":           opponent_id,
                    "round_idx":          round_idx,
                    "prompt":             prompt,
                    "gen_img_path":       gen_img_path,
                    score_col_contender:  s_cont,
                    score_col_opponent:   s_opp,
                    "round_winner":       winner,
                }
                round_rows_temp.append(row)

            total_rounds = round_idx
            match_winner = (
                "no_decision" if total_rounds == 0
                else ("contender" if win_count > 3 else "opponent")
            )

            for r in round_rows_temp:
                r.update({
                    "total_rounds_for_match":        total_rounds,
                    "contender_win_count_for_match": win_count,
                    "match_winner_for_match":        match_winner,
                })
                all_rows.append(r)

            print(f"[INFO] Contender {contender_id} vs Opponent {opponent_id}: "
                  f"rounds={total_rounds}, contender_wins={win_count}, "
                  f"match_winner={match_winner}")

        if not all_rows:
            print(f"[INFO] No valid rows for contender {contender_id}; skipping CSV.")
            continue

        df = pd.DataFrame(all_rows)

        columns_order = [
            "contender_id",
            "opponent_id",
            "match_id",
            "round_idx",
            "prompt",
            "gen_img_path",
            score_col_contender,
            score_col_opponent,
            "round_winner",
            "total_rounds_for_match",
            "contender_win_count_for_match",
            "match_winner_for_match",
        ]
        df = df.reindex(columns=columns_order)

        csv_path = os.path.join(args.output_dir, f"{contender_id}.csv")
        df.to_csv(csv_path, index=False)
        print(f"[SAVE] Wrote CSV for contender {contender_id}: {csv_path}")


# ========================
# Entry point
# ========================
if __name__ == "__main__":
    args = parse_args()
    run_tournament(args)
