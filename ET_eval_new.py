# pip install transformers
# pip install torch torchvision
# pip install Pillow
# pip install lpips                  # only needed for --proximity_metric lpips
# # CSD setup (only needed for --proximity_metric csd):
# #   git clone https://github.com/learn2phoenix/CSD && pip install -e CSD/
#
# Example runs:
#
#   CLIP:
#   python compute_proximity_scores.py \
#       --proximity_metric clip \
#       --mapping_json /path/to/mapping.json \
#       --output_csv /path/to/clip_scores.csv \
#       --top_save_root /path/to/candidates \
#       --orig_emb_out /path/to/original_clip_embs.pt \
#       --gen_emb_out /path/to/generated_clip_embs.pt \
#       --top_n 10
#
#   CSD:
#   python compute_proximity_scores.py \
#       --proximity_metric csd \
#       --model_path /path/to/checkpoint.pth \
#       --csd_repo_dir /content/ArtArena/CSD \
#       --mapping_json /path/to/mapping.json \
#       --output_csv /path/to/csd_scores.csv \
#       --top_save_root /path/to/candidates \
#       --orig_emb_out /path/to/original_csd_embs.pt \
#       --gen_emb_out /path/to/generated_csd_embs.pt \
#       --top_n 10
#
#   LPIPS:
#   python compute_proximity_scores.py \
#       --proximity_metric lpips \
#       --mapping_json /path/to/mapping.json \
#       --output_csv /path/to/lpips_scores.csv \
#       --top_save_root /path/to/candidates \
#       --top_n 10

import os
import json
import csv
import time
import hashlib
import shutil
import argparse
from typing import Dict, List, Tuple

import torch
from PIL import Image, ImageFile

# Prevent PIL from erroring on truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

VALID_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tiff", ".webp")

# -----------------------
# Argument Parsing
# -----------------------
parser = argparse.ArgumentParser(description="Proximity Metric Scoring Script (CLIP / CSD / LPIPS)")

# Metric selection
parser.add_argument(
    "--proximity_metric",
    type=str,
    required=True,
    choices=["clip", "csd", "lpips"],
    help="Proximity metric to use. Choices: clip | csd | lpips"
)

# Path arguments
parser.add_argument(
    "--mapping_json",
    type=str,
    required=True,
    help="Path to the JSON mapping file: {original_path: generated_path}"
)
parser.add_argument(
    "--output_csv",
    type=str,
    required=True,
    help="Path to save the output scores CSV"
)
parser.add_argument(
    "--top_save_root",
    type=str,
    default=None,
    help="Root directory to save top-N image pairs. Pass nothing to skip."
)
parser.add_argument(
    "--orig_emb_out",
    type=str,
    default=None,
    help="Path to save original embeddings .pt file (auto-derived if not set)"
)
parser.add_argument(
    "--gen_emb_out",
    type=str,
    default=None,
    help="Path to save generated embeddings .pt file (auto-derived if not set)"
)

# Behaviour flags
parser.add_argument(
    "--top_n",
    type=int,
    default=10,
    help="Number of top pairs to save (default: 10)"
)
parser.add_argument(
    "--no_rank_prefix",
    action="store_false",
    dest="copy_with_rank_prefix",
    default=True,
    help="Disable rank+hash prefix on copied filenames (prefix is ON by default)"
)
parser.add_argument(
    "--no_amp",
    action="store_true",
    default=False,
    help="Disable AMP (mixed precision). AMP is enabled by default on CUDA."
)
parser.add_argument(
    "--no_enforce_image_extensions",
    action="store_true",
    default=False,
    help="Disable image extension validation on mapping paths (default: enabled)"
)

# CSD-specific args
parser.add_argument(
    "--model_path",
    type=str,
    default=None,
    help="[CSD only] Path to the CSD model checkpoint .pth file"
)
parser.add_argument(
    "--csd_repo_dir",
    type=str,
    default=None,
    help="[CSD only] Path to the cloned CSD repo directory containing model.py and utils.py "
         "(e.g. /content/ArtArena/CSD). Added to sys.path at runtime."
)
parser.add_argument(
    "--use_cosine_similarity",
    action="store_true",
    default=False,
    help="[CSD only] L2-normalise embeddings before dot product (cosine sim). "
         "Default: raw dot product."
)

args = parser.parse_args()

# Derived boolean flags (cleaner to name positively)
USE_AMP                    = not args.no_amp
ENFORCE_IMAGE_EXTENSIONS   = not args.no_enforce_image_extensions
COPY_WITH_RANK_PREFIX      = args.copy_with_rank_prefix

# -----------------------
# Metric Initialisation
# -----------------------
if args.proximity_metric == "clip":
    from transformers import CLIPModel, CLIPProcessor
    CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

    def init_metric_model(device: torch.device):
        """Initialise CLIP model and processor."""
        model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        model.eval()
        print(f"[INFO] CLIP model '{CLIP_MODEL_ID}' loaded successfully!")
        return model, processor

    def load_embedding(model, processor, image_path: str, device: torch.device) -> torch.Tensor:
        """Load image and return L2-normalised CLIP embedding, shape [1, D]."""
        with Image.open(image_path) as im:
            im = im.convert("RGB")
        inputs = processor(images=im, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        use_amp = USE_AMP and device.type == "cuda"
        with torch.no_grad():
            if use_amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    img_feat = model.get_image_features(pixel_values=pixel_values)
            else:
                img_feat = model.get_image_features(pixel_values=pixel_values)

        img_feat = img_feat / (img_feat.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        return img_feat

    def compute_score(a: torch.Tensor, b: torch.Tensor, model=None) -> float:
        """Cosine similarity between two L2-normalised CLIP embeddings."""
        a = a / (a.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        b = b / (b.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        return float((a @ b.T).item())

    SCORE_COLUMN_NAME  = "clip_cosine_similarity"
    SORT_ASCENDING     = False   # higher cosine similarity = better
    CACHE_EMBEDDINGS   = True    # CLIP embeddings are small [1,D]; cache to disk

elif args.proximity_metric == "csd":
    import sys
    from torchvision import transforms
    import torchvision.transforms.functional as TF

    if args.model_path is None:
        raise ValueError("--model_path is required for --proximity_metric csd")

    # Add the *parent* of the CSD repo to sys.path so it can be imported
    # as a package (model.py uses relative imports like `from .utils import ...`)
    if args.csd_repo_dir is not None:
        csd_parent = os.path.dirname(os.path.abspath(args.csd_repo_dir))
        csd_pkg    = os.path.basename(os.path.abspath(args.csd_repo_dir))  # e.g. "CSD"
        sys.path.insert(0, csd_parent)
        print(f"[INFO] CSD parent dir added to sys.path: {csd_parent} (package name: '{csd_pkg}')")
    else:
        csd_pkg = "CSD"
        print("[WARN] --csd_repo_dir not set. Assuming CSD package is already on sys.path.")

    # Import as a package to support relative imports inside model.py / utils.py
    import importlib
    csd_model_mod = importlib.import_module(f"{csd_pkg}.model")
    CSD_CLIP            = csd_model_mod.CSD_CLIP
    convert_state_dict  = csd_model_mod.convert_state_dict

    # Whether to L2-normalise embeddings before dot product (cosine sim)
    # Controlled by --use_cosine_similarity flag (default: raw dot product)
    USE_COSINE_SIMILARITY = args.use_cosine_similarity

    def init_metric_model(device: torch.device):
        """Initialise CSD_CLIP model and its preprocessing pipeline."""
        model = CSD_CLIP("vit_large", "default")

        # Load checkpoint from the path provided via --model_path
        checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
        state_dict = convert_state_dict(checkpoint["model_state_dict"])
        model.load_state_dict(state_dict, strict=False)

        model = model.to(device)
        model.eval()

        # CSD-standard normalisation and transforms
        normalize = transforms.Normalize(
            (0.48145466, 0.4578275,  0.40821073),
            (0.26862954, 0.26130258, 0.27577711)
        )
        preprocess = transforms.Compose([
            transforms.Resize(size=224,
                              interpolation=TF.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        print(f"[INFO] CSD model loaded from '{args.model_path}' | "
              f"cosine_sim={USE_COSINE_SIMILARITY}")
        return model, preprocess

    def load_embedding(model, processor, image_path: str,
                       device: torch.device) -> torch.Tensor:
        """
        Load image, apply CSD preprocess pipeline, run model,
        return style embedding tensor [1, D] (kept on device).
        """
        with Image.open(image_path) as im:
            im = im.convert("RGB")
        img_tensor = processor(im).unsqueeze(0).to(device)

        use_amp = USE_AMP and device.type == "cuda"
        if use_amp:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                style_output = model(img_tensor)   # (1, D)
        else:
            style_output = model(img_tensor)       # (1, D)
        return style_output   # keep on device

    def compute_score(a: torch.Tensor, b: torch.Tensor, model=None) -> float:
        """
        CSD similarity between two style embeddings.
        - Raw dot product by default (USE_COSINE_SIMILARITY=False).
        - L2-normalised cosine similarity if USE_COSINE_SIMILARITY=True.
        Returns a Python float (higher = more similar).
        """
        if USE_COSINE_SIMILARITY:
            eps = 1e-8
            a = a / (a.norm(dim=1, keepdim=True) + eps)
            b = b / (b.norm(dim=1, keepdim=True) + eps)
        sim = a @ b.T   # (1, 1)
        return float(sim.item())

    SCORE_COLUMN_NAME  = "csd_score"
    SORT_ASCENDING     = False   # higher CSD score = more similar
    CACHE_EMBEDDINGS   = True    # CSD embeddings are small [1,D]; cache to disk

elif args.proximity_metric == "lpips":
    import torch.nn.functional as F
    import torchvision.transforms as T
    from PIL import ImageOps

    # ---- LPIPS config ----
    LPIPS_NET                  = "alex"    # "alex" | "vgg" | "squeeze"
    LPIPS_NORMALIZE_TO_MINUS1_1 = True     # LPIPS expects [-1, 1] inputs
    INTERP_MODE                = "bicubic" # interpolation when resizing orig to gen dims

    # Optional: set this to a local .pth path for offline / air-gapped environments.
    # The prepare_torchvision_cache_for_alexnet() helper will copy it into the
    # torch hub checkpoints folder so lpips can find it without downloading.
    # LOCAL_ALEXNET_WEIGHT_PATH = "/path/to/alexnet-owt-7be5be79.pth"

    to_tensor_01 = T.Compose([T.ToTensor()])  # float32 in [0,1], CxHxW

    def load_tensor_for_lpips(path: str, device: torch.device,
                              target_size: Tuple[int, int] = None) -> torch.Tensor:
        """
        Load image as a tensor [1,3,H,W] normalised for LPIPS.
        Optionally resizes to target_size=(H_target, W_target) via bicubic interpolation.
        """
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
        x = to_tensor_01(im)               # [3,H,W], float32 in [0,1]
        if LPIPS_NORMALIZE_TO_MINUS1_1:
            x = x * 2.0 - 1.0             # -> [-1,1]
        x = x.unsqueeze(0).to(device)     # [1,3,H,W]
        if target_size is not None:
            h_t, w_t = target_size
            try:
                x = F.interpolate(x, size=(h_t, w_t), mode=INTERP_MODE,
                                  align_corners=False, antialias=True)   # torch>=2.0
            except TypeError:
                x = F.interpolate(x, size=(h_t, w_t), mode=INTERP_MODE,
                                  align_corners=False)                   # fallback
        return x

    def init_metric_model(device: torch.device):
        """Initialise the LPIPS loss network."""
        try:
            import lpips
        except ImportError:
            raise RuntimeError("lpips not installed. Run: pip install lpips")
        loss_fn = lpips.LPIPS(net=LPIPS_NET)
        loss_fn = loss_fn.to(device)
        loss_fn.eval()
        print(f"[INFO] LPIPS model (net={LPIPS_NET}) loaded successfully!")
        return loss_fn, None   # no processor for LPIPS

    def load_embedding(model, processor, image_path: str,
                       device: torch.device) -> torch.Tensor:
        """
        For LPIPS 'embedding' = the raw image tensor [1,3,H,W].
        Actual resizing to match the paired image is done inside compute_score.
        """
        return load_tensor_for_lpips(image_path, device, target_size=None)

    def compute_score(a: torch.Tensor, b: torch.Tensor, model) -> float:
        """
        LPIPS distance between tensors a (orig) and b (gen).
        Resizes a to match b's spatial dimensions before scoring.
        Lower score = more perceptually similar.
        """
        if a.shape[2:] != b.shape[2:]:
            h_t, w_t = b.shape[2], b.shape[3]
            try:
                a = F.interpolate(a, size=(h_t, w_t), mode=INTERP_MODE,
                                  align_corners=False, antialias=True)
            except TypeError:
                a = F.interpolate(a, size=(h_t, w_t), mode=INTERP_MODE,
                                  align_corners=False)
        d = model(a, b)                    # lpips_fn(x, y) -> scalar tensor
        return float(d.item())

    SCORE_COLUMN_NAME  = "lpips_distance"
    SORT_ASCENDING     = True    # lower LPIPS distance = better (ascending sort)
    CACHE_EMBEDDINGS   = False   # LPIPS tensors are full [1,3,H,W]; skip disk cache

# -----------------------
# Utilities
# -----------------------
def is_nonempty_str(s) -> bool:
    return isinstance(s, str) and len(s.strip()) > 0


def sanitize_path(p):
    if not isinstance(p, str):
        return None
    p = p.strip()
    return p if p else None


def has_valid_image_extension(path: str) -> bool:
    if not ENFORCE_IMAGE_EXTENSIONS:
        return True
    _, ext = os.path.splitext(path)
    return ext.lower() in VALID_IMAGE_EXTS


def read_and_validate_mapping(mapping_json_path: str) -> Dict[str, str]:
    """Read mapping JSON and return only valid {original: generated} pairs."""
    with open(mapping_json_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Mapping JSON must be a dict of {original: generated}.")

    valid = {}
    invalid_entries = []
    for orig, gen in data.items():
        orig_clean = sanitize_path(orig)
        gen_clean  = sanitize_path(gen)

        if not is_nonempty_str(orig_clean) or not is_nonempty_str(gen_clean):
            invalid_entries.append((orig, gen, "empty_or_nonstring"))
            continue

        if ENFORCE_IMAGE_EXTENSIONS:
            if not has_valid_image_extension(orig_clean) or not has_valid_image_extension(gen_clean):
                invalid_entries.append((orig_clean, gen_clean, "bad_extension"))
                continue

        valid[orig_clean] = gen_clean

    print(f"[INFO] Mapping: total={len(data)}, valid={len(valid)}, invalid={len(invalid_entries)}")
    if invalid_entries:
        log_path = os.path.join(os.path.dirname(mapping_json_path), "mapping_invalid_entries.log")
        with open(log_path, "w", encoding="utf-8") as lf:
            for o, g, reason in invalid_entries:
                lf.write(f"[INVALID ({reason})]: original={repr(o)} generated={repr(g)}\n")
        print(f"[WARN] Invalid entries written to: {log_path}")
    return valid


def auto_embedding_paths(mapping_json_path: str, metric: str) -> Tuple[str, str]:
    """Derive default embedding save paths next to the mapping JSON."""
    base_dir   = os.path.dirname(mapping_json_path)
    base_name  = os.path.splitext(os.path.basename(mapping_json_path))[0].replace("_mapping", "")
    orig_out   = os.path.join(base_dir, f"{base_name}_original_{metric}_embs.pt")
    gen_out    = os.path.join(base_dir, f"{base_name}_generated_{metric}_embs.pt")
    return orig_out, gen_out


def short_hash(text: str, length: int = 8) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def unique_target_path(dst_dir: str, base_name: str) -> str:
    target = os.path.join(dst_dir, base_name)
    if not os.path.exists(target):
        return target
    stem, ext = os.path.splitext(base_name)
    i = 1
    while True:
        candidate = os.path.join(dst_dir, f"{stem}__{i}{ext}")
        if not os.path.exists(candidate):
            return candidate
        i += 1


def copy_with_rank_and_hash(src_path: str, dst_dir: str, rank: int) -> str:
    os.makedirs(dst_dir, exist_ok=True)
    base = os.path.basename(src_path)
    h    = short_hash(src_path)
    ranked_name = f"[rank:{rank:02d}]__{h}__{base}" if COPY_WITH_RANK_PREFIX else base
    target_path = unique_target_path(dst_dir, ranked_name)
    shutil.copy2(src_path, target_path)
    return target_path


def save_top_pairs(results: List[Tuple[str, str, float]], top_root: str, top_n: int):
    if top_root is None:
        return
    gen_dir  = os.path.join(top_root, f"top{top_n}_generated")
    orig_dir = os.path.join(top_root, f"top{top_n}_original")
    os.makedirs(gen_dir,  exist_ok=True)
    os.makedirs(orig_dir, exist_ok=True)
    print(f"[INFO] Saving top-{top_n} pairs → generated: {gen_dir} | original: {orig_dir}")
    for rank, (orig_path, gen_path, score) in enumerate(results[:top_n], start=1):
        try:
            gen_dst  = copy_with_rank_and_hash(gen_path,  gen_dir,  rank)
            orig_dst = copy_with_rank_and_hash(orig_path, orig_dir, rank)
            print(f"[TOP {rank:02d}] score={score:.6f}  gen: {gen_dst}  orig: {orig_dst}")
        except Exception as e:
            print(f"[ERROR] Failed copying pair rank={rank} ({orig_path}, {gen_path}): {e}")


# -----------------------
# Main
# -----------------------
def main():
    t0     = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device} | Metric: {args.proximity_metric} | AMP: {USE_AMP}")

    # Initialise chosen metric model
    model, processor = init_metric_model(device)

    # Read mapping
    print(f"[INFO] Reading mapping JSON: {args.mapping_json}")
    mapping = read_and_validate_mapping(args.mapping_json)

    # Resolve output paths
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    print(f"[INFO] Output CSV: {args.output_csv}")

    orig_emb_out = None
    gen_emb_out  = None
    if CACHE_EMBEDDINGS:
        orig_emb_out = args.orig_emb_out
        gen_emb_out  = args.gen_emb_out
        if orig_emb_out is None or gen_emb_out is None:
            auto_orig, auto_gen = auto_embedding_paths(args.mapping_json, args.proximity_metric)
            orig_emb_out = orig_emb_out or auto_orig
            gen_emb_out  = gen_emb_out  or auto_gen
        os.makedirs(os.path.dirname(os.path.abspath(orig_emb_out)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(gen_emb_out)),  exist_ok=True)
        print(f"[INFO] Embedding cache → original: {orig_emb_out} | generated: {gen_emb_out}")
    else:
        print(f"[INFO] Embedding caching disabled for '{args.proximity_metric}' (tensors computed per-pair)")

    if args.top_save_root:
        os.makedirs(args.top_save_root, exist_ok=True)
        print(f"[INFO] Top pairs root: {args.top_save_root}")

    orig_embeddings: Dict[str, torch.Tensor] = {}
    gen_embeddings:  Dict[str, torch.Tensor] = {}
    results: List[Tuple[str, str, float]]    = []
    errors:  List[str]                       = []

    total = len(mapping)
    print(f"[INFO] Processing {total} valid pairs...")

    with torch.no_grad():
        for idx, (orig_path, gen_path) in enumerate(mapping.items(), start=1):
            if not is_nonempty_str(orig_path) or not is_nonempty_str(gen_path):
                msg = f"Skipping invalid pair at idx={idx}"
                print(f"[WARN] {msg}"); errors.append(msg); continue

            if not os.path.isfile(orig_path):
                msg = f"Missing original file: {orig_path}"
                print(f"[WARN] {msg}"); errors.append(msg); continue

            if not os.path.isfile(gen_path):
                msg = f"Missing generated file: {gen_path}"
                print(f"[WARN] {msg}"); errors.append(msg); continue

            try:
                if CACHE_EMBEDDINGS:
                    # Load + cache to memory and disk (used by CLIP)
                    if orig_path not in orig_embeddings:
                        emb_o = load_embedding(model, processor, orig_path, device)
                        orig_embeddings[orig_path] = emb_o.detach().cpu()
                        torch.save(orig_embeddings, orig_emb_out)

                    if gen_path not in gen_embeddings:
                        emb_g = load_embedding(model, processor, gen_path, device)
                        gen_embeddings[gen_path] = emb_g.detach().cpu()
                        torch.save(gen_embeddings, gen_emb_out)

                    a = orig_embeddings[orig_path].to(device)
                    b = gen_embeddings[gen_path].to(device)
                else:
                    # Load directly without caching (used by LPIPS - tensors are large)
                    a = load_embedding(model, processor, orig_path, device)
                    b = load_embedding(model, processor, gen_path,  device)

                score = compute_score(a, b, model)
                results.append((orig_path, gen_path, score))

                if idx % 50 == 0 or idx == total:
                    print(f"[INFO] {idx}/{total} processed | last score={score:.6f}")

            except Exception as e:
                msg = f"Error at pair ({orig_path}, {gen_path}): {e}"
                print(f"[ERROR] {msg}"); errors.append(msg)

    # Sort: ascending for distance metrics (LPIPS), descending for similarity metrics (CLIP)
    results.sort(key=lambda x: x[2], reverse=not SORT_ASCENDING)

    # Write CSV
    print("[INFO] Writing CSV...")
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "original_path", "generated_path", SCORE_COLUMN_NAME])
        for rank, (orig_path, gen_path, score) in enumerate(results, start=1):
            writer.writerow([rank, orig_path, gen_path, f"{score:.8f}"])

    t1 = time.time()
    print(f"[DONE] {len(results)} rows → {args.output_csv} | Elapsed: {t1-t0:.2f}s | Errors: {len(errors)}")

    if errors:
        err_log = os.path.splitext(args.output_csv)[0] + "_errors.log"
        with open(err_log, "w", encoding="utf-8") as ef:
            ef.write("\n".join(errors))
        print(f"[INFO] Error log: {err_log}")

    save_top_pairs(results, args.top_save_root, args.top_n)


if __name__ == "__main__":
    main()
