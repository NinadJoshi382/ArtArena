'''
download the checkpoint.pt for CSD by running the below code on google colab

!pip install gdown -q

import gdown

file_id = "1FX0xs8p-C7Ob-h5Y4cUhTeOepHzXv_46"
output = "checkpoint.pt"

gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)
'''

# pip install transformers
# pip install torch torchvision
# pip install Pillow
# pip install lpips                  # only needed for --proximity_metric lpips
# # CSD setup (only needed for --proximity_metric csd):
# #   git clone https://github.com/learn2phoenix/CSD
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
#       --csd_repo_dir /path/to/CSD_repo \
#       --model_path /path/to/checkpoint.pt \
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

import contextlib
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
    help="Path to the JSON mapping file: {original_path: [generated_path, ...]} or {original_path: generated_path}"
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
    default=20,
    help="Number of top pairs to save (default: 20)"
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
    "--csd_repo_dir",
    type=str,
    default=None,
    help=(
        "[CSD only] Path to the cloned CSD repository root — the directory that "
        "contains model.py and utils.py. The script registers its parent as a "
        "Python package root so that relative imports inside model.py resolve correctly."
    ),
)
parser.add_argument(
    "--model_path",
    type=str,
    default=None,
    help="[CSD only] Path to the CSD model checkpoint .pt file"
)
parser.add_argument(
    "--use_cosine_similarity",
    action="store_true",
    default=False,
    help="[CSD only] L2-normalise embeddings before dot product (cosine sim). "
         "Default: raw dot product."
)

args = parser.parse_args()

# Derived boolean flags
USE_AMP                  = not args.no_amp
ENFORCE_IMAGE_EXTENSIONS = not args.no_enforce_image_extensions
COPY_WITH_RANK_PREFIX    = args.copy_with_rank_prefix


# -----------------------
# Shared AMP context helper
# -----------------------
def amp_context(device: torch.device):
    """
    Return the appropriate autocast context manager.
    Uses torch.amp.autocast (non-deprecated, torch >= 1.10).
    Falls back to a no-op context when AMP is disabled or device is CPU.
    """
    if USE_AMP and device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


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
        """
        Load image and return an L2-normalised CLIP image embedding, shape [1, D].

        Calls vision_model + visual_projection explicitly instead of
        get_image_features() to avoid a version-dependent return-type issue:
        newer transformers builds can return a BaseModelOutputWithPooling object
        rather than a plain tensor from get_image_features(), causing the
        subsequent .norm() call to fail with an AttributeError.
        """
        with Image.open(image_path) as im:
            im = im.convert("RGB")
        inputs = processor(images=im, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad(), amp_context(device):
            vision_outputs = model.vision_model(pixel_values=pixel_values)
            pooled_output  = vision_outputs.pooler_output            # [1, hidden_size]
            img_feat       = model.visual_projection(pooled_output)  # [1, projection_dim]

        img_feat = img_feat / (img_feat.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        return img_feat  # [1, D]

    def compute_score(a: torch.Tensor, b: torch.Tensor, model=None) -> float:
        """Cosine similarity between two L2-normalised CLIP embeddings."""
        a = a / (a.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        b = b / (b.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        return float((a @ b.T).item())

    SCORE_COLUMN_NAME = "clip_cosine_similarity"
    SORT_ASCENDING    = False  # higher cosine similarity = better
    CACHE_EMBEDDINGS  = True   # CLIP embeddings are small [1,D]; cache to disk

elif args.proximity_metric == "csd":
    # -----------------------------------------------------------------------
    # CSD imports — the repo uses relative imports (from .utils import …),
    # so we must load it as a proper Python *package* rather than a bare
    # module.  Strategy:
    #   1. Resolve the absolute path of the repo directory (--csd_repo_dir).
    #   2. Insert its *parent* directory into sys.path.
    #   3. Touch an __init__.py inside the repo dir (if absent) so Python
    #      recognises it as a package.
    #   4. Import via importlib using the package-qualified name, e.g.
    #      "CSD.model" — this makes all relative imports inside model.py
    #      resolve correctly against the package.
    # -----------------------------------------------------------------------
    import sys
    import importlib
    from torchvision import transforms
    import torchvision.transforms.functional as TF

    if args.csd_repo_dir is None:
        raise ValueError(
            "--csd_repo_dir is required for --proximity_metric csd.\n"
            "Pass the path to the cloned CSD repository root that contains model.py."
        )
    if args.model_path is None:
        raise ValueError("--model_path is required for --proximity_metric csd")

    _csd_repo_abs = os.path.abspath(args.csd_repo_dir)
    _csd_parent   = os.path.dirname(_csd_repo_abs)
    _csd_pkg_name = os.path.basename(_csd_repo_abs)

    # Make the repo importable as a package
    if _csd_parent not in sys.path:
        sys.path.insert(0, _csd_parent)

    # Create __init__.py if missing so Python treats the folder as a package
    _init_file = os.path.join(_csd_repo_abs, "__init__.py")
    if not os.path.exists(_init_file):
        open(_init_file, "w").close()
        print(f"[INFO] Created missing {_init_file} to enable package-relative imports.")

    print(f"[INFO] CSD parent dir added to sys.path: {_csd_parent} (package name: '{_csd_pkg_name}')")

    # Import model.py as part of the package so relative imports resolve correctly
    _csd_model_mod = importlib.import_module(f"{_csd_pkg_name}.model")
    CSD_CLIP       = _csd_model_mod.CSD_CLIP

    # convert_state_dict location varies by CSD version:
    #   - some builds export it from model.py
    #   - others keep it in utils.py
    # Try model.py first, then fall back to utils.py.
    if hasattr(_csd_model_mod, "convert_state_dict"):
        convert_state_dict = _csd_model_mod.convert_state_dict
        print("[INFO] CSD: convert_state_dict loaded from model.py")
    else:
        _csd_utils_mod     = importlib.import_module(f"{_csd_pkg_name}.utils")
        convert_state_dict = _csd_utils_mod.convert_state_dict
        print("[INFO] CSD: convert_state_dict loaded from utils.py")

    USE_COSINE_SIMILARITY = args.use_cosine_similarity

    def init_metric_model(device: torch.device):
        """Initialise CSD_CLIP model and its preprocessing pipeline."""
        model      = CSD_CLIP("vit_large", "default")
        # weights_only=False required: checkpoint contains numpy scalars (PyTorch 2.6+)
        checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
        state_dict = convert_state_dict(checkpoint["model_state_dict"])
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model.eval()

        normalize = transforms.Normalize(
            (0.48145466, 0.4578275,  0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        )
        preprocess = transforms.Compose([
            transforms.Resize(size=224, interpolation=TF.InterpolationMode.BICUBIC),
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
        Load image, apply CSD preprocess pipeline, run model, and return
        the STYLE embedding tensor [1, D] (kept on device).
        CSD_CLIP.forward() returns (content_output, style_output) — we take [1].
        """
        with Image.open(image_path) as im:
            im = im.convert("RGB")
        img_tensor = processor(im).unsqueeze(0).to(device)

        with amp_context(device):
            output = model(img_tensor)

        # CSD_CLIP returns (content_output, style_output); extract style embedding
        style_output = output[1] if isinstance(output, (tuple, list)) else output
        return style_output  # [1, D], kept on device

    def compute_score(a: torch.Tensor, b: torch.Tensor, model=None) -> float:
        """
        CSD similarity — raw dot product by default;
        cosine similarity when --use_cosine_similarity is set.
        Higher = more similar.
        """
        if USE_COSINE_SIMILARITY:
            eps = 1e-8
            a = a / (a.norm(dim=1, keepdim=True) + eps)
            b = b / (b.norm(dim=1, keepdim=True) + eps)
        return float((a @ b.T).item())

    SCORE_COLUMN_NAME = "csd_score"
    SORT_ASCENDING    = False  # higher CSD score = more similar
    CACHE_EMBEDDINGS  = True   # CSD embeddings are small [1,D]; cache to disk

elif args.proximity_metric == "lpips":
    import torch.nn.functional as F
    import torchvision.transforms as T
    from PIL import ImageOps

    LPIPS_NET                   = "alex"     # "alex" | "vgg" | "squeeze"
    LPIPS_NORMALIZE_TO_MINUS1_1 = True       # LPIPS expects [-1, 1] inputs
    INTERP_MODE                 = "bicubic"  # interpolation when resizing orig to gen dims

    to_tensor_01 = T.Compose([T.ToTensor()])  # float32 in [0,1], CxHxW

    def load_tensor_for_lpips(path: str, device: torch.device,
                              target_size: Tuple[int, int] = None) -> torch.Tensor:
        """
        Load image as a tensor [1,3,H,W] normalised for LPIPS.
        Optionally resizes to target_size=(H_target, W_target) via bicubic interpolation.
        """
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
        x = to_tensor_01(im)           # [3,H,W], float32 in [0,1]
        if LPIPS_NORMALIZE_TO_MINUS1_1:
            x = x * 2.0 - 1.0         # -> [-1,1]
        x = x.unsqueeze(0).to(device) # [1,3,H,W]
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
        loss_fn = lpips.LPIPS(net=LPIPS_NET).to(device)
        loss_fn.eval()
        print(f"[INFO] LPIPS model (net={LPIPS_NET}) loaded successfully!")
        return loss_fn, None  # no processor for LPIPS

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
        return float(model(a, b).item())

    SCORE_COLUMN_NAME = "lpips_distance"
    SORT_ASCENDING    = True   # lower LPIPS distance = better (ascending sort)
    CACHE_EMBEDDINGS  = False  # LPIPS tensors are full [1,3,H,W]; skip disk cache


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


def diagnose_missing_file(path: str) -> str:
    """
    When a file is not found, inspect its parent directory and return a short
    diagnostic string listing what IS there. Helps pinpoint path mismatches
    (wrong folder, wrong extension, wrong naming convention) without ls-ing manually.
    """
    parent = os.path.dirname(path)
    if not os.path.isdir(parent):
        return f"  ↳ Parent directory does not exist: {parent}"
    entries = sorted(os.listdir(parent))
    if not entries:
        return f"  ↳ Parent directory exists but is EMPTY: {parent}"
    sample = entries[:10]
    suffix = f"  … and {len(entries) - 10} more" if len(entries) > 10 else ""
    return (
        f"  ↳ Parent dir : {parent}\n"
        f"  ↳ Contents   : {sample}{suffix}"
    )


def read_and_validate_mapping(mapping_json_path: str) -> Dict[str, List[str]]:
    """
    Read mapping JSON and return only valid pairs as {original_path: [generated_path, ...]}.

    Accepted input formats:
      - list value:   {"orig.png": ["gen1.png", "gen2.png"]}
      - string value: {"orig.png": "gen.png"}  -> wrapped to ["gen.png"] automatically

    Invalid entries are skipped and written to a log file next to the mapping JSON.
    """
    with open(mapping_json_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            "Mapping JSON must be a dict of {original_path: [generated_path, ...]}."
        )

    valid: Dict[str, List[str]] = {}
    invalid_entries = []

    for orig, gen_val in data.items():
        orig_clean = sanitize_path(orig)

        if not is_nonempty_str(orig_clean):
            invalid_entries.append((orig, gen_val, "empty_or_nonstring_original"))
            continue

        if ENFORCE_IMAGE_EXTENSIONS and not has_valid_image_extension(orig_clean):
            invalid_entries.append((orig_clean, gen_val, "bad_extension_original"))
            continue

        # Normalise generated value to a flat list of strings
        if isinstance(gen_val, str):
            gen_candidates = [gen_val]
        elif isinstance(gen_val, list):
            gen_candidates = gen_val
        else:
            invalid_entries.append((orig_clean, gen_val, "generated_not_str_or_list"))
            continue

        if len(gen_candidates) == 0:
            invalid_entries.append((orig_clean, gen_val, "generated_list_is_empty"))
            continue

        valid_gens: List[str] = []
        for gen in gen_candidates:
            gen_clean = sanitize_path(gen)
            if not is_nonempty_str(gen_clean):
                invalid_entries.append((orig_clean, gen, "empty_or_nonstring_generated"))
                continue
            if ENFORCE_IMAGE_EXTENSIONS and not has_valid_image_extension(gen_clean):
                invalid_entries.append((orig_clean, gen_clean, "bad_extension_generated"))
                continue
            valid_gens.append(gen_clean)

        if not valid_gens:
            invalid_entries.append((orig_clean, gen_val, "no_valid_generated_paths"))
            continue

        valid[orig_clean] = valid_gens

    print(
        f"[INFO] Mapping: total={len(data)}, valid={len(valid)}, "
        f"invalid_entries={len(invalid_entries)}"
    )
    if invalid_entries:
        log_path = os.path.join(
            os.path.dirname(mapping_json_path), "mapping_invalid_entries.log"
        )
        with open(log_path, "w", encoding="utf-8") as lf:
            for o, g, reason in invalid_entries:
                lf.write(f"[INVALID ({reason})]: original={repr(o)} generated={repr(g)}\n")
        print(f"[WARN] Invalid entries written to: {log_path}")

    return valid


def auto_embedding_paths(mapping_json_path: str, metric: str) -> Tuple[str, str]:
    base_dir  = os.path.dirname(mapping_json_path)
    base_name = os.path.splitext(os.path.basename(mapping_json_path))[0].replace("_mapping", "")
    orig_out  = os.path.join(base_dir, f"{base_name}_original_{metric}_embs.pt")
    gen_out   = os.path.join(base_dir, f"{base_name}_generated_{metric}_embs.pt")
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
    base        = os.path.basename(src_path)
    h           = short_hash(src_path)
    ranked_name = f"[rank:{rank:02d}]__{h}__{base}" if COPY_WITH_RANK_PREFIX else base
    target_path = unique_target_path(dst_dir, ranked_name)
    shutil.copy2(src_path, target_path)
    return target_path


def save_top_pairs(
    results: List[Tuple[str, List[str], float]],
    top_root: str,
    top_n: int,
):
    if top_root is None:
        return
    gen_dir  = os.path.join(top_root, f"top{top_n}_generated")
    orig_dir = os.path.join(top_root, f"top{top_n}_original")
    os.makedirs(gen_dir,  exist_ok=True)
    os.makedirs(orig_dir, exist_ok=True)
    print(f"[INFO] Saving top-{top_n} pairs → generated: {gen_dir} | original: {orig_dir}")
    for rank, (orig_path, gen_paths, score) in enumerate(results[:top_n], start=1):
        try:
            orig_dst = copy_with_rank_and_hash(orig_path, orig_dir, rank)
            gen_dsts = [copy_with_rank_and_hash(gp, gen_dir, rank) for gp in gen_paths]
            print(
                f"[TOP {rank:02d}] avg_score={score:.6f}  "
                f"orig: {orig_dst}  gen: {gen_dsts}"
            )
        except Exception as e:
            print(f"[ERROR] Failed copying pair rank={rank} ({orig_path}): {e}")


# -----------------------
# Main
# -----------------------
def main():
    t0     = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device} | Metric: {args.proximity_metric} | AMP: {USE_AMP}")

    model, processor = init_metric_model(device)

    print(f"[INFO] Reading mapping JSON: {args.mapping_json}")
    mapping: Dict[str, List[str]] = read_and_validate_mapping(args.mapping_json)

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
        print(
            f"[INFO] Embedding caching disabled for '{args.proximity_metric}' "
            "(tensors computed per-pair)"
        )

    if args.top_save_root:
        os.makedirs(args.top_save_root, exist_ok=True)
        print(f"[INFO] Top pairs root: {args.top_save_root}")

    orig_embeddings: Dict[str, torch.Tensor] = {}
    gen_embeddings:  Dict[str, torch.Tensor] = {}
    results: List[Tuple[str, List[str], float]] = []
    errors:  List[str] = []

    total = len(mapping)
    print(f"[INFO] Processing {total} valid originals...")

    with torch.no_grad():
        for idx, (orig_path, gen_paths) in enumerate(mapping.items(), start=1):

            # ---- validate original ----
            if not is_nonempty_str(orig_path):
                msg = f"Skipping invalid original path at idx={idx}"
                print(f"[WARN] {msg}"); errors.append(msg); continue

            if not os.path.isfile(orig_path):
                diag = diagnose_missing_file(orig_path)
                msg  = f"Missing original file: {orig_path}\n{diag}"
                print(f"[WARN] {msg}"); errors.append(msg); continue

            # ---- embed the original once ----
            try:
                if CACHE_EMBEDDINGS:
                    if orig_path not in orig_embeddings:
                        emb_o = load_embedding(model, processor, orig_path, device)
                        orig_embeddings[orig_path] = emb_o.detach().cpu()
                        torch.save(orig_embeddings, orig_emb_out)
                    a = orig_embeddings[orig_path].to(device)
                else:
                    a = load_embedding(model, processor, orig_path, device)
            except Exception as e:
                msg = f"Error embedding original ({orig_path}): {e}"
                print(f"[ERROR] {msg}"); errors.append(msg); continue

            # ---- score against every generated image, then average ----
            per_gen_scores: List[float] = []

            for gen_path in gen_paths:
                if not is_nonempty_str(gen_path):
                    msg = f"Skipping empty generated path for original={orig_path}"
                    print(f"[WARN] {msg}"); errors.append(msg); continue

                if not os.path.isfile(gen_path):
                    diag = diagnose_missing_file(gen_path)
                    msg  = f"Missing generated file: {gen_path} (original={orig_path})\n{diag}"
                    print(f"[WARN] {msg}"); errors.append(msg); continue

                try:
                    if CACHE_EMBEDDINGS:
                        if gen_path not in gen_embeddings:
                            emb_g = load_embedding(model, processor, gen_path, device)
                            gen_embeddings[gen_path] = emb_g.detach().cpu()
                            torch.save(gen_embeddings, gen_emb_out)
                        b = gen_embeddings[gen_path].to(device)
                    else:
                        b = load_embedding(model, processor, gen_path, device)

                    s = compute_score(a, b, model)
                    per_gen_scores.append(s)

                except Exception as e:
                    msg = f"Error scoring pair ({orig_path}, {gen_path}): {e}"
                    print(f"[ERROR] {msg}"); errors.append(msg)

            if not per_gen_scores:
                msg = f"No valid generated scores for original={orig_path}; skipping."
                print(f"[WARN] {msg}"); errors.append(msg); continue

            avg_score = sum(per_gen_scores) / len(per_gen_scores)
            results.append((orig_path, gen_paths, avg_score))

            if idx % 50 == 0 or idx == total:
                print(
                    f"[INFO] {idx}/{total} processed | "
                    f"n_gen={len(per_gen_scores)} | avg_score={avg_score:.6f}"
                )

    # Sort: ascending for distance metrics (LPIPS), descending for similarity (CLIP/CSD)
    results.sort(key=lambda x: x[2], reverse=not SORT_ASCENDING)

    # Write CSV — one row per original; generated paths joined by "|"
    print("[INFO] Writing CSV...")
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "original_path", "generated_paths",
            "num_generated", SCORE_COLUMN_NAME,
        ])
        for rank, (orig_path, gen_paths, avg_score) in enumerate(results, start=1):
            writer.writerow([
                rank,
                orig_path,
                "|".join(gen_paths),
                len(gen_paths),
                f"{avg_score:.8f}",
            ])

    t1 = time.time()
    print(
        f"[DONE] {len(results)} rows → {args.output_csv} | "
        f"Elapsed: {t1-t0:.2f}s | Errors: {len(errors)}"
    )

    if errors:
        err_log = os.path.splitext(args.output_csv)[0] + "_errors.log"
        with open(err_log, "w", encoding="utf-8") as ef:
            ef.write("\n".join(errors))
        print(f"[INFO] Error log: {err_log}")

    save_top_pairs(results, args.top_save_root, args.top_n)


if __name__ == "__main__":
    main()
