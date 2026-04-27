"""
get_leadger.py
----------------
Tournament CSV builder for style-leakage experiments.

Usage:
    python get_leadger.py \\
        --delta 0.01 \\
        --metric clip \\
        --root_dir /path/to/tournament/results \\
        --motifs_path /path/to/motifs.json \\
        --clean_maps_path /path/to/clean_maps.pt \\
        --imi_score_path /path/to/imi_score.pt

Arguments:
    --delta           Margin threshold for the round-winner decision (e.g. 0.01).
    --metric          Similarity metric used: 'clip', 'csd', or 'lpips'.
                      For 'clip' and 'csd': higher score wins the round.
                      For 'lpips': lower score wins the round.
    --root_dir        Root directory containing per-contender CSV files.
    --motifs_path     Path to the motifs JSON file (e.g. SD_clip.json or CSD_v3.json).
    --clean_maps_path Path to the clean artwork title mapping (.pt file).
    --imi_score_path  Path to the imitation score file (.pt file).

Outputs (written to root_dir):
    match_sheet_{DELTA}.csv        -- Round-level flat sheet with motif IDs.
    wins_summary_{DELTA}.csv       -- Per-artwork win counts aggregated by role.
    winners_sheet_only_{DELTA}.csv -- Pair-level sheet with count winner + imitation winner.
"""

import os
import re
import glob
import json
import string
import unicodedata
import argparse
from collections import defaultdict
from typing import Optional, List, Set

import torch
import pandas as pd

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Tournament CSV builder for style-leakage experiments."
)
parser.add_argument(
    "--delta",
    type=float,
    required=True,
    help="Margin threshold for the round-winner decision (e.g. 0.01).",
)
parser.add_argument(
    "--metric",
    type=str,
    required=True,
    choices=["clip", "csd", "lpips"],
    help=(
        "Similarity metric used to score rounds. "
        "'clip' and 'csd': higher score wins. "
        "'lpips': lower score wins."
    ),
)
parser.add_argument(
    "--root_dir",
    type=str,
    required=True,
    help="Root directory containing per-contender CSV files (tournament results).",
)
parser.add_argument(
    "--motifs_path",
    type=str,
    required=True,
    help=(
        "Path to the motifs JSON file. "
        "Examples: SD_clip.json, dict_prompts_by_map_id_CSD_v3.json"
    ),
)
parser.add_argument(
    "--clean_maps_path",
    type=str,
    required=True,
    help="Path to the clean artwork title mapping (.pt file, e.g. clean_maps.pt).",
)
parser.add_argument(
    "--imi_score_path",
    type=str,
    required=True,
    help="Path to the imitation score file (.pt file, e.g. imi_score.pt).",
)
arg = parser.parse_args()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT_TOURNAMENT_DIR: str = arg.root_dir
ARTWORKS: List[str] = [f"A{i}" for i in range(1, 21)]

# Decision margin between metric scores.
# For clip / csd: if (opponent - contender) > DELTA -> Challenger wins the round.
#                 if (contender - opponent) > DELTA -> Defender wins the round.
# For lpips:      if (contender - opponent) > DELTA -> Challenger wins the round.
#                 if (opponent - contender) > DELTA -> Defender wins the round.
# Otherwise -> Tie.
DELTA: float = arg.delta
METRIC: str = arg.metric

# ---------------------------------------------------------------------------
# Paths for the dictionaries
# ---------------------------------------------------------------------------

MOTIFS_PATH: str     = arg.motifs_path
CLEAN_MAPS_PATH: str = arg.clean_maps_path
IMI_SCORE_PATH: str  = arg.imi_score_path

# ---------------------------------------------------------------------------
# Load dictionaries
# ---------------------------------------------------------------------------

try:
    with open(MOTIFS_PATH, "r") as file:
        t1 = json.load(file)  # motifs dict: { "A1": [...], ... }
except Exception as e:
    print(f"[WARN] Failed to load motifs dict at {MOTIFS_PATH}: {e}")
    t1 = {}

try:
    clean_mapping = torch.load(CLEAN_MAPS_PATH)
except Exception as e:
    print(f"[WARN] Failed to load clean mapping at {CLEAN_MAPS_PATH}: {e}")
    clean_mapping = {}

# ---------------------------------------------------------------------------
# Winner normalization
# ---------------------------------------------------------------------------

def _normalize_winner(tag: Optional[str]) -> Optional[str]:
    """
    Map 'contender'/'opponent' (or their synonyms) to 'Defender'/'Challenger'.
    Also recognize tie / no-decision variants.
    """
    if tag is None:
        return None
    t = str(tag).strip().lower()
    if t in {"contender", "defender"}:
        return "Defender"
    if t in {"opponent", "challenger"}:
        return "Challenger"
    if t in {"tie", "draw"}:
        return "Tie"
    if t in {"no_decision", "no decision", "none", ""}:
        return "No decision"
    return None


def _infer_winner(def_score, chal_score) -> Optional[str]:
    """Infer winner from numeric scores; return None if not comparable."""
    try:
        ds = float(def_score)
        cs = float(chal_score)
    except Exception:
        return None
    if pd.isna(ds) or pd.isna(cs):
        return None
    if ds > cs:
        return "Defender"
    if cs > ds:
        return "Challenger"
    return "Tie"


def _round_winner_by_delta(
    def_score,
    chal_score,
    delta: float,
    metric: str = "clip",
) -> Optional[str]:
    """
    Decide round winner by margin (delta).

    For 'clip' and 'csd' (higher score is better):
        if (chal - def) > delta -> 'Challenger'
        if (def - chal) > delta -> 'Defender'
        else                    -> 'Tie'

    For 'lpips' (lower score is better — smaller distance means more similar):
        if (def - chal) > delta -> 'Challenger'  (challenger is closer / lower)
        if (chal - def) > delta -> 'Defender'    (defender is closer / lower)
        else                    -> 'Tie'

    Returns None only if scores are non-numeric or missing.
    """
    ds = pd.to_numeric(def_score, errors="coerce")
    cs = pd.to_numeric(chal_score, errors="coerce")
    if pd.isna(ds) or pd.isna(cs):
        return None

    if metric in {"clip", "csd"}:
        # Higher score wins
        if (cs - ds) > delta:
            return "Challenger"
        if (ds - cs) > delta:
            return "Defender"
    elif metric == "lpips":
        # Lower score wins (smaller perceptual distance = more similar)
        if (ds - cs) > delta:
            return "Challenger"
        if (cs - ds) > delta:
            return "Defender"
    else:
        raise ValueError(f"Unknown metric '{metric}'. Expected 'clip', 'csd', or 'lpips'.")

    return "Tie"

# ---------------------------------------------------------------------------
# Motif helpers
# ---------------------------------------------------------------------------

WS_RE = re.compile(r"\s+")


def _normalize_motif_text(s: Optional[str]) -> str:
    """Trim, collapse whitespace, strip quotes, and drop trailing period."""
    if not s or not isinstance(s, str):
        return ""
    s = WS_RE.sub(" ", s.strip())
    s = s.strip("\"'")
    if s.endswith("."):
        s = s[:-1].strip()
    return s


def _extract_prompt_head_before_style(prompt: Optional[str]) -> Optional[str]:
    """Take the part of prompt before 'in the style of' (case-insensitive)."""
    if not prompt or not isinstance(prompt, str):
        return None
    src = prompt.strip()
    low = src.lower()
    keys = [" in the style of", " in style of"]
    idx = -1
    for k in keys:
        idx = low.find(k)
        if idx != -1:
            break
    if idx == -1:
        return None
    head = src[:idx]
    head = _normalize_motif_text(head)
    return head or None

# ---------------------------------------------------------------------------
# Robust fuzzy matching utilities
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "the", "with", "and", "or", "of", "to", "in", "on", "for", "by",
    "featuring", "showing", "depicting", "depiction", "scene", "visible"
}

_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})


def _ascii_fold(s: str) -> str:
    """Normalize unicode accents and smart quotes to ASCII."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def _normalize_for_tokens(s: str) -> str:
    """Stronger normalization for token-based similarity."""
    s = _ascii_fold(s)
    s = s.lower().translate(_PUNCT_TABLE)
    s = WS_RE.sub(" ", s).strip()
    return s


def _tokenize(s: str) -> List[str]:
    if not s:
        return []
    tokens = _normalize_for_tokens(s).split()
    return [t for t in tokens if t not in _STOPWORDS]


def _levenshtein_ratio(a: str, b: str) -> float:
    """Character-level similarity ratio using Levenshtein distance."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        ca = a[i - 1]
        for j in range(1, lb + 1):
            temp = dp[j]
            cost = 0 if ca == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,       # deletion
                dp[j - 1] + 1,   # insertion
                prev + cost      # substitution
            )
            prev = temp
    dist = dp[lb]
    return 1.0 - dist / max(la, lb)


def _jaccard(tokens_a, tokens_b) -> float:
    sa, sb = set(tokens_a), set(tokens_b)
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _dice(tokens_a, tokens_b) -> float:
    sa, sb = set(tokens_a), set(tokens_b)
    if not sa and not sb:
        return 1.0
    inter = 2 * len(sa & sb)
    denom = len(sa) + len(sb)
    return inter / denom if denom else 0.0


def _partial_token_window_ratio(a_tokens, b_tokens) -> float:
    """
    Compare the smaller side to sliding windows of the larger side.
    Joins tokens into strings to allow character-level partial match.
    """
    if not a_tokens or not b_tokens:
        return 0.0
    small, large = (
        (a_tokens, b_tokens) if len(a_tokens) <= len(b_tokens)
        else (b_tokens, a_tokens)
    )
    small_s = " ".join(small)
    best = 0.0
    n = len(small)
    for i in range(0, len(large) - n + 1):
        window = " ".join(large[i:i + n])
        best = max(best, _levenshtein_ratio(small_s, window))
        if best >= 0.99:
            break
    return best


def _composite_similarity(head: str, motif: str) -> float:
    """
    Blend several signals:
        1) strict normalized equality boost
        2) character-level ratio
        3) token Jaccard and Dice
        4) partial token-window ratio for local alignment
    """
    h_norm = _normalize_motif_text(head)
    m_norm = _normalize_motif_text(motif)
    if h_norm and m_norm and h_norm == m_norm:
        return 1.0

    char_sim = _levenshtein_ratio(h_norm, m_norm)  # noqa: F841 (available for future blends)
    ht = _tokenize(head)
    mt = _tokenize(motif)
    jacc = _jaccard(ht, mt)

    # For current use we rely on Jaccard only
    score = jacc  # could be a blend if needed
    return score

# ---------------------------------------------------------------------------
# Motif ID finder
# ---------------------------------------------------------------------------

def _find_motif_id_for_row(
    defender_id: str,
    prompt: Optional[str],
    motifs_dict: dict,
    used_motif_ids: Optional[Set[int]] = None,
) -> Optional[int]:
    """
    Find motif_ID as a 1-based index into motifs_dict[defender_id] by matching
    the prompt head (before 'in the style of') to one of the motif strings.

    Updated behavior:
        - No thresholding. Always select the highest-scoring motif.
        - Avoid repetition of motif_id for the same defender_id by skipping
          any motif present in 'used_motif_ids' (1-based) if supplied.
    """
    if not defender_id or defender_id not in motifs_dict:
        return None

    head = _extract_prompt_head_before_style(prompt)
    if not head:
        return None

    motif_list = motifs_dict.get(defender_id, [])
    if not isinstance(motif_list, list) or not motif_list:
        return None

    target_text = _normalize_motif_text(head)

    # Compute similarity scores for all motifs
    scores = []
    for i, m in enumerate(motif_list):
        try:
            s = _composite_similarity(target_text, m)
        except Exception:
            s = float("-inf")
        scores.append((i, s))

    # Sort by highest score first; deterministic tie-break by index
    scores.sort(key=lambda t: (-t[1], t[0]))

    # Convert provided used 1-based IDs to zero-based indices
    used_zero_based: Set[int] = set()
    if used_motif_ids:
        used_zero_based = {
            mid - 1 for mid in used_motif_ids
            if isinstance(mid, int) and mid > 0
        }

    # Pick the first highest-scoring motif that is not already used
    for idx, _ in scores:
        if idx not in used_zero_based:
            return idx + 1  # convert back to 1-based

    # All motifs for this defender have been used already
    return None


def _map_artwork_name(art_id: str) -> str:
    """Replace raw artwork ID with clean title if available."""
    return clean_mapping.get(art_id, art_id)

# ---------------------------------------------------------------------------
# Round-level CSV builder
# ---------------------------------------------------------------------------

def build_match_sheet_round_level(
    root_dir: str,
    out_csv_name: str = "match_sheet.csv",
    restrict_to_artworks: Optional[List[str]] = ARTWORKS,
    drop_ties: bool = False,
) -> pd.DataFrame:
    """
    Produce a flat CSV across all input files with columns:
        Defender artwork | Challenger artwork | prompt |
        Defender score | Challenger score | who won? | motif_ID

    Winner decision is recomputed per round using margin DELTA:
        if (opponent - contender) > DELTA -> Challenger
        if (contender - opponent) > DELTA -> Defender
        else -> Tie
    """
    csv_paths = glob.glob(os.path.join(root_dir, "*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {root_dir}")

    out_rows = []

    for path in csv_paths:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[WARN] Failed to read {path}: {e}")
            continue

        missing = [
            c for c in [
                "contender_id", "opponent_id", "prompt",
                "clip_cosine_to_contender", "clip_cosine_to_opponent",
            ]
            if c not in df.columns
        ]
        if missing:
            print(f"[WARN] {os.path.basename(path)} missing columns {missing}. Skipping.")
            continue

        for _, row in df.iterrows():
            defender = str(row["contender_id"])
            challenger = str(row["opponent_id"])

            # Filter using raw IDs
            if restrict_to_artworks is not None:
                if defender not in restrict_to_artworks or challenger not in restrict_to_artworks:
                    continue

            prompt_val = row["prompt"] if pd.notna(row["prompt"]) else None
            def_score = row["clip_cosine_to_contender"]
            chal_score = row["clip_cosine_to_opponent"]

            # Parse to float for clean output when possible
            try:
                def_score_out = float(pd.to_numeric(def_score, errors="coerce"))
            except Exception:
                def_score_out = def_score
            try:
                chal_score_out = float(pd.to_numeric(chal_score, errors="coerce"))
            except Exception:
                chal_score_out = chal_score

            # Recompute round winner strictly from score margin, DELTA, and METRIC
            winner = _round_winner_by_delta(def_score_out, chal_score_out, DELTA, METRIC)

            if drop_ties and winner in {"Tie", "No decision", None}:
                continue

            # Compute motif_ID using challenger as the lookup key
            motif_id = _find_motif_id_for_row(challenger, prompt_val, t1)

            out_rows.append({
                "Defender artwork": _map_artwork_name(defender),
                "Challenger artwork": _map_artwork_name(challenger),
                "prompt": prompt_val,
                "Defender score": def_score_out,
                "Challenger score": chal_score_out,
                "who won?": winner,
                "motif_ID": motif_id,
            })

    out_df = pd.DataFrame(out_rows).drop_duplicates().reset_index(drop=True)

    out_path = os.path.join(root_dir, out_csv_name)
    out_df.to_csv(out_path, index=False)
    print(f"[SAVE] Match sheet written to {out_path}")
    print(f"[INFO] Rows: {len(out_df)}")
    return out_df

# ---------------------------------------------------------------------------
# Wins summary builder
# ---------------------------------------------------------------------------

def build_wins_summary(
    root_dir: str,
    match_csv_name: str = "match_sheet.csv",
    out_csv_name: str = "wins_summary.csv",
) -> pd.DataFrame:
    """
    Read the round-level match CSV and aggregate match wins by artwork and role.
    """
    match_path = os.path.join(root_dir, match_csv_name)
    if not os.path.exists(match_path):
        raise FileNotFoundError(
            f"Round-level CSV not found at {match_path}. "
            "Run build_match_sheet_round_level first."
        )

    df = pd.read_csv(match_path)

    required_id_cols = {"Defender artwork", "Challenger artwork"}
    if not required_id_cols.issubset(df.columns):
        raise ValueError(
            "Missing required columns. Expected columns include "
            "'Defender artwork' and 'Challenger artwork'."
        )

    # Prefer 'who won?' if present; otherwise fall back to match-level column.
    winner_col = None
    if "who won?" in df.columns:
        winner_col = "who won?"
    elif "match_winner_for_match" in df.columns:
        winner_col = "match_winner_for_match"
    else:
        raise ValueError(
            "No winner column found. Expected one of 'who won?' or 'match_winner_for_match'."
        )

    def normalize_winner(val):
        if pd.isna(val):
            return ""
        s = str(val).strip().lower()
        if s in {"defender", "d", "contender"}:
            return "defender"
        if s in {"challenger", "c", "opponent"}:
            return "challenger"
        if s in {"tie", "draw"}:
            return "tie"
        if s in {"no_decision", "no decision", "nd", "n/a"}:
            return "no_decision"
        return s

    df["_winner_tag"] = df[winner_col].map(normalize_winner)

    grp = (
        df.groupby(["Defender artwork", "Challenger artwork"], as_index=False)["_winner_tag"]
        .first()
    )

    wins_defender: defaultdict = defaultdict(int)
    wins_challenger: defaultdict = defaultdict(int)

    for _, row in grp.iterrows():
        defender_art = str(row["Defender artwork"])
        challenger_art = str(row["Challenger artwork"])
        tag = row["_winner_tag"]

        if tag in {"tie", "no_decision", ""}:
            continue

        if tag == "defender":
            wins_defender[defender_art] += 1
        elif tag == "challenger":
            wins_challenger[challenger_art] += 1
        else:
            print(
                f"[WARN] Unknown winner tag '{tag}' for match "
                f"({defender_art}, {challenger_art}). Skipping."
            )

    all_artworks = sorted(
        set(list(wins_defender.keys()) + list(wins_challenger.keys()))
    )
    out = pd.DataFrame({"artwork": all_artworks})
    out["wins as Defender"] = out["artwork"].map(lambda a: wins_defender.get(a, 0)).astype(int)
    out["wins as Challenger"] = out["artwork"].map(lambda a: wins_challenger.get(a, 0)).astype(int)
    out["total wins"] = out["wins as Defender"] + out["wins as Challenger"]

    out = out.sort_values(
        ["total wins", "artwork"], ascending=[False, True]
    ).reset_index(drop=True)

    out_path = os.path.join(root_dir, out_csv_name)
    out.to_csv(out_path, index=False)
    print(f"[SAVE] Wins summary written to {out_path}")
    print(f"[INFO] Rows: {len(out)}")
    return out

# ---------------------------------------------------------------------------
# Winners (pair-level) sheet builder
# ---------------------------------------------------------------------------

def build_winners_sheet_only(
    root_dir: str,
    match_csv_name: str = "match_sheet.csv",
    out_csv_name: str = "winners_sheet_only.csv",
    imi_score_path: str = IMI_SCORE_PATH,
) -> pd.DataFrame:
    """
    Produce a pair-level CSV (one row per (Defender, Challenger)) with:
        Defender | Challenger | who won the count | Imitation win

    who won the count:
        Count round-level wins from 'who won?' for each pair.
        defender_wins  = count of 'Defender'
        challenger_wins = count of 'Challenger'
        If defender_wins > challenger_wins  -> 'Defender'
        If challenger_wins > defender_wins  -> 'Challenger'
        If equal                            -> 'Tie'

    Imitation win:
        Load imi_score.pt (keys are clean artwork titles).
        If both scores present: higher score wins; equal -> 'Tie'; missing -> 'Unknown'.
    """
    match_path = os.path.join(root_dir, match_csv_name)
    if not os.path.exists(match_path):
        raise FileNotFoundError(
            f"Round-level CSV not found at {match_path}. "
            "Run build_match_sheet_round_level first."
        )

    df = pd.read_csv(match_path)

    # Validate required columns
    req_cols = {"Defender artwork", "Challenger artwork", "who won?"}
    if not req_cols.issubset(df.columns):
        missing = req_cols - set(df.columns)
        raise ValueError(f"Expected columns missing in match CSV: {missing}")

    # Normalize winner tags from round-level 'who won?'
    def _norm_round_winner(val: str) -> str:
        if pd.isna(val):
            return ""
        s = str(val).strip().lower()
        if s in {"defender", "d", "contender"}:
            return "defender"
        if s in {"challenger", "c", "opponent"}:
            return "challenger"
        if s in {"tie", "draw"}:
            return "tie"
        if s in {"no_decision", "no decision", "nd", "n/a"}:
            return "no_decision"
        return ""

    df["_winner_round_norm"] = df["who won?"].map(_norm_round_winner)

    # Base pairs set — ensures inclusion even if only ties or no-decisions
    pairs = df[["Defender artwork", "Challenger artwork"]].drop_duplicates()

    # Count decisive wins per pair using round-level outcomes only
    decisive = df[df["_winner_round_norm"].isin(["defender", "challenger"])].copy()
    counts = (
        decisive
        .groupby(["Defender artwork", "Challenger artwork", "_winner_round_norm"])
        .size()
        .unstack("_winner_round_norm", fill_value=0)
        .reset_index()
    )

    # Ensure both columns exist even if absent in data
    if "defender" not in counts.columns:
        counts["defender"] = 0
    if "challenger" not in counts.columns:
        counts["challenger"] = 0

    # Merge with full pairs to include pairs with zero decisive rounds
    merged = pairs.merge(
        counts, on=["Defender artwork", "Challenger artwork"], how="left"
    )
    merged["defender"] = merged["defender"].fillna(0).astype(int)
    merged["challenger"] = merged["challenger"].fillna(0).astype(int)

    def _who_won_count(dw: int, cw: int) -> str:
        if dw > cw:
            return "Defender"
        if cw > dw:
            return "Challenger"
        return "Tie"

    merged["who won the count"] = merged.apply(
        lambda r: _who_won_count(r["defender"], r["challenger"]), axis=1
    )

    # Load imitation scores
    try:
        imi_scores = torch.load(imi_score_path)
        if not isinstance(imi_scores, dict):
            raise TypeError("imi_score.pt did not load to a dict.")
    except Exception as e:
        raise RuntimeError(f"Failed to load imitation scores at {imi_score_path}: {e}")

    # Compute imitation win based on clean titles
    def _imi_winner(def_name: str, chal_name: str) -> str:
        sd = imi_scores.get(str(def_name), None)
        sc = imi_scores.get(str(chal_name), None)
        if sd is None or sc is None:
            return "Unknown"
        try:
            sd_f = float(sd)
            sc_f = float(sc)
        except Exception:
            return "Unknown"
        if sd_f > sc_f:
            return "Defender"
        if sc_f > sd_f:
            return "Challenger"
        return "Tie"

    merged["Imitation win"] = merged.apply(
        lambda r: _imi_winner(r["Defender artwork"], r["Challenger artwork"]),
        axis=1,
    )

    # Final output columns
    out_df = merged.rename(columns={
        "Defender artwork": "Defender",
        "Challenger artwork": "Challenger",
    })[["Defender", "Challenger", "who won the count", "Imitation win"]]

    out_path = os.path.join(root_dir, out_csv_name)
    out_df.to_csv(out_path, index=False)
    print(f"[SAVE] Winners sheet written to {out_path}")
    print(f"[INFO] Rows: {len(out_df)}")
    return out_df

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(
        f"[CONFIG] metric={METRIC} | delta={DELTA} | root_dir={ROOT_TOURNAMENT_DIR}\n"
        f"         motifs_path={MOTIFS_PATH}\n"
        f"         clean_maps_path={CLEAN_MAPS_PATH}\n"
        f"         imi_score_path={IMI_SCORE_PATH}"
    )

    ms_path = os.path.join(ROOT_TOURNAMENT_DIR, f"match_sheet_{DELTA}.csv")
    ws_path = os.path.join(ROOT_TOURNAMENT_DIR, f"wins_summary_{DELTA}.csv")

    # 1) Build round-level sheet if missing
    if not os.path.exists(ms_path):
        build_match_sheet_round_level(
            root_dir=ROOT_TOURNAMENT_DIR,
            out_csv_name=f"match_sheet_{DELTA}.csv",
            restrict_to_artworks=ARTWORKS,
            drop_ties=False,
        )
    else:
        print(f"[SKIP] {ms_path} already exists. Not rebuilding.")

    # 2) Aggregate wins by artwork and role if missing
    if not os.path.exists(ws_path):
        build_wins_summary(
            root_dir=ROOT_TOURNAMENT_DIR,
            match_csv_name=f"match_sheet_{DELTA}.csv",
            out_csv_name=f"wins_summary_{DELTA}.csv",
        )
    else:
        print(f"[SKIP] {ws_path} already exists. Not rebuilding.")

    # 3) Build winners_sheet_only.csv (always refresh)
    build_winners_sheet_only(
        root_dir=ROOT_TOURNAMENT_DIR,
        match_csv_name=f"match_sheet_{DELTA}.csv",
        out_csv_name=f"winners_sheet_only_{DELTA}.csv",
        imi_score_path=IMI_SCORE_PATH,
    )
