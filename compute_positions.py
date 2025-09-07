import argparse
import hashlib
import json
import os
import re
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import emoji  # pypi "emoji"
from sentence_transformers import SentenceTransformer
import umap


def build_canonical_emojis() -> Tuple[List[str], List[str]]:
    """Return (emojis, labels) with default/yellow skin and gender-neutral if available,
    otherwise man variant; skip keycaps/variation selectors; de-duplicate groups.
    """
    all_emoji = list(emoji.EMOJI_DATA.keys())

    # Drop any graphemes containing Fitzpatrick skin tone modifiers to avoid
    # light/medium/dark duplicates at the source.
    def has_skin_tone_cp(e: str) -> bool:
        return any(0x1F3FB <= ord(ch) <= 0x1F3FF for ch in e)

    base_emoji = [e for e in all_emoji if not has_skin_tone_cp(e)]

    tone_pat = re.compile(r"_(light|medium_light|medium|medium_dark|dark)_skin_tone")

    def alias_of(e: str):
        a = emoji.demojize(e)
        if not (a.startswith(":") and a.endswith(":")):
            return None
        return a.strip(":")

    def strip_tone(alias: str) -> str:
        return tone_pat.sub("", alias)

    def group_key(alias: str) -> str:
        toks = alias.split("_")
        mapped = [
            ("person" if t in {"man", "men", "boy", "woman", "women", "girl"} else t)
            for t in toks
        ]
        return "_".join(mapped)

    groups: "OrderedDict[str, Dict[str, dict]]" = OrderedDict()
    for e in base_emoji:
        a = alias_of(e)
        if not a:
            continue
        if re.search(r"keycap|variation selector", a, re.I):
            continue
        # Extra guard: skip anything that still references skin tone textually
        if re.search(r"skin\s*tone", a, re.I):
            continue
        a0 = strip_tone(a)
        gk = group_key(a0)
        entry = groups.get(gk)
        if entry is None:
            entry = {"person": None, "man": None, "any": None}
            groups[gk] = entry
        tokens = set(a0.split("_"))
        if "person" in tokens and entry["person"] is None:
            entry["person"] = {"alias": a0, "e": e}
        elif "man" in tokens and entry["man"] is None:
            entry["man"] = {"alias": a0, "e": e}
        elif entry["any"] is None:
            entry["any"] = {"alias": a0, "e": e}

    selected = []
    for entry in groups.values():
        choice = entry["person"] or entry["man"] or entry["any"]
        if not choice:
            continue
        try:
            e = emoji.emojize(f":{choice['alias']}:")
            if e.startswith(":") and e.endswith(":"):
                e = choice["e"]
        except Exception:
            e = choice["e"]
        selected.append((choice["alias"], e))

    emjs = [e for _, e in selected]
    labels = [alias.replace("_", " ").replace("-", " ") for alias, _ in selected]
    return emjs, labels


def compute_positions(emjs: List[str], labels: List[str], *,
                      model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                      n_neighbors: int = 30,
                      min_dist: float = 0.25,
                      metric: str = "cosine",
                      seed: int = 42) -> np.ndarray:
    model = SentenceTransformer(model_name)
    emb = model.encode(labels, normalize_embeddings=True, batch_size=256, show_progress_bar=True)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=seed)
    proj = reducer.fit_transform(emb)
    return proj


def write_cache(out_path: str, emjs: List[str], labels: List[str], proj: np.ndarray,
                meta: dict) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "meta": meta,
        "emoji": emjs,
        "label": labels,
        "x": proj[:, 0].astype(float).tolist(),
        "y": proj[:, 1].astype(float).tolist(),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"wrote {out_path}")


def load_cache(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Compute and cache 2D emoji positions")
    ap.add_argument("--out", default="data/emoji_positions.json", help="Output JSON path")
    ap.add_argument("--force", action="store_true", help="Recompute even if cache exists and matches")
    args = ap.parse_args()

    out_path = args.out

    emjs, labels = build_canonical_emojis()

    # Compute a signature over inputs and params to validate cache
    umap_params = {"n_neighbors": 30, "min_dist": 0.25, "metric": "cosine", "seed": 42}
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    sig_src = json.dumps({
        "emoji": emjs,
        "labels": labels,
        "umap": umap_params,
        "model": model_name,
        "emoji_lib": getattr(emoji, "__version__", "unknown"),
    }, ensure_ascii=False)
    sig = hashlib.sha256(sig_src.encode("utf-8")).hexdigest()

    if not args.force:
        cached = load_cache(out_path)
        if cached and cached.get("meta", {}).get("signature") == sig:
            print(f"cache up-to-date at {out_path}; skip recompute")
            return

    proj = compute_positions(emjs, labels,
                             model_name=model_name,
                             n_neighbors=umap_params["n_neighbors"],
                             min_dist=umap_params["min_dist"],
                             metric=umap_params["metric"],
                             seed=umap_params["seed"])

    meta = {
        "signature": sig,
        "umap": umap_params,
        "model": model_name,
        "count": len(emjs),
    }
    write_cache(out_path, emjs, labels, proj, meta)


if __name__ == "__main__":
    main()
