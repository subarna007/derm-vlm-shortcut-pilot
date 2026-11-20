"""Read Derm7pt and PAD-UFES-20 and build one sample manifest with shared labels."""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from . import config as C


def derm7pt_diagnosis(raw: str):
    d = str(raw).strip().lower()
    if "melanoma" in d:
        return "melanoma"
    if "nevus" in d:
        return "nevus"
    if "basal cell carcinoma" in d:
        return "basal cell carcinoma"
    if "seborrheic keratosis" in d:
        return "seborrheic keratosis"
    if "dermatofibroma" in d:
        return "dermatofibroma"
    if "vascular lesion" in d:
        return "vascular lesion"
    return None  # lentigo, melanosis, miscellaneous: not in the shared label set


def _norm(v):
    return str(v).strip().lower()


def derm7pt_criteria(row) -> dict:
    """Turn Derm7pt's attribute columns into the seven binary checklist criteria.

    Vascular structures follow the grouping of Kawahara et al. (2019): dotted and
    linear irregular vessels are irregular, the other named patterns are regular.
    """
    vs = _norm(row["vascular_structures"])
    return {
        "atypical_network": int(_norm(row["pigment_network"]) == "atypical"),
        "blue_white_veil": int(_norm(row["blue_whitish_veil"]) == "present"),
        "atypical_vascular": int(vs in {"dotted", "linear irregular"}),
        "irregular_streaks": int(_norm(row["streaks"]) == "irregular"),
        "irregular_pigmentation": int("irregular" in _norm(row["pigmentation"])),
        "irregular_dots": int(_norm(row["dots_and_globules"]) == "irregular"),
        "regression": int(_norm(row["regression_structures"]) != "absent"),
    }


def _find(base: Path, names):
    for n in names:
        p = base / n
        if p.exists():
            return p
    hits = [p for n in names for p in base.rglob(Path(n).name)]
    return hits[0] if hits else None


def load_derm7pt(base=C.DERM7PT_DIR) -> pd.DataFrame:
    base = Path(base)
    meta = _find(base, ["meta/meta.csv", "release_v0/meta/meta.csv", "meta.csv"])
    if meta is None:
        raise FileNotFoundError(f"meta.csv not found under {base}")
    img_root = meta.parent.parent / "images"
    df = pd.read_csv(meta)
    rows = []
    for _, r in df.iterrows():
        dx = derm7pt_diagnosis(r["diagnosis"])
        if dx is None:
            continue
        path = img_root / str(r["derm"])
        if not path.exists():
            continue
        rows.append({
            "image_id": f"d7_{int(r['case_num']):04d}",
            "dataset": "derm7pt",
            "modality": "dermoscopic",
            "path": str(path),
            "label": dx,
            "raw_label": r["diagnosis"],
            **derm7pt_criteria(r),
        })
    return pd.DataFrame(rows)


PAD_MAP = {
    "MEL": "melanoma",
    "NEV": "nevus",
    "BCC": "basal cell carcinoma",
    "SCC": "squamous cell carcinoma",
    "ACK": "actinic keratosis",
    "SEK": "seborrheic keratosis",
}


def load_pad(base=C.PAD_DIR) -> pd.DataFrame:
    base = Path(base)
    meta = _find(base, ["metadata.csv"])
    if meta is None:
        raise FileNotFoundError(f"metadata.csv not found under {base}")
    df = pd.read_csv(meta)
    files = {p.name: p for p in base.rglob("*.png")}
    rows = []
    for _, r in df.drop_duplicates("lesion_id").iterrows():  # one image per lesion
        dx = PAD_MAP.get(str(r["diagnostic"]).strip().upper())
        p = files.get(str(r["img_id"]))
        if dx is None or p is None:
            continue
        rows.append({
            "image_id": "pad_" + Path(str(r["img_id"])).stem,
            "dataset": "pad",
            "modality": "clinical",
            "path": str(p),
            "label": dx,
            "raw_label": r["diagnostic"],
        })
    return pd.DataFrame(rows)


def sample(df: pd.DataFrame, per_class, seed=C.SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for label, g in df.groupby("label"):
        n = per_class.get(label) if isinstance(per_class, dict) else per_class
        if n is None or n >= len(g):
            parts.append(g)
        else:
            parts.append(g.iloc[rng.choice(len(g), n, replace=False)])
    return pd.concat(parts).sort_values("image_id").reset_index(drop=True)


def build_manifest(derm7pt_dir=C.DERM7PT_DIR, pad_dir=C.PAD_DIR) -> pd.DataFrame:
    d7 = sample(load_derm7pt(derm7pt_dir), C.DERM7PT_PER_CLASS)
    pad = sample(load_pad(pad_dir), C.PAD_PER_CLASS)
    return pd.concat([d7, pad], ignore_index=True)


def load_image(path, max_side=C.MAX_SIDE) -> np.ndarray:
    """RGB uint8 array, longest side at most max_side."""
    im = Image.open(path).convert("RGB")
    s = max_side / max(im.size)
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    return np.asarray(im)
