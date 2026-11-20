"""Two steps:

    python -m src.run prepare           sample images, make masks and variants, write a QC sheet
    python -m src.run infer --model qwen
    python -m src.run infer --model medgemma

Inference writes one JSON line per question to work/raw/<model>/ and skips anything
already there, so a run that stops (Kaggle sessions end after 12 hours) can be restarted.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
try:
    from tqdm import tqdm
except ImportError:  # progress bars are optional
    def tqdm(it, **_):
        return it

from . import config as C
from . import data, masks, perturb
from . import prompts as P


def prepare(derm7pt_dir=C.DERM7PT_DIR, pad_dir=C.PAD_DIR, work=C.WORK_DIR, manifest=None):
    work = Path(work)
    for v in C.VARIANTS:
        (work / "images" / v).mkdir(parents=True, exist_ok=True)
    (work / "masks").mkdir(parents=True, exist_ok=True)
    df = manifest if manifest is not None else data.build_manifest(derm7pt_dir, pad_dir)

    bad_file = C.RESULTS_DIR / "bad_masks.txt"  # ids rejected by eye after looking at the QC sheet
    manual_bad = set(bad_file.read_text().split()) if bad_file.exists() else set()

    oks, reasons, fracs, qc = [], [], [], []
    for r in tqdm(df.itertuples(), total=len(df), desc="prepare"):
        rgb = data.load_image(r.path)
        m = masks.lesion_mask(rgb)
        ok, why = masks.mask_ok(m)
        if r.image_id in manual_bad:
            ok, why = False, "rejected on QC sheet"
        oks.append(ok)
        reasons.append(why)
        fracs.append(float(m.mean()))
        Image.fromarray(m.astype(np.uint8) * 255).save(work / "masks" / f"{r.image_id}.png")
        for v in C.VARIANTS:
            if v != "original" and not ok:
                continue
            out = perturb.apply(rgb, m, v, r.image_id)
            Image.fromarray(out).save(work / "images" / v / f"{r.image_id}.png")
        qc.append((r.image_id, rgb, m, ok))

    df = df.copy()
    df["mask_ok"], df["mask_note"], df["mask_frac"] = oks, reasons, fracs
    df.to_csv(work / "manifest.csv", index=False)
    for i in range(0, len(qc), 64):
        masks.qc_sheet(qc[i:i + 64], work / f"mask_qc_{i // 64 + 1:02d}.jpg")
    print(df.groupby(["dataset", "label"]).size().to_string())
    print(f"masks passing checks: {sum(oks)} of {len(oks)}")
    return df


def _done(path: Path):
    keys = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                keys.add(json.loads(line)["key"])
    return keys


def _write(f, rec):
    f.write(json.dumps(rec) + "\n")
    f.flush()


def infer(model, work=C.WORK_DIR, limit=None, variants=C.VARIANTS):
    work = Path(work)
    df = pd.read_csv(work / "manifest.csv")
    if limit:
        df = df.groupby("dataset").head(limit)
    out_dir = work / "raw" / model.key
    out_dir.mkdir(parents=True, exist_ok=True)

    def img(v, iid):
        p = work / "images" / v / f"{iid}.png"
        return Image.open(p).convert("RGB") if p.exists() else None

    # 1. multiple-choice diagnosis, every variant, both option orders
    path = out_dir / "mcq.jsonl"
    done = _done(path)
    with path.open("a") as f:
        for r in tqdm(list(df.itertuples()), desc=f"{model.key} diagnosis"):
            for v in variants:
                im = None
                for order in ("forward", "reversed"):
                    key = f"{r.image_id}|{v}|{order}"
                    if key in done:
                        continue
                    im = im or img(v, r.image_id)
                    if im is None:
                        break
                    p = model.option_probs(im, P.mcq(r.modality, order))
                    _write(f, {"key": key, "image_id": r.image_id, "variant": v, "order": order,
                               "probs": [round(float(x), 6) for x in p]})

    # 2. checklist criteria, Derm7pt originals only (they have expert labels)
    path = out_dir / "criteria.jsonl"
    done = _done(path)
    with path.open("a") as f:
        for r in tqdm(list(df[df.dataset == "derm7pt"].itertuples()), desc=f"{model.key} criteria"):
            im = None
            for name in C.CRITERIA:
                key = f"{r.image_id}|{name}"
                if key in done:
                    continue
                im = im or img("original", r.image_id)
                _write(f, {"key": key, "image_id": r.image_id, "criterion": name,
                           "p_yes": round(model.yes_prob(im, P.criterion(name)), 6)})

    # 3. free-text description with stated features and confidence, originals only
    path = out_dir / "describe.jsonl"
    done = _done(path)
    with path.open("a") as f:
        for r in tqdm(list(df.itertuples()), desc=f"{model.key} describe"):
            if r.image_id in done:
                continue
            text = model.generate(img("original", r.image_id), P.describe(r.modality))
            _write(f, {"key": r.image_id, "image_id": r.image_id, "text": text, **P.parse_describe(text)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prepare", "infer"])
    ap.add_argument("--model", choices=list(C.MODELS))
    ap.add_argument("--four-bit", action="store_true", help="force 4-bit loading")
    ap.add_argument("--limit", type=int, default=None, help="images per dataset, for a quick test")
    a = ap.parse_args()
    if a.step == "prepare":
        prepare()
    else:
        from .models import VLM
        infer(VLM(a.model, four_bit=True if a.four_bit else None), limit=a.limit)


if __name__ == "__main__":
    main()
