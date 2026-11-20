"""Turn the raw model answers into the four pilot results.

    python -m src.analyze

writes results/metrics.json, results/results.md and figures in results/figures/.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from . import config as C
from . import prompts as P

K = len(C.CLASSES)  # index K is "cannot be determined"


# ---------- loading ----------

def _read(path):
    if not Path(path).exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()])


def to_class_space(probs, order):
    """Map letter probabilities to [CLASSES..., cannot]."""
    out = np.zeros(K + 1)
    for p, cls in zip(probs, P.option_order(order)):
        out[K if cls is None else C.CLASSES.index(cls)] += p
    return out


def mcq_table(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per (image, variant); probabilities averaged over the two option orders."""
    if raw.empty:
        return raw
    raw = raw.copy()
    raw["p"] = [to_class_space(p, o) for p, o in zip(raw.probs, raw.order)]
    g = raw.groupby(["image_id", "variant"])
    t = g.p.apply(lambda s: np.mean(np.stack(s.values), 0)).reset_index()
    t["n_orders"] = g.size().values
    P9 = np.stack(t.p.values)
    P8 = P9[:, :K] / P9[:, :K].sum(1, keepdims=True)
    t["pred"] = [C.CLASSES[i] for i in P8.argmax(1)]  # forced choice among diagnoses
    t["conf"] = P8.max(1)
    t["p_cannot"] = P9[:, K]
    t["abstain"] = P9.argmax(1) == K
    t["p_mel"] = P8[:, C.CLASSES.index("melanoma")]
    t["p8"] = list(P8)
    return t


# ---------- statistics ----------

def boot(fn, n, seed=C.SEED, reps=C.N_BOOT):
    """Point estimate and 95% percentile interval of fn(idx) over bootstrap resamples."""
    rng = np.random.default_rng(seed)
    base = fn(np.arange(n))
    vals = [fn(rng.integers(0, n, n)) for _ in range(reps)]
    vals = [v for v in vals if np.isfinite(v)]
    lo, hi = (np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan))
    return {"est": float(base), "lo": float(lo), "hi": float(hi)}


def balanced_acc(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    cls = np.unique(y)
    return float(np.mean([(yhat[y == c] == c).mean() for c in cls])) if len(cls) else np.nan


def ece(conf, correct, bins=C.ECE_BINS):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    if len(conf) == 0:
        return np.nan
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def reliability(conf, correct, bins=C.ECE_BINS):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        rows.append((edges[b], edges[b + 1], int(m.sum()),
                     float(conf[m].mean()) if m.any() else np.nan,
                     float(correct[m].mean()) if m.any() else np.nan))
    return rows


def _safe_auc(y, s):
    y = np.asarray(y)
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


# ---------- the four tests ----------

def analyse_model(key, man, work):
    raw = Path(work) / "raw" / key
    t = mcq_table(_read(raw / "mcq.jsonl"))
    if t.empty:
        return None
    t = t.merge(man[["image_id", "dataset", "label", "mask_ok"]], on="image_id")
    t["correct"] = t.pred == t.label
    orig = t[t.variant == "original"].set_index("image_id")
    res = {"model": C.MODELS.get(key, key), "n_images": int(len(orig))}

    # baseline accuracy
    base = {}
    for ds, g in orig.groupby("dataset"):
        y, yh = g.label.values, g.pred.values
        base[ds] = {
            "n": int(len(g)),
            "accuracy": boot(lambda i: (yh[i] == y[i]).mean(), len(g)),
            "balanced_accuracy": boot(lambda i: balanced_acc(y[i], yh[i]), len(g)),
            "abstain_rate": float(g.abstain.mean()),
            "chance_balanced_accuracy": 1 / K,
            "most_common_prediction": g.pred.value_counts(normalize=True).head(3).round(3).to_dict(),
        }
    res["baseline"] = base

    # (a) lesion removed
    rem = t[t.variant == "lesion_removed"].set_index("image_id")
    ids = rem.index.intersection(orig.index)
    a = {"n": int(len(ids))}
    if len(ids):
        o, r = orig.loc[ids], rem.loc[ids]
        a["abstain_original"] = boot(lambda i: o.abstain.values[i].mean(), len(ids))
        a["abstain_removed"] = boot(lambda i: r.abstain.values[i].mean(), len(ids))
        a["confident_removed"] = boot(
            lambda i: ((~r.abstain.values[i]) & (r.conf.values[i] >= 0.5)).mean(), len(ids))
        a["same_diagnosis"] = boot(lambda i: (o.pred.values[i] == r.pred.values[i]).mean(), len(ids))
        a["conf_original"] = float(o.conf.mean())
        a["conf_removed"] = float(r.conf.mean())
        a["accuracy_original"] = float((o.pred == o.label).mean())
        a["accuracy_removed"] = float((r.pred == r.label).mean())
    res["lesion_removed"] = a

    # (b) background changes
    b = {}
    for v in ("bg_tone", "bg_marker", "bg_ruler"):
        vv = t[t.variant == v].set_index("image_id")
        ids = vv.index.intersection(orig.index)
        if not len(ids):
            continue
        o, x = orig.loc[ids], vv.loc[ids]
        po, px = np.stack(o.p8.values), np.stack(x.p8.values)
        tv = 0.5 * np.abs(po - px).sum(1)
        dmel = x.p_mel.values - o.p_mel.values
        b[v] = {
            "n": int(len(ids)),
            "flip_rate": boot(lambda i: (o.pred.values[i] != x.pred.values[i]).mean(), len(ids)),
            "mean_tv_distance": float(tv.mean()),
            "delta_p_melanoma": boot(lambda i: dmel[i].mean(), len(ids)),
            "accuracy_original": float((o.pred == o.label).mean()),
            "accuracy_variant": float((x.pred == x.label).mean()),
        }
    res["background"] = b

    # (c) checklist criteria against Derm7pt labels
    cr = _read(raw / "criteria.jsonl")
    c = {}
    if not cr.empty:
        lab = man.set_index("image_id")
        piv = cr.pivot_table(index="image_id", columns="criterion", values="p_yes")
        for name in C.CRITERIA:
            if name not in piv:
                continue
            s = piv[name].dropna()
            y = lab.loc[s.index, name].astype(int).values
            pv = s.values
            c[name] = {
                "n": int(len(s)), "prevalence": float(y.mean()),
                "auroc": boot(lambda i: _safe_auc(y[i], pv[i]), len(s)),
                "said_yes_rate": float((pv > 0.5).mean()),
            }
        # does the model's own checklist go with its own melanoma probability?
        ids = piv.dropna().index.intersection(orig.index)
        if len(ids) > 5:
            score = piv.loc[ids, list(C.CRITERIA)].sum(axis=1).values
            truth = lab.loc[ids, list(C.CRITERIA)].sum(axis=1).values
            pm = orig.loc[ids, "p_mel"].values
            ismel = (lab.loc[ids, "label"] == "melanoma").astype(int).values
            c["_consistency"] = {
                "spearman_model_checklist_vs_model_p_melanoma": float(spearmanr(score, pm).statistic),
                "spearman_expert_checklist_vs_model_p_melanoma": float(spearmanr(truth, pm).statistic),
                "auroc_melanoma_from_model_checklist": float(_safe_auc(ismel, score)),
                "auroc_melanoma_from_model_p_melanoma": float(_safe_auc(ismel, pm)),
            }
    res["criteria"] = c

    # (d) calibration
    d = {}
    for ds, g in orig.groupby("dataset"):
        conf, cor = g.conf.values, g.correct.values.astype(float)
        d[ds] = {
            "token_ece": boot(lambda i: ece(conf[i], cor[i]), len(g)),
            "token_mean_conf": float(conf.mean()), "accuracy": float(cor.mean()),
            "token_reliability": reliability(conf, cor),
        }
    de = _read(raw / "describe.jsonl")
    if not de.empty:
        de = de.merge(man[["image_id", "dataset", "label"]], on="image_id")
        for ds, g in de.groupby("dataset"):
            ok = g.diagnosis.notna() & g.verbal_conf.notna()
            gg = g[ok]
            conf = gg.verbal_conf.values / 100.0
            cor = (gg.diagnosis == gg.label).values.astype(float)
            d.setdefault(ds, {})
            d[ds].update({
                "verbal_n_parsed": int(ok.sum()), "verbal_n_total": int(len(g)),
                "verbal_ece": boot(lambda i: ece(conf[i], cor[i]), len(gg)) if len(gg) else None,
                "verbal_mean_conf": float(conf.mean()) if len(gg) else None,
                "verbal_accuracy": float(cor.mean()) if len(gg) else None,
                "verbal_reliability": reliability(conf, cor) if len(gg) else None,
            })
    res["calibration"] = d
    return res


# ---------- report ----------

def _ci(x, pct=True):
    if not isinstance(x, dict):
        return "–"
    f = (lambda v: f"{100 * v:.1f}") if pct else (lambda v: f"{v:.3f}")
    return f"{f(x['est'])} ({f(x['lo'])}–{f(x['hi'])})"


def report(all_res) -> str:
    L = ["# Pilot results", "",
         "All numbers come from `python -m src.analyze` on the raw answers in `work/raw/`. "
         "Intervals are 95% bootstrap intervals over images. Percentages unless noted.", ""]
    for key, r in all_res.items():
        L += [f"## {r['model']}", ""]
        L += ["### Baseline diagnosis (original images, forced choice among 8 diagnoses)", "",
              "| Dataset | n | Accuracy | Balanced accuracy | Chose \"cannot be determined\" |",
              "|---|---|---|---|---|"]
        for ds, b in r["baseline"].items():
            L.append(f"| {ds} | {b['n']} | {_ci(b['accuracy'])} | {_ci(b['balanced_accuracy'])} | "
                     f"{100 * b['abstain_rate']:.1f} |")
        L += ["", f"Chance balanced accuracy is {100 / K:.1f}.", ""]

        a = r["lesion_removed"]
        if a.get("n"):
            L += ["### (a) Lesion removed", "",
                  f"{a['n']} images with a usable mask.", "",
                  "| | Original | Lesion removed |", "|---|---|---|",
                  f"| Chose \"cannot be determined\" | {_ci(a['abstain_original'])} | {_ci(a['abstain_removed'])} |",
                  f"| Mean confidence (forced choice) | {100 * a['conf_original']:.1f} | {100 * a['conf_removed']:.1f} |",
                  f"| Accuracy | {100 * a['accuracy_original']:.1f} | {100 * a['accuracy_removed']:.1f} |", "",
                  f"Still a confident diagnosis (not abstaining, confidence ≥ 50%) with the lesion removed: "
                  f"{_ci(a['confident_removed'])}. Same diagnosis as on the original: {_ci(a['same_diagnosis'])}.", ""]

        if r["background"]:
            L += ["### (b) Background changed, lesion pixels identical", "",
                  "| Change | n | Diagnosis flipped | Mean change in P(melanoma), points | Accuracy before → after |",
                  "|---|---|---|---|---|"]
            for v, b in r["background"].items():
                dm = b["delta_p_melanoma"]
                L.append(f"| {v} | {b['n']} | {_ci(b['flip_rate'])} | "
                         f"{100 * dm['est']:+.1f} ({100 * dm['lo']:+.1f} to {100 * dm['hi']:+.1f}) | "
                         f"{100 * b['accuracy_original']:.1f} → {100 * b['accuracy_variant']:.1f} |")
            L.append("")

        c = {k: v for k, v in r["criteria"].items() if not k.startswith("_")}
        if c:
            L += ["### (c) 7-point checklist criteria against Derm7pt expert labels", "",
                  "| Criterion | n | Prevalence | AUROC | Model says yes |", "|---|---|---|---|---|"]
            for name, x in c.items():
                L.append(f"| {name} | {x['n']} | {100 * x['prevalence']:.1f} | {_ci(x['auroc'], pct=False)} | "
                         f"{100 * x['said_yes_rate']:.1f} |")
            cons = r["criteria"].get("_consistency")
            if cons:
                L += ["", "Consistency between the model's own checklist and its own diagnosis "
                      "(Spearman correlation of summed criterion probabilities with P(melanoma)): "
                      f"{cons['spearman_model_checklist_vs_model_p_melanoma']:.2f}. "
                      f"Expert checklist with model P(melanoma): {cons['spearman_expert_checklist_vs_model_p_melanoma']:.2f}."]
            L.append("")

        L += ["### (d) Calibration (original images)", "",
              "| Dataset | Accuracy | Mean token confidence | Token ECE | Verbal ECE (parsed answers) |",
              "|---|---|---|---|---|"]
        for ds, x in r["calibration"].items():
            ve = x.get("verbal_ece")
            vtxt = f"{_ci(ve, pct=False)} ({x['verbal_n_parsed']}/{x['verbal_n_total']})" if ve else "–"
            L.append(f"| {ds} | {100 * x.get('accuracy', np.nan):.1f} | "
                     f"{100 * x.get('token_mean_conf', np.nan):.1f} | {_ci(x.get('token_ece'), pct=False)} | {vtxt} |")
        L.append("")
    return "\n".join(L)


def figures(all_res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    names = {k: r["model"].split("/")[-1] for k, r in all_res.items()}
    colors = ["#1F6F9E", "#C0273D", "#4A7A3A"]

    # reliability diagrams
    dss = sorted({ds for r in all_res.values() for ds in r["calibration"]})
    if dss:
        fig, axes = plt.subplots(1, len(dss), figsize=(4.2 * len(dss), 4), squeeze=False)
        for ax, ds in zip(axes[0], dss):
            ax.plot([0, 1], [0, 1], color="#999", lw=1, ls="--")
            for (k, r), col in zip(all_res.items(), colors):
                rel = r["calibration"].get(ds, {}).get("token_reliability")
                if not rel:
                    continue
                xs = [x[3] for x in rel if x[2] > 0]
                ys = [x[4] for x in rel if x[2] > 0]
                ax.plot(xs, ys, marker="o", color=col, label=names[k])
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Confidence", ylabel="Accuracy", title=ds)
            ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "reliability.png", dpi=200)
        plt.close(fig)

    # flip rates under background changes
    vs = ["bg_tone", "bg_marker", "bg_ruler"]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    w = 0.8 / max(1, len(all_res))
    for j, ((k, r), col) in enumerate(zip(all_res.items(), colors)):
        est = [r["background"].get(v, {}).get("flip_rate", {}).get("est", np.nan) for v in vs]
        lo = [r["background"].get(v, {}).get("flip_rate", {}).get("lo", np.nan) for v in vs]
        hi = [r["background"].get(v, {}).get("flip_rate", {}).get("hi", np.nan) for v in vs]
        x = np.arange(len(vs)) + j * w
        ax.bar(x, np.array(est) * 100, w, color=col, label=names[k],
               yerr=[(np.array(est) - lo) * 100, (np.array(hi) - est) * 100], capsize=3)
    ax.set_xticks(np.arange(len(vs)) + w * (len(all_res) - 1) / 2,
                  ["Darker skin", "Skin marker", "Ruler"])
    ax.set_ylabel("Diagnosis changed (%)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "background_flips.png", dpi=200)
    plt.close(fig)

    # criterion AUROC
    crits = list(C.CRITERIA)
    if any(r["criteria"] for r in all_res.values()):
        fig, ax = plt.subplots(figsize=(7, 3.6))
        for j, ((k, r), col) in enumerate(zip(all_res.items(), colors)):
            est = [r["criteria"].get(c, {}).get("auroc", {}).get("est", np.nan) for c in crits]
            ax.bar(np.arange(len(crits)) + j * w, est, w, color=col, label=names[k])
        ax.axhline(0.5, color="#999", lw=1, ls="--")
        ax.set_xticks(np.arange(len(crits)) + w * (len(all_res) - 1) / 2,
                      [c.replace("_", "\n") for c in crits], fontsize=8)
        ax.set_ylim(0.3, 1)
        ax.set_ylabel("AUROC against expert label")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / "criteria_auroc.png", dpi=200)
        plt.close(fig)


def example_grid(work, out, dataset="pad", n=3):
    """A few PAD-UFES-20 images (CC BY 4.0) in every variant, for the README."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    man = pd.read_csv(Path(work) / "manifest.csv")
    ex = man[(man.dataset == dataset) & man.mask_ok].head(n)
    if ex.empty:
        return
    titles = {"original": "Original", "lesion_removed": "Lesion removed", "bg_tone": "Darker skin",
              "bg_marker": "Skin marker", "bg_ruler": "Ruler"}
    fig, axes = plt.subplots(len(ex), len(C.VARIANTS), figsize=(2.2 * len(C.VARIANTS), 2.2 * len(ex)),
                             squeeze=False)
    for i, r in enumerate(ex.itertuples()):
        for j, v in enumerate(C.VARIANTS):
            ax = axes[i][j]
            ax.imshow(Image.open(Path(work) / "images" / v / f"{r.image_id}.png"))
            ax.axis("off")
            if i == 0:
                ax.set_title(titles[v], fontsize=10)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "variants.jpg", dpi=130)
    plt.close(fig)


def main(work=C.WORK_DIR, results=C.RESULTS_DIR):
    work, results = Path(work), Path(results)
    man = pd.read_csv(work / "manifest.csv")
    all_res = {}
    for key in sorted(p.name for p in (work / "raw").iterdir() if p.is_dir()):
        r = analyse_model(key, man, work)
        if r:
            all_res[key] = r
    results.mkdir(parents=True, exist_ok=True)
    (results / "metrics.json").write_text(json.dumps(all_res, indent=1, default=float))
    (results / "results.md").write_text(report(all_res))
    figures(all_res, results / "figures")
    example_grid(work, results / "figures")
    summary = man.groupby(["dataset", "label"]).agg(n=("image_id", "size"), mask_ok=("mask_ok", "sum"))
    summary.to_csv(results / "sample.csv")
    print(report(all_res))
    return all_res


if __name__ == "__main__":
    main()
