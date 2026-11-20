import json

import numpy as np
import pandas as pd
import pytest

from src import analyze, config as C, data, masks, perturb, prompts as P, run
from src.models import FakeVLM
from tests.synthetic import lesion_image, make


def test_bg_variants_leave_lesion_untouched():
    rng = np.random.default_rng(1)
    img = lesion_image(rng)
    m = masks.lesion_mask(img)
    keep = masks.dilate(m, C.MASK_DILATE_PX)
    for v in ("bg_tone", "bg_marker", "bg_ruler"):
        out = perturb.apply(img, m, v, "x")
        assert np.array_equal(out[keep], img[keep])
        assert not np.array_equal(out[~keep], img[~keep])


def test_lesion_removed_changes_lesion_only_area():
    rng = np.random.default_rng(2)
    img = lesion_image(rng)
    m = masks.lesion_mask(img)
    out = perturb.apply(img, m, "lesion_removed", "x")
    keep = masks.dilate(m, C.MASK_DILATE_PX)
    assert np.array_equal(out[~keep], img[~keep])
    # the removed area should now look like skin, not lesion
    assert out[m].mean() > img[m].mean() + 40


def test_mask_finds_synthetic_lesion():
    rng = np.random.default_rng(3)
    for _ in range(5):
        img = lesion_image(rng)
        m = masks.lesion_mask(img)
        assert masks.mask_ok(m)[0]
        dark = img.mean(2) < 130
        iou = (m & dark).sum() / (m | dark).sum()
        assert iou > 0.8


def test_reversed_order_maps_back():
    probs = np.zeros(9)
    probs[0] = 1.0  # letter A
    assert analyze.to_class_space(probs, "forward")[C.CLASSES.index("melanoma")] == 1
    assert analyze.to_class_space(probs, "reversed")[C.CLASSES.index("vascular lesion")] == 1
    probs = np.zeros(9)
    probs[8] = 1.0
    assert analyze.to_class_space(probs, "reversed")[-1] == 1


def test_ece():
    assert analyze.ece([0.9] * 10, [1] * 9 + [0]) == pytest.approx(0.0)
    assert analyze.ece([1.0] * 4, [0] * 4) == pytest.approx(1.0)
    assert analyze.ece([0.25, 0.75], [0, 1]) == pytest.approx(0.25)


def test_parse_describe():
    t = "Description: Brown lesion.\nFeatures: network\n**Diagnosis:** Melanocytic nevus\nConfidence: 85%"
    r = P.parse_describe(t)
    assert r["diagnosis"] == "nevus" and r["verbal_conf"] == 85
    assert P.parse_describe("Diagnosis: cannot be determined\nConfidence: 10")["diagnosis"] is None
    assert P.parse_describe("Diagnosis: basal cell carcinoma, not melanoma")["diagnosis"] == "basal cell carcinoma"


def test_derm7pt_labels():
    row = {"pigment_network": "atypical", "blue_whitish_veil": "absent", "vascular_structures": "dotted",
           "streaks": "regular", "pigmentation": "localized irregular", "dots_and_globules": "absent",
           "regression_structures": "blue areas"}
    c = data.derm7pt_criteria(row)
    assert c == {"atypical_network": 1, "blue_white_veil": 0, "atypical_vascular": 1, "irregular_streaks": 0,
                 "irregular_pigmentation": 1, "irregular_dots": 0, "regression": 1}
    assert data.derm7pt_diagnosis("melanoma metastasis") == "melanoma"
    assert data.derm7pt_diagnosis("reed or spitz nevus") == "nevus"
    assert data.derm7pt_diagnosis("lentigo") is None


def test_end_to_end(tmp_path, monkeypatch):
    d7, pad = make(tmp_path / "data")
    monkeypatch.setattr(C, "RESULTS_DIR", tmp_path / "results")
    work = tmp_path / "work"
    man = data.build_manifest(d7, pad)
    assert set(man.dataset) == {"derm7pt", "pad"}
    assert "lentigo" not in set(man.raw_label)
    run.prepare(work=work, manifest=man)
    run.infer(FakeVLM("fake"), work=work)
    run.infer(FakeVLM("fake"), work=work)  # second run must add nothing
    lines = (work / "raw" / "fake" / "mcq.jsonl").read_text().splitlines()
    assert len(lines) == len({json.loads(l)["key"] for l in lines})
    res = analyze.main(work=work, results=tmp_path / "results")
    r = res["fake"]
    assert r["lesion_removed"]["n"] > 0
    assert set(r["background"]) == {"bg_tone", "bg_marker", "bg_ruler"}
    assert len([k for k in r["criteria"] if not k.startswith("_")]) == 7
    assert (tmp_path / "results" / "results.md").read_text().startswith("# Pilot results")
    for f in ("reliability.png", "background_flips.png", "criteria_auroc.png", "variants.jpg"):
        assert (tmp_path / "results" / "figures" / f).exists()


def test_pad_only(tmp_path):
    d7, pad = make(tmp_path / "data")
    man = data.build_manifest(tmp_path / "nope", pad)
    assert set(man.dataset) == {"pad"}
