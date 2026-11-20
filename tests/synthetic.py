"""Small fake copies of the two datasets, in their real folder layouts, for tests."""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

D7_DX = ["melanoma (less than 0.76 mm)", "clark nevus", "basal cell carcinoma",
         "seborrheic keratosis", "dermatofibroma", "vascular lesion", "lentigo"]
PAD_DX = ["MEL", "NEV", "BCC", "SCC", "ACK", "SEK"]


def lesion_image(rng, size=(300, 400), red=False):
    h, w = size
    skin = np.array([205, 160, 140]) + rng.normal(0, 8, 3)
    img = np.clip(skin + rng.normal(0, 6, (h, w, 3)), 0, 255).astype(np.uint8)
    c = (int(w / 2 + rng.normal(0, w / 12)), int(h / 2 + rng.normal(0, h / 12)))
    axes = (int(rng.uniform(0.12, 0.25) * w), int(rng.uniform(0.12, 0.25) * h))
    col = (190, 70, 70) if red else tuple(int(v) for v in rng.uniform(40, 110, 3))
    cv2.ellipse(img, c, axes, rng.uniform(0, 180), 0, 360, col, -1)
    return img


def make(root: Path, n_d7=21, n_pad=18, seed=0):
    rng = np.random.default_rng(seed)
    d7 = root / "derm7pt" / "release_v0"
    (d7 / "meta").mkdir(parents=True)
    (d7 / "images" / "AAA").mkdir(parents=True)
    rows = []
    for i in range(n_d7):
        name = f"AAA/case{i:03d}.jpg"
        cv2.imwrite(str(d7 / "images" / name), lesion_image(rng)[..., ::-1])
        rows.append({
            "case_num": i + 1, "diagnosis": D7_DX[i % len(D7_DX)], "seven_point_score": 0,
            "pigment_network": rng.choice(["absent", "typical", "atypical"]),
            "streaks": rng.choice(["absent", "regular", "irregular"]),
            "pigmentation": rng.choice(["absent", "diffuse regular", "localized irregular"]),
            "regression_structures": rng.choice(["absent", "blue areas"]),
            "dots_and_globules": rng.choice(["absent", "regular", "irregular"]),
            "blue_whitish_veil": rng.choice(["absent", "present"]),
            "vascular_structures": rng.choice(["absent", "arborizing", "dotted", "linear irregular"]),
            "clinic": name, "derm": name,
        })
    pd.DataFrame(rows).to_csv(d7 / "meta" / "meta.csv", index=False)

    pad = root / "pad-ufes-20"
    (pad / "images" / "imgs_part_1").mkdir(parents=True)
    rows = []
    for i in range(n_pad):
        dx = PAD_DX[i % len(PAD_DX)]
        img_id = f"PAT_{i}_{i}_{100 + i}.png"
        cv2.imwrite(str(pad / "images" / "imgs_part_1" / img_id), lesion_image(rng, red=dx in {"BCC", "ACK"})[..., ::-1])
        rows.append({"patient_id": f"PAT_{i}", "lesion_id": i, "diagnostic": dx, "img_id": img_id})
    pd.DataFrame(rows).to_csv(pad / "metadata.csv", index=False)
    return root / "derm7pt", pad
