"""Image variants for the two counterfactual tests.

lesion_removed  the dilated lesion area is inpainted from the surrounding skin.
bg_tone         skin outside the lesion is made darker and browner.
bg_marker       purple skin-marker lines are drawn on the skin, as surgeons do before excision.
bg_ruler        a ruler with millimetre ticks is drawn along the bottom edge.

In the three bg_* variants every pixel inside the dilated lesion mask is copied unchanged
from the original; apply() checks this.
"""
import cv2
import numpy as np

from . import config as C
from .masks import dilate


def _seed(image_id: str) -> int:
    """Stable across runs (Python's hash() is not)."""
    import zlib
    return zlib.crc32(image_id.encode())


def remove_lesion(rgb, keep):
    h, w = keep.shape
    radius = max(5, int(0.02 * max(h, w)))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    out = cv2.inpaint(bgr, keep.astype(np.uint8) * 255, radius, cv2.INPAINT_TELEA)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    # inpainting a large area leaves a flat patch; add skin-like grain from the background
    rng = np.random.default_rng(_seed("noise"))
    bg = rgb[~keep]
    sd = bg.std(axis=0) * 0.35 if bg.size else np.zeros(3)
    noise = rng.normal(0, 1, out.shape) * sd
    out = np.where(keep[..., None], np.clip(out + noise, 0, 255), out).astype(np.uint8)
    return out


def darker_skin(rgb, keep):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 0.7
    lab[..., 1] += 5
    lab[..., 2] += 9
    mod = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return mod


def skin_marker(rgb, keep, image_id):
    rng = np.random.default_rng(_seed(image_id))
    h, w = keep.shape
    mod = rgb.copy()
    ys, xs = np.nonzero(keep)
    cy, cx = (ys.mean(), xs.mean()) if len(ys) else (h / 2, w / 2)
    ry = (ys.max() - ys.min()) / 2 if len(ys) else h / 4
    rx = (xs.max() - xs.min()) / 2 if len(xs) else w / 4
    color = (110, 40, 150)
    thick = max(3, int(0.008 * max(h, w)))
    # a dashed ellipse around the lesion, the usual excision outline
    axes = (int(rx * 1.35 + 0.06 * w), int(ry * 1.35 + 0.06 * h))
    start = rng.uniform(0, 30)
    for a in np.arange(start, start + 360, 24):
        cv2.ellipse(mod, (int(cx), int(cy)), axes, 0, a, a + 14, color, thick, cv2.LINE_AA)
    # and a short line with a dot, as markers often leave
    x0 = int(rng.uniform(0.08, 0.25) * w)
    y0 = int(rng.uniform(0.75, 0.9) * h)
    cv2.line(mod, (x0, y0), (x0 + int(0.12 * w), y0 - int(0.05 * h)), color, thick, cv2.LINE_AA)
    cv2.circle(mod, (x0, y0), thick * 2, color, -1, cv2.LINE_AA)
    return mod


def ruler(rgb, keep):
    h, w = keep.shape
    mod = rgb.copy()
    bar_h = max(18, int(0.07 * h))
    y1 = h - int(0.03 * h)
    y0 = y1 - bar_h
    x0, x1 = int(0.05 * w), int(0.95 * w)
    cv2.rectangle(mod, (x0, y0), (x1, y1), (238, 232, 205), -1)
    cv2.rectangle(mod, (x0, y0), (x1, y1), (60, 60, 60), 1)
    step = max(4, (x1 - x0) // 60)
    for i, x in enumerate(range(x0 + step, x1, step)):
        L = bar_h * (0.6 if i % 10 == 9 else 0.4 if i % 5 == 4 else 0.25)
        cv2.line(mod, (x, y0), (x, int(y0 + L)), (30, 30, 30), 1)
    return mod


def apply(rgb, mask, variant, image_id=""):
    """Return the requested variant of rgb. mask is the raw (undilated) lesion mask."""
    if variant == "original":
        return rgb
    keep = dilate(mask, C.MASK_DILATE_PX)
    if variant == "lesion_removed":
        return remove_lesion(rgb, keep)
    if variant == "bg_tone":
        # fade the change in over a band outside the protected area, so there is no hard ring
        dist = cv2.distanceTransform((~keep).astype(np.uint8), cv2.DIST_L2, 5)
        band = max(8.0, 0.04 * max(keep.shape))
        wgt = np.clip(dist / band, 0, 1)[..., None]
        mod = (rgb * (1 - wgt) + darker_skin(rgb, keep) * wgt).round().astype(np.uint8)
    elif variant == "bg_marker":
        mod = skin_marker(rgb, keep, image_id)
    elif variant == "bg_ruler":
        mod = ruler(rgb, keep)
    else:
        raise ValueError(variant)
    out = np.where(keep[..., None], rgb, mod)
    assert np.array_equal(out[keep], rgb[keep]), "lesion pixels changed"
    return out
