"""Rough automatic lesion masks.

Neither dataset ships segmentation masks, so the lesion is found with Otsu thresholding
on whichever channel separates lesion from skin best (darkness or redness), keeping the
connected region closest to the image centre. Masks are only used to decide which pixels
to protect or remove, so they are dilated by a safety margin, and every mask is checked
by eye on a contact sheet before the run (see qc_sheet).
"""
import cv2
import numpy as np


def _otsu(channel: np.ndarray):
    t, m = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg, bg = channel[m > 0], channel[m == 0]
    if fg.size == 0 or bg.size == 0:
        return m, 0.0
    sep = (fg.mean() - bg.mean()) ** 2 / (channel.var() + 1e-6)  # share of variance between classes
    return m, sep


def _dark_corner_mask(h, w, frac=0.12):
    """Dermoscopy images often have black vignette corners; ignore them."""
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot((yy - h / 2) / (h / 2), (xx - w / 2) / (w / 2))
    return r > (1.41 - frac)


def lesion_mask(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    blur = cv2.GaussianBlur(rgb, (0, 0), sigmaX=max(h, w) / 200)
    lab = cv2.cvtColor(blur, cv2.COLOR_RGB2LAB)
    dark = 255 - lab[..., 0]
    red = lab[..., 1]

    candidates = [_otsu(dark), _otsu(red)]
    m, _ = max(candidates, key=lambda c: c[1])
    m = m > 0
    # the lesion is the class that touches the image border less
    if _border_share(m) > _border_share(~m):
        m = ~m
    corners = _dark_corner_mask(h, w) & (lab[..., 0] < 40)
    m &= ~corners

    k = max(3, (max(h, w) // 100) | 1)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3 * k, 3 * k), np.uint8))

    n, lab_cc, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.zeros((h, w), bool)
    # score = area, penalised by distance from the centre
    best, best_score = 0, -1.0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        cx, cy = cents[i]
        d = np.hypot((cx - w / 2) / w, (cy - h / 2) / h)
        score = area * (1 - min(d, 0.7))
        if score > best_score:
            best, best_score = i, score
    out = (lab_cc == best).astype(np.uint8)

    # fill holes: flood the background from outside a one-pixel frame
    ff = np.pad(out, 1)
    cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 1)
    out = out | (1 - ff[1:-1, 1:-1])
    return out.astype(bool)


def _border_share(m):
    return np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]]).mean()


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def mask_ok(mask: np.ndarray, lo=0.02, hi=0.75) -> tuple:
    """Automatic sanity check. Returns (ok, reason)."""
    frac = mask.mean()
    if frac < lo:
        return False, f"too small ({frac:.1%})"
    if frac > hi:
        return False, f"too large ({frac:.1%})"
    h, w = mask.shape
    border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    if border.mean() > 0.5:
        return False, "lesion fills the border"
    return True, ""


def overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    edge = dilate(mask, 2) & ~mask
    out[edge] = (0, 255, 255)
    return out


def qc_sheet(items, path, cols=8, tile=180):
    """items: list of (image_id, rgb, mask, ok). Writes one JPEG contact sheet."""
    rows = (len(items) + cols - 1) // cols
    sheet = np.full((rows * (tile + 18), cols * tile, 3), 255, np.uint8)
    for i, (iid, rgb, m, ok) in enumerate(items):
        t = cv2.resize(overlay(rgb, m), (tile, tile))
        r, c = divmod(i, cols)
        y, x = r * (tile + 18), c * tile
        sheet[y:y + tile, x:x + tile] = t
        cv2.putText(sheet, iid[-12:], (x + 3, y + tile + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (0, 0, 0) if ok else (200, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
