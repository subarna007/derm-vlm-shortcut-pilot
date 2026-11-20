"""Prompts and parsing of the models' free-text answers."""
import re

from . import config as C

SYSTEM = "You are an experienced dermatologist."


def option_order(order: str):
    """Class order for the answer options. 'reversed' is used to cancel letter bias."""
    classes = list(C.CLASSES)
    if order == "reversed":
        classes = classes[::-1]
    return classes + [None]  # None stands for "cannot be determined"


def mcq(modality: str, order: str) -> str:
    lines = [f"This is a {modality} image of a skin lesion. What is the most likely diagnosis?", ""]
    for letter, cls in zip(C.LETTERS, option_order(order)):
        lines.append(f"{letter}. {C.OPTION_TEXT[cls] if cls else C.CANNOT_TELL}")
    lines += ["", "Answer with the letter of one option only."]
    return "\n".join(lines)


def criterion(name: str) -> str:
    return (f"This is a dermoscopic image of a skin lesion. Does the lesion show {C.CRITERIA[name]}? "
            "Answer yes or no.")


def describe(modality: str) -> str:
    opts = ", ".join(C.OPTION_TEXT[c].split(" (")[0].lower() for c in C.CLASSES)
    return (
        f"This is a {modality} image of a skin lesion.\n"
        "Reply in exactly this format and nothing else:\n"
        "Description: <two sentences on what you see>\n"
        "Features: <the visual features that support your diagnosis, comma separated>\n"
        f"Diagnosis: <one of: {opts}, or cannot be determined>\n"
        "Confidence: <a number from 0 to 100>"
    )


_ALIASES = {
    "melanoma": ["melanoma"],
    "nevus": ["nevus", "naevus", "mole"],
    "basal cell carcinoma": ["basal cell", "bcc"],
    "squamous cell carcinoma": ["squamous cell", "scc"],
    "actinic keratosis": ["actinic"],
    "seborrheic keratosis": ["seborrh"],
    "dermatofibroma": ["dermatofibroma"],
    "vascular lesion": ["vascular", "angioma", "hemangioma"],
}


def parse_describe(text: str) -> dict:
    def field(name):
        m = re.search(rf"^\s*\**{name}\**\s*:\s*(.+)$", text, re.I | re.M)
        return m.group(1).strip() if m else ""

    dx_text = field("Diagnosis").lower()
    dx = None
    if dx_text and "cannot" not in dx_text and "not be determined" not in dx_text:
        # first class mentioned wins
        pos = {c: min((dx_text.find(a) for a in al if a in dx_text), default=-1)
               for c, al in _ALIASES.items()}
        pos = {c: p for c, p in pos.items() if p >= 0}
        if pos:
            dx = min(pos, key=pos.get)
    conf = None
    m = re.search(r"(\d{1,3}(?:\.\d+)?)", field("Confidence"))
    if m:
        conf = min(100.0, float(m.group(1)))
    return {
        "description": field("Description"),
        "features": field("Features"),
        "diagnosis_text": dx_text,
        "diagnosis": dx,
        "verbal_conf": conf,
    }
