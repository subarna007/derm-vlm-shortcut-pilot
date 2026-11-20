"""Check that batched answers match one-at-a-time answers, and time both.

    from src.check import compare; compare("medgemma")
"""
import time

import numpy as np
from PIL import Image

from . import config as C
from . import data
from . import prompts as P


def compare(key, n_images=2, pad_dir=None):
    from .models import VLM
    pad = data.load_pad(pad_dir or C.PAD_DIR).head(n_images)
    images = [Image.fromarray(data.load_image(p)) for p in pad.path]
    model = VLM(key)
    print(key, "4-bit" if model.four_bit else "", model.dtype)
    texts = [P.mcq("clinical", o) for o in ("forward", "reversed")] * 5
    for im in images:
        model.option_probs(im, texts[0])  # warm-up
        t = time.time()
        single = np.stack([model.option_probs(im, tx) for tx in texts])
        t1 = time.time()
        batch = model.option_probs_batch([im] * len(texts), texts)
        t2 = time.time()
        print(f"one at a time {t1 - t:.1f}s, batch of {len(texts)} {t2 - t1:.1f}s, "
              f"largest difference {np.abs(single - batch).max():.4f}, "
              f"same top answer {np.mean(single.argmax(1) == batch.argmax(1)):.0%}")
    t = time.time()
    out = model.generate_batch(images, [P.describe("clinical")] * len(images))
    print(f"generate batch of {len(images)}: {time.time() - t:.1f}s")
    print(out[0][:300])
    return model
