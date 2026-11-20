"""Compare loading options and batching for one model on a few images, and time them.

    from src.check import compare; compare("medgemma")
"""
import gc
import time

import numpy as np
from PIL import Image

from . import config as C
from . import data
from . import prompts as P


def _free(model):
    import torch
    del model
    gc.collect()
    torch.cuda.empty_cache()


def compare(key, n_images=3, batch=4, pad_dir=None):
    from .models import VLM
    pad = data.load_pad(pad_dir or C.PAD_DIR).head(n_images)
    images = [Image.fromarray(data.load_image(p)) for p in pad.path]
    texts = [P.mcq("clinical", o) for o in ("forward", "reversed")]

    results = {}
    for name, kw in (("4-bit", {"four_bit": True}), ("float16", {"half": True})):
        try:
            model = VLM(key, **kw)
        except Exception as e:  # noqa: BLE001
            print(name, "could not load:", e)
            continue
        try:
            model.option_probs(images[0], texts[0])  # warm-up
            t = time.time()
            single = np.stack([model.option_probs(im, tx) for im in images for tx in texts])
            t_single = (time.time() - t) / len(single)
            jobs = [(im, tx) for im in images for tx in texts]
            t = time.time()
            batched = np.concatenate([model.option_probs_batch([j[0] for j in jobs[i:i + batch]],
                                                               [j[1] for j in jobs[i:i + batch]])
                                      for i in range(0, len(jobs), batch)])
            t_batch = (time.time() - t) / len(batched)
            print(f"{name}: {t_single:.2f}s per question one at a time, {t_batch:.2f}s in batches of {batch}; "
                  f"batch vs single max difference {np.abs(single - batched).max():.4f}")
            results[name] = single
            t = time.time()
            out = model.generate_batch(images[:2], [P.describe("clinical")] * 2)
            print(f"{name}: generating 2 descriptions took {time.time() - t:.1f}s")
            print("   ", out[0][:200].replace("\n", " | "))
        except Exception as e:  # noqa: BLE001
            print(name, "failed:", str(e)[:300])
        _free(model)
    if len(results) == 2:
        a, b = results["4-bit"], results["float16"]
        print(f"4-bit vs float16: max difference {np.abs(a - b).max():.3f}, "
              f"same top answer {np.mean(a.argmax(1) == b.argmax(1)):.0%}")
