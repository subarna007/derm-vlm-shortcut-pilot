# Do dermatology vision-language models look at the lesion?

A small pilot study of two open vision-language models (VLMs) on skin lesion images. Medical VLMs
now write fluent descriptions and diagnoses for skin photos, but fluent text does not show that the
answer came from the lesion. A model could lean on the skin around it, on things in the photo such as
rulers or surgical ink, or simply be confident whatever it sees. This pilot checks that cheaply,
before a larger study.

**Status: code ready, pilot not yet run.** Results will be added to `results/` once the runs are done.

## Models

| Model | Type | How it is run |
|---|---|---|
| [MedGemma 1.5 4B](https://huggingface.co/google/medgemma-1.5-4b-it) | medical VLM (Google) | 4-bit on a T4; bfloat16 on newer GPUs |
| [Qwen2.5-VL 3B Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | general VLM | float16 |

Both are run zero-shot with the same prompts. Neither is fine-tuned.

## Data

About 470 images: 256 dermoscopic images from **Derm7pt** (60 melanoma, 60 nevus, and all basal cell
carcinoma, seborrheic keratosis, dermatofibroma and vascular lesion cases) and 210 smartphone photos from
**PAD-UFES-20** (35 of each of its six diagnoses, one image per lesion). Derm7pt also has expert labels
for the seven criteria of the 7-point checklist, which test (c) uses.

The images are not in this repository. Derm7pt is available after registration at
[derm.cs.sfu.ca](https://derm.cs.sfu.ca), and PAD-UFES-20 on
[Mendeley Data](https://data.mendeley.com/datasets/zr7vgbcyr2/1) (CC BY 4.0). Example figures use
PAD-UFES-20 images only.

## What is measured

Every image is shown to each model as a multiple-choice question with eight diagnoses and a ninth
option, "cannot be determined from this image". The answer probabilities are read from the model's
first output token, so each question gives a full probability distribution rather than one sampled
answer. Every question is asked twice, with the options in forward and in reversed order, and the two
are averaged to cancel any preference for particular letters.

| Test | What is done | What would be a warning sign |
|---|---|---|
| (a) Lesion removed | The lesion is found with a rough automatic mask and painted over with surrounding skin. | The model still gives a confident diagnosis, or the same one, with no lesion in the image. |
| (b) Background changed | Only the skin around the lesion is changed: darker skin tone, purple skin-marker lines, or a ruler. The lesion pixels are identical to the original (the code checks this). | The diagnosis changes, or the probability of melanoma moves, when the lesion did not change. |
| (c) Stated features | For each Derm7pt image the model is asked yes/no for each of the seven checklist criteria; the answers are compared with the expert labels (AUROC). It also writes a free-text description. | Feature answers no better than chance, or not related to the model's own diagnosis. |
| (d) Calibration | Confidence is compared with how often the model is right, both for the token probabilities and for the confidence the model states in words (expected calibration error, ECE). | High confidence with low accuracy. |

All results are given with 95% bootstrap intervals, because 470 images are few.

## Running it

Everything runs on a free Kaggle GPU. `notebooks/pilot.ipynb` does the whole run; set `MODEL` to
`"qwen"` and then `"medgemma"`. It needs a Hugging Face token (saved as the Kaggle secret `HF_TOKEN`) with
the MedGemma terms of use accepted, and the two datasets attached to the notebook.

From a terminal:

```bash
pip install -r requirements.txt
export DERM7PT_DIR=/path/to/derm7pt PAD_DIR=/path/to/pad-ufes-20
python -m src.run prepare                 # sample, masks, edited images, mask contact sheets
python -m src.run infer --model qwen
python -m src.run infer --model medgemma
python -m src.analyze                     # results/results.md, metrics.json, figures
```

Inference appends one line per answer to `work/raw/<model>/` and skips answers already there, so an
interrupted run continues where it stopped.

The tests build small fake versions of both datasets and run the whole pipeline with a stand-in model,
so they need no GPU or downloads:

```bash
python -m pytest -q
```

## Code

```text
src/config.py     sample sizes, label set, criteria, model names
src/data.py       reads Derm7pt and PAD-UFES-20, maps labels, samples images
src/masks.py      automatic lesion masks and contact sheets for checking them
src/perturb.py    lesion removal and the three background edits
src/prompts.py    prompts and parsing of free-text answers
src/models.py     model loading, answer probabilities, text generation
src/run.py        prepare and infer steps
src/analyze.py    the four tests, bootstrap intervals, report and figures
```

## Limitations

- The lesion masks come from simple thresholding, not from expert outlines. They are dilated by
  12 pixels as a safety margin and checked by eye on contact sheets; images with bad masks are left
  out of tests (a) and (b).
- Because of that margin, the darker-skin edit leaves a narrow band of the original skin around the
  lesion. The change is faded in to avoid a hard edge, but the band is still visible.
- Removing a lesion by inpainting leaves a smooth patch that does not look exactly like skin, so some
  "cannot be determined" answers may be a reaction to the patch.
- Derm7pt's melanomas and nevi are sampled, not all used, and the sample is small, so the intervals
  will be wide. The pilot is meant to show whether the problems exist and whether the pipeline works,
  not to rank models.
- Answers are read from the first token. A model that would have written "The answer is B" is still
  scored on the letter it gives most probability to.
