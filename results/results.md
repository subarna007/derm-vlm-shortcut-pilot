# Pilot results

All numbers come from `python -m src.analyze` on the raw answers in `work/raw/`. Intervals are 95% bootstrap intervals over images. Percentages unless noted.

## google/medgemma-1.5-4b-it

### Baseline diagnosis (original images, forced choice among 8 diagnoses)

| Dataset | n | Accuracy | Balanced accuracy | Chose "cannot be determined" |
|---|---|---|---|---|
| derm7pt | 256 | 33.2 (27.3–39.1) | 28.0 (23.8–32.0) | 9.0 |
| pad | 205 | 42.4 (36.1–49.3) | 43.4 (38.7–48.4) | 29.8 |

Chance balanced accuracy is 12.5.

### (a) Lesion removed

337 images with a usable mask.

| | Original | Lesion removed |
|---|---|---|
| Chose "cannot be determined" | 18.4 (14.2–22.6) | 48.1 (42.7–53.1) |
| Mean confidence (forced choice) | 75.7 | 70.4 |
| Accuracy | 39.5 | 28.8 |

Still a confident diagnosis (not abstaining, confidence ≥ 50%) with the lesion removed: 50.7 (45.7–56.1). Same diagnosis as on the original: 32.6 (27.6–37.7).

### (b) Background changed, lesion pixels identical

| Change | n | Diagnosis flipped | Mean change in P(melanoma), points | Accuracy before → after |
|---|---|---|---|---|
| bg_tone | 337 | 13.4 (9.8–17.2) | +0.7 (-0.7 to +2.1) | 39.5 → 36.5 |
| bg_marker | 337 | 10.1 (7.1–13.4) | -0.1 (-1.0 to +0.8) | 39.5 → 39.8 |
| bg_ruler | 337 | 6.2 (3.9–8.9) | +1.9 (+1.1 to +2.8) | 39.5 → 38.9 |

### (c) 7-point checklist criteria against Derm7pt expert labels

| Criterion | n | Prevalence | AUROC | Model says yes |
|---|---|---|---|---|
| atypical_network | 256 | 17.2 | 0.807 (0.744–0.865) | 59.0 |
| blue_white_veil | 256 | 19.5 | 0.843 (0.792–0.889) | 64.5 |
| atypical_vascular | 256 | 5.5 | 0.617 (0.455–0.761) | 38.7 |
| irregular_streaks | 256 | 19.9 | 0.754 (0.687–0.820) | 69.5 |
| irregular_pigmentation | 256 | 32.8 | 0.771 (0.709–0.829) | 94.1 |
| irregular_dots | 256 | 46.5 | 0.625 (0.558–0.692) | 94.5 |
| regression | 256 | 19.5 | 0.524 (0.441–0.606) | 73.0 |

Consistency between the model's own checklist and its own diagnosis (Spearman correlation of summed criterion probabilities with P(melanoma)): 0.71. Expert checklist with model P(melanoma): 0.48.

### (d) Calibration (original images)

| Dataset | Accuracy | Mean token confidence | Token ECE | Verbal ECE (answers that named a diagnosis) |
|---|---|---|---|---|
| derm7pt | 33.2 | 68.3 | 0.351 (0.295–0.408) | 0.611 (0.554–0.670) (242/256) |
| pad | 42.4 | 80.6 | 0.388 (0.322–0.451) | 0.449 (0.372–0.524) (156/205) |

## Qwen/Qwen2.5-VL-3B-Instruct

### Baseline diagnosis (original images, forced choice among 8 diagnoses)

| Dataset | n | Accuracy | Balanced accuracy | Chose "cannot be determined" |
|---|---|---|---|---|
| derm7pt | 256 | 27.7 (21.9–33.2) | 20.9 (18.2–23.9) | 30.5 |
| pad | 205 | 27.8 (21.5–34.1) | 27.7 (21.9–33.5) | 81.0 |

Chance balanced accuracy is 12.5.

### (a) Lesion removed

337 images with a usable mask.

| | Original | Lesion removed |
|---|---|---|
| Chose "cannot be determined" | 59.9 (54.6–65.0) | 96.7 (94.7–98.5) |
| Mean confidence (forced choice) | 40.8 | 26.2 |
| Accuracy | 29.1 | 15.7 |

Still a confident diagnosis (not abstaining, confidence ≥ 50%) with the lesion removed: 2.4 (0.9–4.2). Same diagnosis as on the original: 30.3 (25.5–35.6).

### (b) Background changed, lesion pixels identical

| Change | n | Diagnosis flipped | Mean change in P(melanoma), points | Accuracy before → after |
|---|---|---|---|---|
| bg_tone | 337 | 29.1 (24.3–34.1) | +2.3 (+1.2 to +3.4) | 29.1 → 27.0 |
| bg_marker | 337 | 23.7 (19.3–28.2) | +0.7 (-0.2 to +1.5) | 29.1 → 29.1 |
| bg_ruler | 337 | 19.0 (14.8–23.1) | +1.1 (+0.4 to +1.8) | 29.1 → 28.5 |

### (c) 7-point checklist criteria against Derm7pt expert labels

| Criterion | n | Prevalence | AUROC | Model says yes |
|---|---|---|---|---|
| atypical_network | 256 | 17.2 | 0.668 (0.584–0.746) | 98.8 |
| blue_white_veil | 256 | 19.5 | 0.738 (0.661–0.816) | 29.7 |
| atypical_vascular | 256 | 5.5 | 0.665 (0.520–0.799) | 92.6 |
| irregular_streaks | 256 | 19.9 | 0.589 (0.504–0.673) | 83.2 |
| irregular_pigmentation | 256 | 32.8 | 0.590 (0.515–0.663) | 99.6 |
| irregular_dots | 256 | 46.5 | 0.554 (0.481–0.626) | 100.0 |
| regression | 256 | 19.5 | 0.457 (0.368–0.542) | 49.2 |

Consistency between the model's own checklist and its own diagnosis (Spearman correlation of summed criterion probabilities with P(melanoma)): 0.25. Expert checklist with model P(melanoma): 0.45.

### (d) Calibration (original images)

| Dataset | Accuracy | Mean token confidence | Token ECE | Verbal ECE (answers that named a diagnosis) |
|---|---|---|---|---|
| derm7pt | 27.7 | 51.1 | 0.234 (0.183–0.290) | 0.528 (0.358–0.678) (36/256) |
| pad | 27.8 | 32.7 | 0.076 (0.048–0.142) | 0.666 (0.455–0.839) (19/205) |
