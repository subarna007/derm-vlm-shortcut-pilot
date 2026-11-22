Per-image model answers from the pilot runs.

- `mcq.jsonl`: answer probabilities for every image, image version and option order.
- `criteria.jsonl`: probability of "yes" for each 7-point checklist criterion (Derm7pt images).
- `describe.jsonl`: the free-text answers.
- `sample.csv`: which images were used and whether their lesion mask passed the checks.

Image ids are `d7_<case_num>` for Derm7pt and `pad_<img_id>` for PAD-UFES-20. Diagnoses and expert labels are not
copied here; they come from the datasets themselves (Derm7pt is CC BY-NC-ND 4.0).
