"""Thin wrapper around Hugging Face vision-language models.

Answer probabilities are read from the logits of the first answer token, so every
question costs one forward pass and gives a full distribution instead of one sample.
"""
import numpy as np

from . import config as C
from . import prompts as P


class VLM:
    def __init__(self, key: str, four_bit: bool | None = None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.key = key
        repo = C.MODELS[key]
        bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        # Gemma 3 models are unstable in float16, so without bfloat16 hardware (e.g. a T4)
        # MedGemma is loaded in 4 bits with float32 compute.
        if four_bit is None:
            four_bit = key == "medgemma" and not bf16
        compute = torch.bfloat16 if bf16 else (torch.float32 if key == "medgemma" else torch.float16)
        kw = {"device_map": "auto"}
        if four_bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute, bnb_4bit_use_double_quant=True)
        else:
            kw["torch_dtype"] = compute
        pkw = {}
        if key == "qwen":
            pkw = {"min_pixels": 256 * 28 * 28, "max_pixels": 640 * 28 * 28}
        self.processor = AutoProcessor.from_pretrained(repo, **pkw)
        self.model = AutoModelForImageTextToText.from_pretrained(repo, **kw).eval()
        self.dtype = compute
        self.four_bit = four_bit
        tok = self.processor.tokenizer
        self.letter_ids = [self._ids(tok, ch) for ch in C.LETTERS]
        self.yes_ids = self._ids(tok, "yes") + self._ids(tok, "Yes")
        self.no_ids = self._ids(tok, "no") + self._ids(tok, "No")

    @staticmethod
    def _ids(tok, s):
        out = []
        for v in (s, " " + s):
            ids = tok.encode(v, add_special_tokens=False)
            if len(ids) == 1:
                out.append(ids[0])
        if not out:
            raise ValueError(f"no single-token encoding for {s!r}")
        return sorted(set(out))

    @staticmethod
    def _conv(image, text):
        return [
            {"role": "system", "content": [{"type": "text", "text": P.SYSTEM}]},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]},
        ]

    def _inputs(self, image, text):
        inp = self.processor.apply_chat_template(
            self._conv(image, text), add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        return inp.to(self.model.device, dtype=self.dtype)

    def _batch_inputs(self, images, texts):
        """Left-padded batch, so the last position is the next token for every row."""
        self.processor.tokenizer.padding_side = "left"
        convs = [self._conv(im, t) for im, t in zip(images, texts)]
        inp = self.processor.apply_chat_template(
            convs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
            padding=True)
        return inp.to(self.model.device, dtype=self.dtype)

    def _batch_logits(self, images, texts):
        torch = self.torch
        with torch.inference_mode():
            out = self.model(**self._batch_inputs(images, texts))
        z = out.logits[:, -1].float()
        if not torch.isfinite(z).all():
            raise RuntimeError("non-finite logits; try four_bit=True or a GPU with bfloat16")
        return z

    def option_probs_batch(self, images, texts):
        z = self._batch_logits(images, texts)
        s = self.torch.stack([self.torch.logsumexp(z[:, ids], 1) for ids in self.letter_ids], 1)
        return self.torch.softmax(s, 1).cpu().numpy()

    def yes_prob_batch(self, images, texts):
        z = self._batch_logits(images, texts)
        y = self.torch.logsumexp(z[:, self.yes_ids], 1)
        n = self.torch.logsumexp(z[:, self.no_ids], 1)
        return self.torch.sigmoid(y - n).cpu().numpy().tolist()

    def generate_batch(self, images, texts, max_new_tokens=160):
        torch = self.torch
        inp = self._batch_inputs(images, texts)
        with torch.inference_mode():
            out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        new = out[:, inp["input_ids"].shape[1]:]
        return [t.strip() for t in self.processor.batch_decode(new, skip_special_tokens=True)]

    def _next_logits(self, image, text):
        torch = self.torch
        with torch.inference_mode():
            out = self.model(**self._inputs(image, text))
        z = out.logits[0, -1].float()
        if not torch.isfinite(z).all():
            raise RuntimeError("non-finite logits; try four_bit=True or a GPU with bfloat16")
        return z

    def option_probs(self, image, text) -> np.ndarray:
        """Probability of each answer letter, renormalised over the letters."""
        z = self._next_logits(image, text)
        s = self.torch.stack([self.torch.logsumexp(z[ids], 0) for ids in self.letter_ids])
        return self.torch.softmax(s, 0).cpu().numpy()

    def yes_prob(self, image, text) -> float:
        z = self._next_logits(image, text)
        y = self.torch.logsumexp(z[self.yes_ids], 0)
        n = self.torch.logsumexp(z[self.no_ids], 0)
        return float(self.torch.sigmoid(y - n))

    def generate(self, image, text, max_new_tokens=220) -> str:
        torch = self.torch
        inp = self._inputs(image, text)
        with torch.inference_mode():
            out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        new = out[0, inp["input_ids"].shape[1]:]
        return self.processor.decode(new, skip_special_tokens=True).strip()


class FakeVLM:
    """Deterministic stand-in used by the tests; behaves like a weak, biased model."""

    def __init__(self, key="fake"):
        self.key = key

    def option_probs_batch(self, images, texts):
        return np.stack([self.option_probs(i, t) for i, t in zip(images, texts)])

    def yes_prob_batch(self, images, texts):
        return [self.yes_prob(i, t) for i, t in zip(images, texts)]

    def generate_batch(self, images, texts, max_new_tokens=160):
        return [self.generate(i, t) for i, t in zip(images, texts)]

    def _rng(self, image, text):
        import zlib
        a = np.asarray(image)
        return np.random.default_rng(zlib.crc32(a[::7, ::7].tobytes() + text.encode()))

    def option_probs(self, image, text):
        z = self._rng(image, text).normal(0, 1.5, len(C.LETTERS))
        z[0] += 1.0  # likes letter A
        e = np.exp(z - z.max())
        return e / e.sum()

    def yes_prob(self, image, text):
        return float(self._rng(image, text).uniform())

    def generate(self, image, text, max_new_tokens=220):
        r = self._rng(image, text)
        dx = C.CLASSES[int(r.integers(len(C.CLASSES)))]
        return (f"Description: A pigmented lesion with uneven borders. It is about the size of the frame.\n"
                f"Features: asymmetry, irregular border\nDiagnosis: {dx}\nConfidence: {int(r.integers(40, 100))}")
