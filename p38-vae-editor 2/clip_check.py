"""U3: confirm each edit with CLIP (zero-shot classification of the edited image)."""
import numpy as np
from PIL import Image
from common import CLASS_NAMES

_m = {}


def _load():
    if not _m:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        name = "openai/clip-vit-base-patch32"
        _m["model"] = CLIPModel.from_pretrained(name).eval()
        _m["proc"] = CLIPProcessor.from_pretrained(name)
        _m["torch"] = torch
    return _m


def class_probs(imgs):
    """imgs: (N,28,28) floats in [0,1]. Returns (N,10) CLIP probabilities over the 10 classes."""
    m = _load()
    pil = [Image.fromarray((np.clip(i, 0, 1) * 255).astype("uint8")).convert("RGB")
           .resize((224, 224), Image.BICUBIC) for i in imgs]
    texts = [f"a photo of a {c}" for c in CLASS_NAMES]
    inp = m["proc"](text=texts, images=pil, return_tensors="pt", padding=True)
    with m["torch"].no_grad():
        logits = m["model"](**inp).logits_per_image
    return logits.softmax(-1).numpy()


def confirm_edit(before, after, target):
    """Compare CLIP's belief in the target class before vs after the edit."""
    p = class_probs(np.stack([before, after]))
    t = CLASS_NAMES.index(target)
    b, a = float(p[0, t]), float(p[1, t])
    return {
        "target_prob_before": b,
        "target_prob_after": a,
        "gain": a - b,
        "clip_top1_after": CLASS_NAMES[int(p[1].argmax())],
        "confirmed": bool(a > b and p[1].argmax() == t),
    }
