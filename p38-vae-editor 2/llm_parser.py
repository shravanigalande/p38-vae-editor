"""U2: map a natural-language instruction to a latent edit using an LLM.

The LLM returns JSON {"target": <one of the 10 classes>, "strength": 0.0-1.5}.
If no GEMINI_API_KEY is set (or the call fails), a keyword parser is used instead,
so the app never breaks during a demo.
Set the key in the platform's *secrets* settings, never in the repo.
"""
import json, os, re
import requests
from common import CLASS_NAMES

SYNONYMS = {
    "t-shirt": ["t-shirt", "tshirt", "t shirt", "tee"],
    "trouser": ["trouser", "trousers", "pants", "jeans"],
    "pullover": ["pullover", "sweater", "jumper", "hoodie"],
    "dress": ["dress", "gown"],
    "coat": ["coat", "jacket"],
    "sandal": ["sandal", "sandals", "flip-flop", "flip flop", "slipper"],
    "shirt": ["shirt", "blouse"],
    "sneaker": ["sneaker", "sneakers", "trainer", "trainers", "shoe", "shoes"],
    "bag": ["bag", "handbag", "purse", "backpack"],
    "ankle boot": ["ankle boot", "boot", "boots"],
}
STRENGTH_WORDS = [
    (["slightly", "a little", "a bit", "somewhat", "hint"], 0.3),
    (["much more", "very", "totally", "completely", "fully", "entirely", "exactly"], 1.0),
    (["extremely", "exaggerate", "extra"], 1.3),
]
DEFAULT_STRENGTH = 0.7


def keyword_parse(text):
    t = text.lower()
    best, best_pos = None, -1
    # longest synonym wins, later mention wins ("less shirt, more like a sandal")
    for cls, syns in SYNONYMS.items():
        for s in sorted(syns, key=len, reverse=True):
            for m in re.finditer(r"\b" + re.escape(s) + r"\b", t):
                if m.start() >= best_pos:
                    best, best_pos = cls, m.start()
    strength = DEFAULT_STRENGTH
    for words, val in STRENGTH_WORDS:
        if any(w in t for w in words):
            strength = val
            break
    return {"target": best, "strength": strength, "parser": "keyword"}


def llm_parse(text):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    prompt = (
        "You convert an image-editing instruction for a Fashion-MNIST item into JSON.\n"
        f"Allowed target classes: {CLASS_NAMES}.\n"
        'Return ONLY JSON: {"target": "<class>", "strength": <number 0.0-1.5>}.\n'
        "strength: 0.3 for 'slightly/a bit', 0.7 for plain 'more like', 1.0 for "
        "'completely/totally', 1.3 for exaggerated. If no class can be inferred use "
        'target "none".\n'
        f"Instruction: {text!r}"
    )
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    r = requests.post(
        url, params={"key": key}, timeout=15,
        json={"contents": [{"parts": [{"text": prompt}]}],
              "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}},
    )
    r.raise_for_status()
    raw = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    d = json.loads(raw)
    tgt = str(d.get("target", "")).lower().strip()
    if tgt not in CLASS_NAMES:
        return {"target": None, "strength": 0.0, "parser": "llm"}
    return {"target": tgt, "strength": float(min(max(d.get("strength", DEFAULT_STRENGTH), 0), 1.5)),
            "parser": "llm"}


def parse_instruction(text):
    """Returns dict(target=<class or None>, strength=float, parser=llm|keyword)."""
    try:
        out = llm_parse(text)
        if out is not None:
            return out
    except Exception as e:  # network error, quota, bad JSON ...
        print("LLM parse failed, using keyword fallback:", e)
    return keyword_parse(text)
