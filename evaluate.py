"""Produces every number/figure the report needs.

 1. Morph strips for 5 instructions                -> results/strips/*.png
 2. CLIP before/after table (+ cGAN comparison)    -> results/clip_table.md / .csv
 3. All 90 source->target pairs, CLIP success rate -> results/pair_success.png, failure_cases.md

Usage: python evaluate.py            (needs models/ from train_vae.py, optionally train_gan.py)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
import edit
from llm_parser import parse_instruction
import clip_check

X, Y = load_test_subset()

INSTRUCTIONS = [  # (source class, instruction)
    ("sneaker", "make it more like a sandal"),
    ("sandal", "make it more like an ankle boot"),
    ("t-shirt", "make it more like a pullover"),
    ("dress", "make it completely a trouser"),
    ("bag", "make it slightly more like a sneaker"),
]

gan = None
gpath = os.path.join(MODELS, "cgan_generator.keras")
if os.path.exists(gpath):
    from tensorflow import keras
    gan = keras.models.load_model(gpath, compile=False)


def gan_target_prob(target, n=8):
    """CLIP prob of the target class for cGAN samples conditioned on the target label."""
    z = np.random.default_rng(0).normal(size=(n, 64)).astype("float32")
    imgs = gan.predict([z, np.full((n, 1), CLASS_NAMES.index(target), dtype="int32")], verbose=0)[..., 0]
    return float(clip_check.class_probs(imgs)[:, CLASS_NAMES.index(target)].mean())


# ---------------- 1 + 2: five instructions ----------------
rows = []
for k, (src, text) in enumerate(INSTRUCTIONS, 1):
    p = parse_instruction(text)
    tgt, strength = p["target"], p["strength"]
    img = X[np.where(Y == CLASS_NAMES.index(src))[0][0]]
    strip, alphas = edit.morph_strip(img, CLASS_NAMES.index(tgt), max_alpha=1.0, steps=9)

    fig, ax = plt.subplots(1, 9, figsize=(13, 1.9))
    for a_, im, al in zip(ax, strip, alphas):
        a_.imshow(im, cmap="gray"); a_.set_title(f"α={al:.2f}", fontsize=8); a_.axis("off")
    fig.suptitle(f'"{text}"  ({src} → {tgt})', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "strips", f"strip_{k}.png"), dpi=150); plt.close(fig)

    z = edit.encode(img)
    edited = edit.decode(edit.edit_latent(z, CLASS_NAMES.index(tgt), strength, CLASS_NAMES.index(src)))[0]
    original = edit.decode(z)[0]          # decoded original = fair "before" (same blur)
    c = clip_check.confirm_edit(original, edited, tgt)
    row = {"instruction": text, "source": src, "target": tgt, "strength": strength,
           "parser": p["parser"], "clip_before": c["target_prob_before"],
           "clip_after": c["target_prob_after"], "gain": c["gain"],
           "clip_top1_after": c["clip_top1_after"], "confirmed": c["confirmed"]}
    if gan is not None:
        row["cgan_clip_target_prob"] = gan_target_prob(tgt)
    rows.append(row)
    print(row)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULTS, "clip_table.csv"), index=False)
with open(os.path.join(RESULTS, "clip_table.md"), "w") as f:
    f.write(df.round(3).to_markdown(index=False))
print(f"\nConfirmed edits: {int(df.confirmed.sum())}/{len(df)}")

# ---------------- 3: all pairs sweep (failure analysis) ----------------
N = 5
M = np.zeros((10, 10)); Pm = np.zeros((10, 10))
for s in range(10):
    imgs = X[np.where(Y == s)[0][:N]]
    zs = np.stack([edit.encode(i) for i in imgs])
    for t in range(10):
        if s == t:
            continue
        edited = edit.decode(np.stack([edit.edit_latent(z, t, 1.0, s) for z in zs]))
        p = clip_check.class_probs(edited)
        M[s, t] = (p.argmax(1) == t).mean()
        Pm[s, t] = p[:, t].mean()
np.save(os.path.join(RESULTS, "pair_success.npy"), M)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
ax.set_xticks(range(10)); ax.set_xticklabels(CLASS_NAMES, rotation=60, ha="right", fontsize=8)
ax.set_yticks(range(10)); ax.set_yticklabels(CLASS_NAMES, fontsize=8)
ax.set_xlabel("target class"); ax.set_ylabel("source class")
ax.set_title("CLIP-confirmed edit rate at α=1 (5 images per pair)")
fig.colorbar(im); fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "pair_success.png"), dpi=150); plt.close(fig)

off = ~np.eye(10, dtype=bool)
overall = M[off].mean()
pairs = sorted([(M[s, t], Pm[s, t], CLASS_NAMES[s], CLASS_NAMES[t]) for s in range(10) for t in range(10) if s != t])
with open(os.path.join(RESULTS, "failure_cases.md"), "w") as f:
    f.write(f"Overall CLIP-confirmed rate across 90 pairs: {overall:.1%}\n\n")
    f.write("Worst 8 pairs (use one as the failure case in the report):\n\n")
    for s_, p_, a_, b_ in pairs[:8]:
        f.write(f"- {a_} -> {b_}: success {s_:.0%}, mean CLIP target prob {p_:.2f}\n")
    f.write("\nBest 5 pairs:\n\n")
    for s_, p_, a_, b_ in pairs[-5:][::-1]:
        f.write(f"- {a_} -> {b_}: success {s_:.0%}, mean CLIP target prob {p_:.2f}\n")
print(open(os.path.join(RESULTS, "failure_cases.md")).read())
