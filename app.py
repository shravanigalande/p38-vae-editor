"""Gradio app: Text-Guided VAE Latent Editor (P38).  Run: python app.py"""
import os
import numpy as np
import gradio as gr
from PIL import Image
from common import *
import edit
from llm_parser import parse_instruction

X, Y = load_test_subset()

_gan = {}
def _get_gan():
    if "g" not in _gan:
        p = os.path.join(MODELS, "cgan_generator.keras")
        if os.path.exists(p):
            from tensorflow import keras
            _gan["g"] = keras.models.load_model(p, compile=False)
        else:
            _gan["g"] = None
    return _gan["g"]


def up(img, s=6):
    im = (np.clip(img, 0, 1) * 255).astype("uint8")
    return np.array(Image.fromarray(im).resize((im.shape[1] * s, im.shape[0] * s), Image.NEAREST))


def run(source, idx, instruction, manual_strength):
    s = CLASS_NAMES.index(source)
    pool = np.where(Y == s)[0]
    img = X[pool[int(idx) % len(pool)]]

    p = parse_instruction(instruction)
    if p["target"] is None:
        return up(img[..., 0]), None, None, p, "Couldn't find a clothing class in that instruction. Try e.g. 'more like a sandal'.", None
    t = CLASS_NAMES.index(p["target"])
    strength = manual_strength if manual_strength > 0 else p["strength"]

    z = edit.encode(img)
    original = edit.decode(z)[0]
    edited = edit.decode(edit.edit_latent(z, t, strength, s))[0]
    strip, alphas = edit.morph_strip(img, t, max_alpha=max(strength, 0.01), steps=9)
    strip_img = np.concatenate([up(f, 3) for f in strip], axis=1)

    # CLIP confirmation (U3) — optional so the app still runs if CLIP can't load
    try:
        import clip_check
        c = clip_check.confirm_edit(original, edited, p["target"])
        verdict = "✅ confirmed" if c["confirmed"] else "⚠️ not confirmed"
        clip_md = (f"**CLIP check: {verdict}**  \n"
                   f"P('{p['target']}') before: {c['target_prob_before']:.2f} → after: {c['target_prob_after']:.2f}  \n"
                   f"CLIP's top guess after edit: **{c['clip_top1_after']}**")
    except Exception as e:
        clip_md = f"CLIP unavailable: {e}"

    # cGAN comparison (U2): same noise, label swapped
    gan_img = None
    g = _get_gan()
    if g is not None:
        z_n = np.random.default_rng(int(idx)).normal(size=(2, 64)).astype("float32")
        lab = np.array([[s], [t]], dtype="int32")
        out = g.predict([z_n[:1].repeat(2, 0), lab], verbose=0)[..., 0]
        gan_img = np.concatenate([up(out[0], 4), up(out[1], 4)], axis=1)

    info = {**p, "strength_used": round(float(strength), 2), "source": source}
    return up(img[..., 0]), up(edited), strip_img, info, clip_md, gan_img


import subprocess, sys, glob


def run_evaluation():
    """Runs evaluate.py (needs internet for CLIP; works on a Hugging Face Space)."""
    r = subprocess.run([sys.executable, "evaluate.py"], capture_output=True, text=True, cwd=ROOT)
    log = (r.stdout + "\n" + r.stderr)[-3000:]
    def rd(name):
        p = os.path.join(RESULTS, name)
        return open(p).read() if os.path.exists(p) else "(missing)"
    figs = [p for p in [os.path.join(RESULTS, "pair_success.png"), os.path.join(RESULTS, "latent_pca.png")] if os.path.exists(p)]
    figs += sorted(glob.glob(os.path.join(RESULTS, "strips", "strip_*.png")))
    return rd("clip_table.md"), rd("failure_cases.md"), rd("vae_metrics.txt"), figs, log


with gr.Blocks(title="Text-Guided VAE Latent Editor") as demo:
    gr.Markdown("# 👕 Text-Guided VAE Latent Editor\n"
                "Pick a Fashion-MNIST item, type an instruction like *'make it more like a sandal'*. "
                "An LLM turns the sentence into a latent-space direction, the VAE decodes the edit, "
                "and CLIP checks whether it worked.")
    with gr.Tab("Editor"):
        with gr.Row():
            source = gr.Dropdown(CLASS_NAMES, value="sneaker", label="Source class")
            idx = gr.Slider(0, 99, value=0, step=1, label="Which item (0–99)")
        instr = gr.Textbox(value="make it more like a sandal", label="Instruction")
        manual = gr.Slider(0, 1.5, value=0, step=0.1, label="Manual strength (0 = let the LLM decide)")
        btn = gr.Button("Edit", variant="primary")
        with gr.Row():
            o1 = gr.Image(label="Original (VAE-decoded input)", height=220)
            o2 = gr.Image(label="Edited", height=220)
        strip_o = gr.Image(label="Morph strip (alpha from 0 to strength)")
        with gr.Row():
            parsed = gr.JSON(label="LLM-parsed instruction")
            clip_o = gr.Markdown()
        gan_o = gr.Image(label="cGAN comparison: same noise, label = source (left) vs target (right)")
        btn.click(run, [source, idx, instr, manual], [o1, o2, strip_o, parsed, clip_o, gan_o])
        gr.Examples([["sneaker", 0, "make it more like a sandal", 0],
                     ["t-shirt", 3, "slightly more like a pullover", 0],
                     ["dress", 1, "completely a trouser", 0]], [source, idx, instr, manual])
    with gr.Tab("Evaluation (for the report)"):
        gr.Markdown("Runs the full CLIP evaluation (5 instructions + all 90 class pairs). "
                    "Takes a few minutes. Copy the tables and save the figures for your report.")
        ebtn = gr.Button("Run evaluation", variant="primary")
        t1 = gr.Textbox(label="clip_table.md (copy this)", lines=10)
        t2 = gr.Textbox(label="failure_cases.md", lines=14)
        t3 = gr.Textbox(label="vae_metrics.txt", lines=3)
        gal = gr.Gallery(label="Figures (right-click to save)", columns=2, height=600)
        lg = gr.Textbox(label="log", lines=6)
        ebtn.click(run_evaluation, None, [t1, t2, t3, gal, lg])

if __name__ == "__main__":
    demo.launch()
