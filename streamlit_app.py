import os, sys, glob, subprocess
import numpy as np
import streamlit as st
from PIL import Image

try:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

from common import CLASS_NAMES, MODELS, RESULTS, ROOT, load_test_subset
import edit
from llm_parser import parse_instruction

st.set_page_config(page_title="Text-Guided VAE Latent Editor", page_icon="👕", layout="wide")


@st.cache_resource
def load_everything():
    edit._load()
    X, Y = load_test_subset()
    gan = None
    gp = os.path.join(MODELS, "cgan_generator.keras")
    if os.path.exists(gp):
        from tensorflow import keras
        gan = keras.models.load_model(gp, compile=False)
    return X, Y, gan


def up(img, s=6):
    im = (np.clip(img, 0, 1) * 255).astype("uint8")
    return np.array(Image.fromarray(im).resize((im.shape[1] * s, im.shape[0] * s), Image.NEAREST))


X, Y, GAN = load_everything()

st.title("👕 Text-Guided VAE Latent Editor")
st.caption("Pick a Fashion-MNIST item and type an instruction like 'make it more like a sandal'.")

tab1, tab2 = st.tabs(["Editor", "Evaluation (for the report)"])

with tab1:
    c1, c2 = st.columns(2)
    source = c1.selectbox("Source class", CLASS_NAMES, index=CLASS_NAMES.index("sneaker"))
    idx = c2.slider("Which item (0-99)", 0, 99, 0)
    instr = st.text_input("Instruction", "make it more like a sandal")
    manual = st.slider("Manual strength (0 = let the LLM decide)", 0.0, 1.5, 0.0, 0.1)
    use_clip = st.checkbox("Verify with CLIP (slower)", value=False)

    if st.button("Edit", type="primary"):
        s = CLASS_NAMES.index(source)
        pool = np.where(Y == s)[0]
        img = X[pool[idx % len(pool)]]
        p = parse_instruction(instr)
        if p["target"] is None:
            st.warning("Couldn't find a clothing class in that instruction. Try 'more like a sandal'.")
        else:
            t = CLASS_NAMES.index(p["target"])
            strength = manual if manual > 0 else p["strength"]
            z = edit.encode(img)
            original = edit.decode(z)[0]
            edited = edit.decode(edit.edit_latent(z, t, strength, s))[0]
            strip, alphas = edit.morph_strip(img, t, max_alpha=max(strength, 0.01), steps=9)

            a, b = st.columns(2)
            a.image(up(img[..., 0]), caption="Original", clamp=True)
            b.image(up(edited), caption=f"Edited to {p['target']} (strength {strength:.2f})", clamp=True)
            st.image(np.concatenate([up(f, 3) for f in strip], axis=1),
                     caption="Morph strip (alpha from 0 to strength)", clamp=True)
            st.json({**p, "strength_used": round(float(strength), 2), "source": source})

            if use_clip:
                try:
                    import clip_check
                    c = clip_check.confirm_edit(original, edited, p["target"])
                    msg = "confirmed" if c["confirmed"] else "not confirmed"
                    st.info(f"CLIP check: {msg}  \n"
                            f"P('{p['target']}') before: {c['target_prob_before']:.2f} -> "
                            f"after: {c['target_prob_after']:.2f}  \n"
                            f"CLIP's top guess after edit: {c['clip_top1_after']}")
                except Exception as e:
                    st.error(f"CLIP unavailable: {e}")

            if GAN is not None:
                z_n = np.random.default_rng(idx).normal(size=(1, 64)).astype("float32")
                out = GAN.predict([z_n.repeat(2, 0), np.array([[s], [t]], dtype="int32")], verbose=0)[..., 0]
                st.image(np.concatenate([up(out[0], 4), up(out[1], 4)], axis=1),
                         caption="cGAN comparison: same noise, source label (left) vs target label (right)", clamp=True)

with tab2:
    st.write("Runs the full CLIP evaluation. Takes a few minutes. Copy the tables and save the figures.")
    if st.button("Run evaluation"):
        with st.spinner("Running evaluation... this can take several minutes"):
            r = subprocess.run([sys.executable, "evaluate.py"], capture_output=True, text=True, cwd=ROOT)
        rd = lambda n: open(os.path.join(RESULTS, n)).read() if os.path.exists(os.path.join(RESULTS, n)) else "(missing)"
        st.subheader("clip_table.md"); st.code(rd("clip_table.md"))
        st.subheader("failure_cases.md"); st.code(rd("failure_cases.md"))
        st.subheader("vae_metrics.txt"); st.code(rd("vae_metrics.txt"))
        figs = ["pair_success.png", "latent_pca.png"] + [os.path.join("strips", os.path.basename(p)) for p in sorted(glob.glob(os.path.join(RESULTS, "strips", "strip_*.png")))]
        for f in figs:
            p = os.path.join(RESULTS, f)
            if os.path.exists(p):
                st.image(p)
        if r.returncode != 0:
            st.error(r.stderr[-2000:])
