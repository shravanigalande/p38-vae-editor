---
title: Text-Guided VAE Latent Editor
emoji: 👕
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
---

# Text-Guided VAE Latent Editor (P38)

Edit a Fashion-MNIST item with a sentence like **"make it more like a sandal"**.

**Live app:** `<PASTE HUGGING FACE SPACE URL>`  ·  **Demo video:** `<PASTE VIDEO LINK>`

## How it works
1. **VAE (TensorFlow 2)** encodes the image into a 16-D latent code.
2. **Class centroids (U1):** the mean latent code of each of the 10 classes.
3. **LLM (U2):** Gemini converts the sentence into `{"target": "sandal", "strength": 0.7}` (keyword fallback if no API key).
4. **Edit:** `z' = z + strength × (centroid[target] − centroid[source])`, then the decoder draws the result. A morph strip shows α going 0 → strength.
5. **CLIP (U3)** zero-shot classifies the edited image and confirms whether the target class became more likely.
6. **cGAN comparison (U2):** same noise vector, class label swapped.

| Unit | Task | File |
|---|---|---|
| U1 | Class centroids in latent space | `train_vae.py` |
| U2 | LLM instruction → latent direction; compare with GAN | `llm_parser.py`, `edit.py`, `train_gan.py` |
| U3 | Confirm each edit with CLIP | `clip_check.py` |
| U4 | VAE in TF2, morph strips | `train_vae.py`, `evaluate.py` |
| U5 | Limits of small, low-res data | `MODEL_CARD.md`, report |

## Run it yourself
```bash
pip install -r requirements.txt
python train_vae.py --epochs 30     # trains VAE, saves centroids + test subset
python train_gan.py --epochs 30     # comparison model
python evaluate.py                  # morph strips, CLIP table, failure cases -> results/
export GEMINI_API_KEY=...           # optional; free key from Google AI Studio
python app.py
```
The trained VAE/cGAN files are already in `models/`, so you can skip training and just launch the app.

## Results
See `results/clip_table.md`, `results/strips/`, `results/pair_success.png`, `results/failure_cases.md`.
Open the app's **Evaluation** tab and click *Run evaluation* (or run `python evaluate.py`) to produce the CLIP table, morph strips and failure cases, then paste the headline numbers here.

## Secrets
The Gemini key lives in the Space's **Settings → Variables and secrets** as `GEMINI_API_KEY`. It is never committed.
