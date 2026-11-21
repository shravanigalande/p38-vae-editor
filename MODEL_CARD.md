# Model Card — Text-Guided VAE Latent Editor

**Author:** Shanu (Roll No. 38) · **Course:** Generative AI · **Project:** P38

## Models
| Model | Details |
|---|---|
| VAE | Conv encoder (2× stride-2 conv → dense 128 → μ, log σ² of 16-D), mirrored decoder, BCE + KL (β=1), Adam 1e-3, 12 epochs, TensorFlow 2 / Keras |
| cGAN | Conditional DCGAN-style, 64-D noise + label embedding, 12 epochs (comparison only) |
| CLIP | `openai/clip-vit-base-patch32`, zero-shot, prompts "a photo of a {class}" (not trained by me) |
| LLM | Gemini (free tier) for instruction parsing; keyword fallback |

## Data
Fashion-MNIST: 60,000 train / 10,000 test, 28×28 greyscale, 10 clothing classes. No personal data.

## Metrics (fill from `results/`)
- VAE test reconstruction MSE: 0.0130 (12 epochs)
- CLIP-confirmed edits (5 demo instructions): `<x/5>`
- CLIP-confirmed rate over all 90 class pairs at α=1: `<from failure_cases.md>`

## Intended use
Educational demo of latent-space editing. Not for production or real fashion/retail images.

## Limitations (U5)
- 28×28 greyscale is tiny: edits are blurry and CLIP (trained on large colour photos) is an unreliable judge at this size.
- Visually similar classes (shirt / t-shirt / pullover / coat) overlap in latent space, so edits between them look subtle.
- Centroid arithmetic assumes the class difference is one straight line in latent space; real class changes are not linear (e.g. bag → sneaker fails).
- The VAE blurs fine detail by design (Gaussian prior + BCE loss).
- The cGAN cannot take a real image as input, so it is a different kind of edit, not a like-for-like one.
- The LLM can mis-parse unusual wording; unknown classes return "none" and no edit is made.
- Dataset reflects only 10 Western-retail clothing categories.
