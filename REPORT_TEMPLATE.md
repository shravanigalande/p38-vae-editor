# Text-Guided VAE Latent Editor — Project Report (P38)
Shanu · Roll No. 38 · Generative AI mini-project · MIT ADT Pune
Live app: <URL> · GitHub: <URL> · Demo video: <URL>

## 1. Problem and goal
Edit a Fashion-MNIST item with a sentence such as "more like a sandal". (2–3 sentences: why latent editing is interesting.)

## 2. System overview
Diagram: instruction → LLM parser → latent direction → VAE decoder → edited image → CLIP check. (Draw in draw.io or paste a screenshot.)

## 3. Methods by unit
- **U1 Foundations:** Fashion-MNIST, VAE basics (reparameterisation, ELBO), class centroids. Figure: `latent_pca.png`.
- **U2 LLM + GAN comparison:** prompt used, JSON output, strength words; cGAN design; how the two edits differ.
- **U3 Multimodal:** CLIP zero-shot confirmation, prompt template "a photo of a {class}".
- **U4 Frameworks / generation:** TF2 VAE architecture table, training setup, loss curve `vae_loss.png`, morph strips (5 instructions).
- **U5 Limits & ethics:** see section 5.

## 4. Results
- VAE test reconstruction MSE: <x>
- Table: 5 instructions × (parsed target, strength, CLIP before, CLIP after, confirmed, cGAN CLIP prob) — paste `results/clip_table.md`
- Figure: 5 morph strips (`results/strips/`)
- Figure: `pair_success.png`; overall confirmed rate <x%> over 90 pairs
- Short discussion: which edits work, VAE vs cGAN.

## 5. Failure case and limitations
One concrete failure from `failure_cases.md` (e.g. bag → sneaker) with the image, and why it fails (non-linear class differences, tiny images, CLIP at 28×28). Then the limits listed in MODEL_CARD.md.

## 6. Deployment
Hugging Face Spaces (Gradio), API key in secrets, repo structure, how to run.

## 7. Conclusion and future work
(e.g. β-VAE or conditional VAE, higher resolution, CelebA/CIFAR.)

## References
Kingma & Welling 2014 (VAE); Radford et al. 2021 (CLIP); Mirza & Osindero 2014 (cGAN); Xiao et al. 2017 (Fashion-MNIST).
