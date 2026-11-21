# Demo video script (target 2:30)

Record the screen + your voice (OBS, Zoom "record", or Loom). Keep it under 3 min.

| Time | Show | Say |
|---|---|---|
| 0:00–0:15 | Title / README | "Hi, I'm Shanu, Roll No. 38. My project is P38, a text-guided VAE latent editor on Fashion-MNIST. You type 'make it more like a sandal' and the image changes." |
| 0:15–0:45 | `latent_pca.png` | "U1: I trained a VAE in TensorFlow 2 and computed the average latent code, the centroid, for each of the 10 classes. Here they are in 2D." |
| 0:45–1:30 | Live app: sneaker → "more like a sandal" | "U2: Gemini turns my sentence into a target class and a strength. I move the latent code along centroid(target) − centroid(source). Here's the morph strip from 0 to full." |
| 1:30–1:55 | CLIP panel in the app | "U3: CLIP checks the result. Probability of 'sandal' went from X to Y, so the edit is confirmed." |
| 1:55–2:10 | cGAN image in app | "For comparison, a conditional GAN with the same noise and a swapped label. It can't edit a real image, but it shows the same class change." |
| 2:10–2:35 | `pair_success.png`, `failure_cases.md` | "U5: Overall X% of 90 class pairs were confirmed. It fails on bag → sneaker because the shapes are too different for a straight line in latent space. 28×28 images also limit how much CLIP can tell." |
| 2:35–2:45 | GitHub + Space URL | "Code is on GitHub, the live app is on Hugging Face Spaces. Thanks!" |
