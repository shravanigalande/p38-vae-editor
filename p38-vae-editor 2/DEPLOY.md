# Deploy checklist

## 1. Models are already trained (in `models/`). No Colab needed.

(Old Colab instructions, only if you ever want to retrain:)
Open `colab_run.ipynb` -> Runtime -> Change runtime type -> T4 GPU -> Run all.
It trains the VAE + cGAN, runs `evaluate.py`, and zips the outputs.
Download `p38_outputs.zip` and unzip it into your repo folder (it fills `models/`, `assets/`, `results/`).

## 2. GitHub
```bash
cd p38-vae-editor
git init && git add . && git commit -m "P38 text-guided VAE latent editor"
git branch -M main
git remote add origin https://github.com/<you>/p38-vae-editor.git
git push -u origin main
```
(Create the empty public repo on github.com first.) The model files are small (a few MB), so they are committed normally.

## 3. Hugging Face Space
1. huggingface.co -> New Space -> SDK **Gradio**, hardware **CPU basic (free)**.
2. Push the same files: `git remote add space https://huggingface.co/spaces/<you>/p38-vae-editor` then `git push space main`
   (use a HF access token with *write* as the password).
3. Space -> Settings -> **Variables and secrets** -> New secret: `GEMINI_API_KEY`.
4. Wait for the build, open the URL and test "make it more like a sandal".
5. Paste the Space URL into `README.md`, push again.

## 4. Submit (class shared drive, folder `38_Shanu`)
- PDF report
- Live app URL + GitHub link (a small text/PDF file or in the report's first page)
- 2-3 min demo video
