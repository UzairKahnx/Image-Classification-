# CNN Image Classification Results Dashboard

A minimal Streamlit app that presents the measured evaluation of a PyTorch CNN trained on the
Intel Image Classification dataset (6 classes), and lets you classify your own image with it.

**Reported result: 88.03% test accuracy.** Open the **Try an image** tab to upload a JPG/PNG and
see the model's prediction and confidence for all six classes.

**Live demo:** [https://uzairkahnx-image-classification--app-enn4ma.streamlit.app/](https://uzairkahnx-image-classification--app-enn4ma.streamlit.app/)

## Files in this repo

- `app.py` — Streamlit dashboard
- `results.json` — measured metrics, per-epoch history and confusion matrix
- `training_curves.png` — exported accuracy/loss plot
- `confusion_matrix.png` — exported confusion matrix
- `best_cnn_model.pth` — best validation checkpoint (available as a download in the app)
- `cnn_weights.npz` — weights exported from the checkpoint for fast, torch-free inference
- `requirements.txt` — Python dependencies

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

[Deploy this app on Streamlit Cloud](https://share.streamlit.io/deploy?repository=UzairKahnx/Image-Classification-&branch=main&mainModule=app.py)

1. Fork/push this repository to GitHub.
2. Go to <https://share.streamlit.io/> and sign in with GitHub.
3. **New app** → select this repository, branch `main`, main file `app.py`.
4. Click **Deploy**.
