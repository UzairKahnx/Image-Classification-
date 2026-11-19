# CNN Image Classification Results Dashboard

A minimal Streamlit app that presents the measured evaluation of a PyTorch CNN trained on the
Intel Image Classification dataset (6 classes).

**Reported result: 88.03% test accuracy.**

## Files in this repo

- `app.py` — Streamlit dashboard
- `results.json` — measured metrics, per-epoch history and confusion matrix
- `training_curves.png` — exported accuracy/loss plot
- `confusion_matrix.png` — exported confusion matrix
- `best_cnn_model.pth` — best validation checkpoint (available as a download in the app)
- `requirements.txt` — Python dependencies

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Fork/push this repository to GitHub.
2. Go to <https://share.streamlit.io/> and sign in with GitHub.
3. **New app** → select this repository, branch `main`, main file `app.py`.
4. Click **Deploy**.
