import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="CNN Scene Classification Results",
    page_icon="📈",
    layout="wide",
)

BASE = Path(__file__).resolve().parent


@st.cache_data
def load_results():
    with (BASE / "results.json").open(encoding="utf-8") as f:
        return json.load(f)


results = load_results()
classes = results["classes"]
cm = np.array(results["confusion_matrix"])
epochs = list(range(1, results["epochs"] + 1))


# ---------------------------------------------------------------------------
# Inference: exact NumPy implementation of the trained PyTorch CNN.
# The weights were exported from best_cnn_model.pth, so predictions match the
# original model while Streamlit Cloud only needs numpy + Pillow (no torch).
# ---------------------------------------------------------------------------
@st.cache_resource
def load_weights():
    with np.load(BASE / "cnn_weights.npz") as data:
        return {key: data[key] for key in data.files}


def conv_bn_relu_pool(x, weight, gamma, beta, mean, var):
    """One Conv(3x3, pad=1) -> BatchNorm(eval) -> ReLU -> MaxPool(2) block."""
    channels, height, width = x.shape
    padded = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(1, 2))
    cols = np.transpose(windows, (1, 2, 0, 3, 4)).reshape(height * width, channels * 9)
    out = cols @ weight.reshape(weight.shape[0], -1).T
    out = out.reshape(height, width, -1).transpose(2, 0, 1)
    out = (out - mean[:, None, None]) / np.sqrt(var[:, None, None] + 1e-5)
    out = out * gamma[:, None, None] + beta[:, None, None]
    out = np.maximum(out, 0)
    return out.reshape(out.shape[0], out.shape[1] // 2, 2, out.shape[2] // 2, 2).max(axis=(2, 4))


def predict_image(image, weights):
    """Return a probability array over the 6 classes for a PIL image."""
    resample = getattr(Image, "Resampling", Image).BICUBIC
    arr = np.asarray(image.convert("RGB").resize((96, 96), resample), dtype=np.float32) / 255.0
    x = ((arr - 0.5) / 0.5).transpose(2, 0, 1)

    for i in range(3):
        x = conv_bn_relu_pool(
            x,
            weights[f"conv{i}_w"],
            weights[f"bn{i}_gamma"],
            weights[f"bn{i}_beta"],
            weights[f"bn{i}_mean"],
            weights[f"bn{i}_var"],
        )

    x = x.reshape(128, 4, 3, 4, 3).mean(axis=(2, 4)).reshape(-1)
    x = np.maximum(weights["fc1_w"] @ x + weights["fc1_b"], 0)
    logits = weights["fc2_w"] @ x + weights["fc2_b"]
    exp = np.exp(logits - logits.max())
    return exp / exp.sum()

st.title("Intel Image Classification — CNN Results")
st.caption(
    "6-class scene classification · PyTorch CNN trained from scratch · "
    "best validation checkpoint used for the held-out test set"
)

# ---------------------------------------------------------------------------
# Headline metrics
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Test accuracy",
    f"{results['test_acc'] * 100:.2f}%",
    help="Accuracy on the completely unseen seg_test split (3,000 images).",
)
col2.metric(
    "Validation accuracy",
    f"{results['final_val_acc'] * 100:.2f}%",
    help="Best validation accuracy achieved during training.",
)
col3.metric(
    "Training accuracy",
    f"{results['final_train_acc'] * 100:.2f}%",
    help="Accuracy on the training split using the best checkpoint.",
)
col4.metric(
    "Test loss",
    f"{results['test_loss']:.4f}",
    help="Cross-entropy loss on the test split.",
)

st.markdown("---")

tab_overview, tab_predict, tab_curves, tab_cm, tab_files = st.tabs(
    ["Overview", "Try an image", "Training curves", "Confusion matrix", "Model & files"]
)

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
with tab_overview:
    st.subheader("What was trained")
    st.write(
        "A convolutional neural network with **three Conv → BatchNorm → ReLU → MaxPool blocks**, "
        "adaptive average pooling, dropout, a 256-unit dense layer and a 6-way softmax output. "
        "The model was trained for **20 epochs** on 96×96 images with Adam, weight decay and "
        "step-wise learning-rate decay. Random flips and small translations were used for augmentation."
    )

    info_col1, info_col2 = st.columns(2)
    with info_col1:
        st.markdown("**Run configuration**")
        st.dataframe(
            pd.DataFrame(
                {
                    "Setting": [
                        "Dataset",
                        "Classes",
                        "Image size",
                        "Epochs",
                        "Batch size",
                        "Device",
                        "Trainable parameters",
                        "Training time",
                    ],
                    "Value": [
                        str(results["dataset"]),
                        str(len(classes)),
                        f"{results['image_size']}×{results['image_size']}",
                        str(results["epochs"]),
                        str(results["batch_size"]),
                        str(results["device"]),
                        f"{results['parameters']:,}",
                        f"{results['training_minutes']:.1f} min",
                    ],
                }
            ),
            hide_index=True,
            width="stretch",
        )
    with info_col2:
        st.markdown("**Images per split**")
        st.bar_chart(pd.Series(results["split_sizes"], name="images"))
        st.dataframe(
            pd.DataFrame(
                {
                    "Split": ["Train", "Validation", "Test"],
                    "Images": [
                        results["split_sizes"]["train"],
                        results["split_sizes"]["validation"],
                        results["split_sizes"]["test"],
                    ],
                }
            ),
            hide_index=True,
            width="stretch",
        )

    st.markdown("**Results and interpretation**")
    st.markdown(
        f"""
1. The CNN reached **{results['test_acc'] * 100:.2f}% test accuracy** after 20 epochs. Validation ({results['final_val_acc'] * 100:.2f}%) and test accuracy are very close, so the model generalizes rather than memorizing the training set.
2. The train–test gap is about {(results['final_train_acc'] - results['test_acc']) * 100:.1f} percentage points; the remaining errors are mostly between visually similar natural classes such as glacier/mountain and sea/glacier.
3. Accuracy could be improved with larger input resolution, stronger augmentation (random crops, color jitter, cutout), and a deeper residual architecture.
4. Transfer learning from an ImageNet-pretrained backbone and cosine learning-rate annealing with early stopping would likely add several points while reducing training time.
        """
    )

# ---------------------------------------------------------------------------
# Try an image
# ---------------------------------------------------------------------------
with tab_predict:
    st.subheader("Classify your own image")
    st.write(
        "Upload a JPG, JPEG or PNG image of a scene. It is resized to 96×96 and passed "
        "through the exact trained CNN above — the same preprocessing used during training."
    )

    uploaded = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png"], help=" buildings · forest · glacier · mountain · sea · street"
    )

    if uploaded is None:
        st.info("Upload an image above to see the model's prediction.")
    else:
        image = Image.open(uploaded)
        preview_col, result_col = st.columns([1, 2])

        with preview_col:
            st.image(image, caption=f"{uploaded.name} · {image.size[0]}×{image.size[1]}", width="stretch")

        with result_col:
            if st.button("Classify image", type="primary"):
                with st.spinner("Running the CNN…"):
                    probabilities = predict_image(image, load_weights())

                best = int(np.argmax(probabilities))
                st.success(f"**{classes[best]}** — {probabilities[best] * 100:.1f}% confidence")

                st.markdown("**Confidence for all six classes**")
                st.bar_chart(
                    pd.DataFrame({"Class": classes, "Confidence": probabilities}).set_index("Class")
                )

                st.dataframe(
                    pd.DataFrame(
                        {
                            "Class": classes,
                            "Confidence": [f"{p * 100:.2f}%" for p in probabilities],
                        }
                    ).sort_values("Confidence", ascending=False, key=lambda s: s.str.rstrip("%").astype(float)),
                    hide_index=True,
                    width="stretch",
                )
                st.caption(
                    "Tip: try a clear landscape photo similar to the training data. "
                    "The model was trained only on these six scene categories."
                )

# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------
with tab_curves:
    st.subheader("Accuracy and loss across epochs")
    history = results["history"]

    acc_df = pd.DataFrame(
        {
            "train": history["train_acc"],
            "validation": history["val_acc"],
        },
        index=epochs,
    )
    acc_df.index.name = "epoch"
    st.markdown("**Accuracy**")
    st.line_chart(acc_df, height=320)

    loss_df = pd.DataFrame(
        {
            "train": history["train_loss"],
            "validation": history["val_loss"],
        },
        index=epochs,
    )
    loss_df.index.name = "epoch"
    st.markdown("**Cross-entropy loss**")
    st.line_chart(loss_df, height=320)

    with st.expander("Show exported plot image"):
        st.image(BASE / "training_curves.png", caption="Exported training curves", width="stretch")

# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
with tab_cm:
    st.subheader("Confusion matrix on the test split")
    st.image(BASE / "confusion_matrix.png", caption="Rows = true class, columns = predicted class", width="stretch")

    per_class = cm.diagonal() / cm.sum(axis=1)
    class_df = pd.DataFrame(
        {
            "Class": classes,
            "Test images": cm.sum(axis=1),
            "Correct": cm.diagonal(),
            "Accuracy": [f"{v * 100:.1f}%" for v in per_class],
        }
    )
    st.dataframe(class_df, hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# Model and files
# ---------------------------------------------------------------------------
with tab_files:
    st.subheader("Model summary")
    st.code(
        """
Input 3x96x96
  Conv 3->32, BN, ReLU, MaxPool  -> 32x48x48
  Conv 32->64, BN, ReLU, MaxPool -> 64x24x24
  Conv 64->128, BN, ReLU, MaxPool-> 128x12x12
  AdaptiveAvgPool(4x4) -> Flatten -> Dropout(0.3)
  Dense 2048->256 -> ReLU -> Dropout(0.3)
  Dense 256->6 (softmax)
""",
        language="text",
    )
    st.write(f"Trainable parameters: **{results['parameters']:,}**")

    st.subheader("Download result files")
    st.download_button(
        "results.json",
        data=(BASE / "results.json").read_bytes(),
        file_name="results.json",
        mime="application/json",
    )
    st.download_button(
        "best_cnn_model.pth",
        data=(BASE / "best_cnn_model.pth").read_bytes(),
        file_name="best_cnn_model.pth",
        mime="application/octet-stream",
    )
    st.download_button(
        "training_curves.png",
        data=(BASE / "training_curves.png").read_bytes(),
        file_name="training_curves.png",
        mime="image/png",
    )
    st.download_button(
        "confusion_matrix.png",
        data=(BASE / "confusion_matrix.png").read_bytes(),
        file_name="confusion_matrix.png",
        mime="image/png",
    )

st.sidebar.title("About")
st.sidebar.info(
    "This dashboard reports the actual evaluation of a CNN trained on the Intel Image "
    "Classification dataset. Source results were produced by the Colab notebook and "
    "are displayed here without modification. Use the **Try an image** tab to classify "
    "your own photo with the same trained model."
)
st.sidebar.markdown(f"**Test accuracy:** `{results['test_acc'] * 100:.2f}%`")
