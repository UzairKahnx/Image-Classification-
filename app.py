import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

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

tab_overview, tab_curves, tab_cm, tab_files = st.tabs(
    ["Overview", "Training curves", "Confusion matrix", "Model & files"]
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
    "are displayed here without modification."
)
st.sidebar.markdown(f"**Test accuracy:** `{results['test_acc'] * 100:.2f}%`")
