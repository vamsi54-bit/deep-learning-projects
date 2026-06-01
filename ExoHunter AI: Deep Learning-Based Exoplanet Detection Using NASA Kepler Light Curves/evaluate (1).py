"""
Light Curve Classifier — Evaluation, Comparison & Model Saving
===============================================================
Sections
--------
  1. evaluate_model()  — metrics + confusion matrix per model
  2. Evaluate both LSTM and BiLSTM on the test set
  3. Side-by-side comparison bar chart (accuracy + F1)
  4. Save model weights to .pth files
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import torch
import numpy as np


# ---------------------------------------------------------------------------
# 1. Evaluation function
# ---------------------------------------------------------------------------

def evaluate_model(model, test_loader, model_name):
    """
    Run inference on test_loader and report binary classification metrics.

    Parameters
    ----------
    model       : nn.Module     Trained model (eval mode set internally)
    test_loader : DataLoader    Test batches
    model_name  : str           Label used in printed output

    Returns
    -------
    metrics : dict
        { "accuracy", "precision", "recall", "f1" }  — all floats in [0, 1]
    """

    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            preds   = (outputs > 0.5).float()       # threshold → {0, 1}

            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Flatten in case of shape (N, 1)
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()

    # ── Metrics ──────────────────────────────────────────────────────────
    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    cm        = confusion_matrix(y_true, y_pred)

    # ── Console output ───────────────────────────────────────────────────
    separator = "=" * 40
    print(f"\n{separator}")
    print(f"  {model_name}")
    print(separator)
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  {cm}")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    return {
        "accuracy"  : accuracy,
        "precision" : precision,
        "recall"    : recall,
        "f1"        : f1,
        "cm"        : cm,
        "y_true"    : y_true,
        "y_pred"    : y_pred,
    }


# ---------------------------------------------------------------------------
# 2. Evaluate both models
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    lstm_metrics   = evaluate_model(lstm_model,   test_loader, "LSTM")
    bilstm_metrics = evaluate_model(bilstm_model, test_loader, "BiLSTM")


    # -----------------------------------------------------------------------
    # 3. Comparison plots
    # -----------------------------------------------------------------------
    model_names  = ["LSTM", "BiLSTM"]
    metric_names = ["accuracy", "precision", "recall", "f1"]
    colors       = ["steelblue", "darkorange"]
    x            = np.arange(len(metric_names))
    bar_width    = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("LSTM vs BiLSTM — Test Set Evaluation", fontsize=14, fontweight="bold")

    # ── Left: grouped bar chart (all 4 metrics) ──────────────────────────
    ax = axes[0]
    lstm_vals   = [lstm_metrics[m]   for m in metric_names]
    bilstm_vals = [bilstm_metrics[m] for m in metric_names]

    bars1 = ax.bar(x - bar_width / 2, lstm_vals,   bar_width, label="LSTM",   color=colors[0], alpha=0.85)
    bars2 = ax.bar(x + bar_width / 2, bilstm_vals, bar_width, label="BiLSTM", color=colors[1], alpha=0.85)

    # Value labels on top of each bar
    for bar in bars1 + bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metric_names])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Metrics Comparison")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # ── Right: confusion matrices side by side ───────────────────────────
    ax = axes[1]
    ax.axis("off")

    for i, (metrics, name) in enumerate(
        zip([lstm_metrics, bilstm_metrics], model_names)
    ):
        sub_ax = fig.add_axes([0.55 + i * 0.22, 0.15, 0.19, 0.65])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=metrics["cm"],
            display_labels=["No Transit", "Transit"],
        )
        disp.plot(ax=sub_ax, colorbar=False, cmap="Blues")
        sub_ax.set_title(name, fontsize=10)
        sub_ax.tick_params(labelsize=7)
        sub_ax.set_xlabel("Predicted", fontsize=8)
        sub_ax.set_ylabel("True", fontsize=8)

    plt.savefig("evaluation_summary.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Plot saved → evaluation_summary.png")


    # -----------------------------------------------------------------------
    # 4. Save model weights
    # -----------------------------------------------------------------------
    torch.save(lstm_model.state_dict(),   "lstm_model.pth")
    torch.save(bilstm_model.state_dict(), "bilstm_model.pth")

    print("\nModels saved:")
    print("  lstm_model.pth")
    print("  bilstm_model.pth")

    # ── Winner summary ───────────────────────────────────────────────────
    winner = "LSTM" if lstm_metrics["f1"] >= bilstm_metrics["f1"] else "BiLSTM"
    print(f"\nBest F1 → {winner}  "
          f"(LSTM: {lstm_metrics['f1']:.4f} | BiLSTM: {bilstm_metrics['f1']:.4f})")
