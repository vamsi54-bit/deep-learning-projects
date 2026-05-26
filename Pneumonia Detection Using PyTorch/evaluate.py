# =============================================================
#         PNEUMONIA DETECTION - MODEL EVALUATION (TEST SET)
# =============================================================


model.eval()
# Switches to evaluation mode — disables Dropout & BatchNorm updates.

all_labels = []
all_predictions = []
# Collect true labels and predicted labels across all batches.
# Needed later for confusion matrix and classification report.

correct = 0
total = 0
# Counters for accuracy calculation.


with torch.no_grad():
    # No gradients needed — we're only doing inference, not training.

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        # Forward pass → raw scores (logits) for NORMAL and PNEUMONIA.

        _, predicted = torch.max(outputs, 1)
        # Picks the class index with the highest score per image.
        # 0 = NORMAL | 1 = PNEUMONIA

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # Counts correct predictions in this batch.

        all_labels.extend(labels.cpu().numpy())
        # .cpu() → moves tensor from GPU back to CPU.
        # .numpy() → converts to NumPy array.
        # .extend() → appends batch results to the full list.

        all_predictions.extend(predicted.cpu().numpy())
        # Same process for predictions.
        # After the loop, all_labels and all_predictions hold
        # results for the entire test set — used for metrics next.


# ── TEST ACCURACY ─────────────────────────────────────────────

test_accuracy = 100 * correct / total
# Overall percentage of correctly classified test images.

print(f"\nTest Accuracy: {test_accuracy:.2f}%")
