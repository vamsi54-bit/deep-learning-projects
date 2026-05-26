# =============================================================
#         PNEUMONIA DETECTION - PLOT TRAINING METRICS
# =============================================================


# ── TRAINING LOSS CURVE ───────────────────────────────────────

plt.figure(figsize=(8, 5))
# Creates a new figure — 8 inches wide, 5 inches tall.

plt.plot(train_losses)
# Plots loss value for each epoch as a line graph.
# train_losses = list returned by train_model().
# A good model shows loss steadily decreasing over epochs.

plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
# Renders and displays the plot.


# ── VALIDATION ACCURACY CURVE ─────────────────────────────────

plt.figure(figsize=(8, 5))

plt.plot(val_accuracies)
# Plots validation accuracy for each epoch.
# val_accuracies = list returned by train_model().
# A good model shows accuracy steadily increasing over epochs.
# If accuracy drops while loss keeps falling → overfitting.

plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()
