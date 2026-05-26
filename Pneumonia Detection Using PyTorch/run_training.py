# =============================================================
#         PNEUMONIA DETECTION - RUN TRAINING & SAVE MODEL
# =============================================================


# ── TRAIN MODEL ───────────────────────────────────────────────

model, train_losses, val_accuracies = train_model(model, epochs=5)
# Calls the train_model() function defined earlier.
# epochs=5 → full pass over the training data 5 times.
#
# Returns 3 things:
#   model          → best version of the model (highest val accuracy)
#   train_losses   → list of average loss per epoch
#   val_accuracies → list of validation accuracy per epoch


# ── SAVE MODEL ────────────────────────────────────────────────

torch.save(
    model.state_dict(),       # Only saves weights, not the full model class
    "pneumonia_resnet18.pth"  # Output file name (.pth is PyTorch convention)
)
# Saves the trained model weights to disk.
# state_dict() → lightweight dictionary of all layer parameters.
# Preferred over saving the full model — more portable and flexible.
#
# To reload later:
#   model = models.resnet18()
#   model.fc = nn.Linear(512, 2)
#   model.load_state_dict(torch.load("pneumonia_resnet18.pth"))

print("\nModel Saved Successfully!")
