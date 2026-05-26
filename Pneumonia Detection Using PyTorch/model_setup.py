# =============================================================
#         PNEUMONIA DETECTION - MODEL SETUP
# =============================================================


# ── LOAD PRETRAINED RESNET18 ──────────────────────────────────

model = models.resnet18(weights="DEFAULT")
# Loads ResNet18 pretrained on ImageNet (1.2M images, 1000 classes).
# Comes with strong feature extraction ability out of the box —
# edges, textures, shapes — all useful for X-ray classification too.


# ── FREEZE PRETRAINED LAYERS ──────────────────────────────────

for param in model.parameters():
    param.requires_grad = False
# Freezes all layers → their weights won't update during training.
# We only want to train the new final layer, not the whole network.
# This saves time, prevents overfitting, and preserves ImageNet knowledge.


# ── REPLACE FINAL LAYER ───────────────────────────────────────

model.fc = nn.Linear(512, 2)
# ResNet18's original final layer outputs 1000 classes (ImageNet).
# We replace it with a new Linear layer: 512 inputs → 2 outputs.
#   Output 0 → NORMAL
#   Output 1 → PNEUMONIA
# Only this layer has requires_grad=True (unfrozen by default).

model = model.to(device)
# Moves the entire model to GPU (if available) or CPU.
# Must match the device tensors are on during training.

print("\nResNet18 Loaded Successfully!")


# ── LOSS FUNCTION ─────────────────────────────────────────────

criterion = nn.CrossEntropyLoss()
# Standard loss for multi-class classification.
# Internally applies Softmax + Log + NLLLoss in one step.
# Penalizes the model more when it's confidently wrong.


# ── OPTIMIZER ─────────────────────────────────────────────────

optimizer = optim.Adam(
    model.fc.parameters(),  # Only optimize the final layer's weights
    lr=0.001                # Learning rate — how big each weight update step is
)
# Adam adapts the learning rate per parameter automatically.
# We pass model.fc.parameters() instead of model.parameters()
# because all other layers are frozen — no point optimizing them.
