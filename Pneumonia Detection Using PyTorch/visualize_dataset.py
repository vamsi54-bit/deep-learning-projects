# =============================================================
#         PNEUMONIA DETECTION - DATASET VISUALIZATION
# =============================================================


# ── CLASS NAMES ───────────────────────────────────────────────

class_names = train_dataset.classes
# Retrieves class labels auto-assigned by ImageFolder.
# Order matches subfolder names alphabetically:
#   ['NORMAL', 'PNEUMONIA'] → index 0 and 1

print(f"Classes: {class_names}")


# ── VISUALIZE FIRST 5 TRAINING IMAGES ────────────────────────

images, labels = next(iter(train_loader))
# iter(train_loader) → creates an iterator over the DataLoader.
# next() → fetches the very first batch (32 images, 32 labels).
# We only use the first 5 from that batch below.

plt.figure(figsize=(15, 5))
# Creates a wide canvas to fit 5 images side by side.

for i in range(5):

    image = images[i].permute(1, 2, 0).numpy()
    # images[i] shape is (C, H, W) — PyTorch tensor format.
    # permute(1, 2, 0) → converts to (H, W, C) for matplotlib.
    # .numpy() → converts tensor to NumPy array for plotting.

    image = image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    # Reverses the Normalize() applied during preprocessing.
    # Without this, colors look washed out / wrong.

    image = image.clip(0, 1)
    # Clamps pixel values to [0, 1] range.
    # Denormalization can push some values slightly out of range.

    plt.subplot(1, 5, i + 1)
    # Places each image in its own subplot slot (1 row, 5 cols).

    plt.imshow(image)
    # Renders the image in the subplot.

    plt.title(class_names[labels[i]])
    # Shows NORMAL or PNEUMONIA as the title above each image.

    plt.axis("off")
    # Hides x/y axis ticks — cleaner look for image grids.

plt.tight_layout()
# Automatically adjusts spacing so subplots don't overlap.

plt.show()
# Renders and displays the final figure.
