# =========================================
# FUNCTION TO DISPLAY MODEL PREDICTIONS
# =========================================

def show_predictions(model, loader):

    # Set model to evaluation mode
    # Disables dropout and batch normalization updates
    model.eval()

    # Get one batch of images and labels from loader
    images, labels = next(iter(loader))

    # Move images to GPU/CPU
    images_device = images.to(device)

    # Disable gradient calculation for inference
    with torch.no_grad():

        # Forward pass
        outputs = model(images_device)

    # Get predicted class index
    # torch.max returns:
    # values -> probabilities
    # indices -> predicted class
    _, predicted = torch.max(outputs, 1)

    # Create figure for displaying images
    plt.figure(figsize=(15, 5))

    # Loop through first 5 images
    for i in range(5):

        # Convert image from:
        # [C, H, W] -> [H, W, C]
        image = images[i].permute(1, 2, 0).numpy()

        # =========================================
        # UNNORMALIZE IMAGE
        # =========================================
        # Reverse normalization for proper display
        image = image * [0.229, 0.224, 0.225] + \
                        [0.485, 0.456, 0.406]

        # Keep pixel values between 0 and 1
        image = image.clip(0, 1)

        # Create subplot
        plt.subplot(1, 5, i + 1)

        # Display image
        plt.imshow(image)

        # Get actual and predicted labels
        true_label = class_names[labels[i]]
        pred_label = class_names[predicted[i]]

        # Set image title
        plt.title(
            f"True: {true_label}\nPred: {pred_label}"
        )

        # Hide axis
        plt.axis("off")

    # Adjust layout
    plt.tight_layout()

    # Display figure
    plt.show()


# =========================================
# DISPLAY FINAL PREDICTIONS
# =========================================

show_predictions(model, test_loader)
