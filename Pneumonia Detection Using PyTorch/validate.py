# =============================================================
#         PNEUMONIA DETECTION - VALIDATION PHASE
# =============================================================
#   (This is the continuation inside train_model() function)
# =============================================================


        # ── VALIDATION PHASE ──────────────────────────────────

        model.eval()
        # Switches model to evaluation mode.
        # Disables Dropout and freezes BatchNorm stats —
        # ensures consistent, reproducible predictions.

        correct = 0
        total = 0
        # Counters to track correct predictions and total samples.

        with torch.no_grad():
            # Disables gradient computation for the entire val loop.
            # We're not updating weights here, so no gradients needed.
            # Saves memory and speeds up inference.

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)
                # Move batch to GPU/CPU — same as in training.

                outputs = model(images)
                # Forward pass — gets raw scores (logits) for each class.
                # Shape: (batch_size, 2)

                _, predicted = torch.max(outputs, 1)
                # torch.max returns (values, indices) along dimension 1.
                # _ ignores the actual score values.
                # predicted → index of the highest score per image.
                #   0 = NORMAL, 1 = PNEUMONIA

                total += labels.size(0)
                # labels.size(0) = number of samples in current batch.
                # Adds to total count across all batches.

                correct += (predicted == labels).sum().item()
                # Compares predicted class vs true label element-wise.
                # .sum() → counts number of correct matches in the batch.
                # .item() → converts tensor scalar to Python int.


        # ── ACCURACY CALCULATION ──────────────────────────────

        accuracy = 100 * correct / total
        # Computes overall validation accuracy as a percentage.

        val_accuracies.append(accuracy)
        # Saves accuracy for this epoch to plot the accuracy curve later.

        print(f"Validation Accuracy: {accuracy:.2f}%")


        # ── SAVE BEST MODEL ───────────────────────────────────

        if accuracy > best_acc:
            best_acc = accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
            # If this epoch's accuracy beats the previous best,
            # save a deep copy of current weights.
            # This ensures we keep the best version, even if
            # accuracy drops in later epochs (overfitting).


    # ── AFTER ALL EPOCHS ──────────────────────────────────────

    print(f"\nBest Validation Accuracy: {best_acc:.2f}%")

    model.load_state_dict(best_model_wts)
    # Restores the model to its best performing weights.
    # Discards any degradation from later epochs.

    return model, train_losses, val_accuracies
    # Returns:
    #   model          → best version of the trained model
    #   train_losses   → loss per epoch (for loss curve plot)
    #   val_accuracies → accuracy per epoch (for accuracy curve plot)
