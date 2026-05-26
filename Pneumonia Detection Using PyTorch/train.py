# =============================================================
#         PNEUMONIA DETECTION - TRAINING FUNCTION
# =============================================================

def train_model(model, epochs=5):

    # ── SETUP ─────────────────────────────────────────────────

    best_acc = 0.0
    # Tracks the highest validation accuracy seen so far.
    # Used to decide when to save the model weights.

    best_model_wts = copy.deepcopy(model.state_dict())
    # Saves a deep copy of the initial weights.
    # state_dict() → dictionary of all layer weights & biases.
    # deepcopy → true independent copy, not just a reference.

    train_losses = []
    val_accuracies = []
    # Lists to store loss and accuracy per epoch for plotting later.


    # ── EPOCH LOOP ────────────────────────────────────────────

    for epoch in range(epochs):
        print(f"\nEpoch [{epoch+1}/{epochs}]")


        # ── TRAINING PHASE ────────────────────────────────────

        model.train()
        # Switches model to training mode.
        # Enables Dropout and BatchNorm to behave as during training.

        running_loss = 0.0
        # Accumulates total loss across all batches in the epoch.

        loop = tqdm(train_loader)
        # Wraps the DataLoader with a live progress bar in terminal.

        for images, labels in loop:

            images = images.to(device)
            labels = labels.to(device)
            # Moves batch to GPU/CPU — must match where model lives.

            optimizer.zero_grad()
            # Clears gradients from the previous batch.
            # PyTorch accumulates gradients by default, so this is
            # required at the start of every batch.

            outputs = model(images)
            # Forward pass — runs images through the network.
            # outputs shape: (batch_size, 2) → raw scores for each class.

            loss = criterion(outputs, labels)
            # Computes CrossEntropyLoss between predictions and true labels.
            # Higher loss = model is more wrong.

            loss.backward()
            # Backward pass — computes gradients of loss
            # with respect to all trainable parameters (only model.fc here).

            optimizer.step()
            # Updates model.fc weights using the computed gradients.
            # One step of gradient descent via Adam optimizer.

            running_loss += loss.item()
            # .item() extracts the scalar loss value from the tensor.
            # Accumulates it to compute average loss at epoch end.

            loop.set_postfix(loss=loss.item())
            # Displays current batch loss on the tqdm progress bar.


        # ── EPOCH LOSS ────────────────────────────────────────

        epoch_loss = running_loss / len(train_loader)
        # Averages loss over all batches in the epoch.

        train_losses.append(epoch_loss)
        # Saves epoch loss for plotting the loss curve later.

        print(f"Train Loss: {epoch_loss:.4f}")
        # Prints rounded loss for the completed epoch.
