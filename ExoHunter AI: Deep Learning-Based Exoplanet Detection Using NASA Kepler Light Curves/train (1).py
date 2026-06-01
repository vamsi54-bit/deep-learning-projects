"""
Light Curve Classifier — Training Loop
=======================================
Defines a reusable `train_model()` function that:
  - Trains for N epochs with tqdm progress bars
  - Evaluates on the test set each epoch
  - Returns full loss / accuracy history for plotting
  - Tracks best model weights (lowest test loss)

Dependencies: torch, tqdm
"""

from tqdm import tqdm
import torch
import copy


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_model(
    model,
    optimizer,
    criterion,
    train_loader,
    test_loader,
    epochs=20,
    save_best=True,         # keep a copy of weights with lowest test loss
):
    """
    Train a binary classifier and evaluate each epoch.

    Parameters
    ----------
    model        : nn.Module        Model to train (mutated in-place)
    optimizer    : torch.optim.*    Optimizer bound to model parameters
    criterion    : nn.Module        Loss function (BCELoss expected)
    train_loader : DataLoader       Training batches
    test_loader  : DataLoader       Evaluation batches
    epochs       : int              Number of full passes over the data
    save_best    : bool             If True, restore best weights after training

    Returns
    -------
    history : dict
        {
          "train_loss" : [float, ...],   # avg train loss per epoch
          "test_loss"  : [float, ...],   # avg test  loss per epoch
          "accuracy"   : [float, ...],   # test accuracy (%) per epoch
        }
    """

    history = {"train_loss": [], "test_loss": [], "accuracy": []}
    best_test_loss = float("inf")
    best_weights   = None

    for epoch in range(epochs):

        # ------------------------------------------------------------------
        # Training phase
        # ------------------------------------------------------------------
        model.train()
        train_loss = 0.0

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1:>2}/{epochs} [TRAIN]",
            leave=False,
        )

        for X_batch, y_batch in train_bar:
            optimizer.zero_grad()

            outputs = model(X_batch)           # forward pass
            loss    = criterion(outputs, y_batch)

            loss.backward()                    # compute gradients
            optimizer.step()                   # update weights

            train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # ------------------------------------------------------------------
        # Evaluation phase
        # ------------------------------------------------------------------
        model.eval()
        test_loss = 0.0
        correct   = 0
        total     = 0

        test_bar = tqdm(
            test_loader,
            desc=f"Epoch {epoch+1:>2}/{epochs} [TEST ]",
            leave=False,
        )

        with torch.no_grad():
            for X_batch, y_batch in test_bar:
                outputs = model(X_batch)
                loss    = criterion(outputs, y_batch)

                test_loss += loss.item()

                preds    = (outputs > 0.5).float()     # threshold → binary
                correct += (preds == y_batch).sum().item()
                total   += y_batch.size(0)

                test_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_test_loss = test_loss / len(test_loader)
        accuracy      = 100.0 * correct / total

        # ------------------------------------------------------------------
        # History + best-weights checkpoint
        # ------------------------------------------------------------------
        history["train_loss"].append(avg_train_loss)
        history["test_loss"].append(avg_test_loss)
        history["accuracy"].append(accuracy)

        if save_best and avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_weights   = copy.deepcopy(model.state_dict())

        print(
            f"Epoch [{epoch+1:>2}/{epochs}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Test Loss: {avg_test_loss:.4f} | "
            f"Accuracy: {accuracy:.2f}%"
        )

    # Restore best weights so `model` leaves this function in its best state
    if save_best and best_weights is not None:
        model.load_state_dict(best_weights)
        print(f"\nBest weights restored  (test loss: {best_test_loss:.4f})")

    return history


# ---------------------------------------------------------------------------
# Run training for both models
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    print("=" * 55)
    print("Training  :  LSTM")
    print("=" * 55)
    lstm_history = train_model(
        model=lstm_model,
        optimizer=lstm_optimizer,
        criterion=criterion,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=20,
    )

    print("\n" + "=" * 55)
    print("Training  :  BiLSTM")
    print("=" * 55)
    bilstm_history = train_model(
        model=bilstm_model,
        optimizer=bilstm_optimizer,
        criterion=criterion,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=20,
    )

    # ------------------------------------------------------------------
    # Quick loss / accuracy summary plot
    # ------------------------------------------------------------------
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training Summary", fontsize=14)

    for ax, history, name in zip(
        axes,
        [lstm_history, bilstm_history],
        ["LSTM", "BiLSTM"],
    ):
        epochs_range = range(1, len(history["train_loss"]) + 1)
        ax2 = ax.twinx()

        ax.plot(epochs_range, history["train_loss"], label="Train Loss", color="steelblue")
        ax.plot(epochs_range, history["test_loss"],  label="Test Loss",  color="orange")
        ax2.plot(epochs_range, history["accuracy"],  label="Accuracy %", color="green", linestyle="--")

        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax2.set_ylabel("Accuracy (%)")

        lines_1, labels_1 = ax.legend_handles, [t.get_label() for t in ax.get_lines()]
        lines_2, labels_2 = ax2.legend_handles, [t.get_label() for t in ax2.get_lines()]
        ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    plt.tight_layout()
    plt.savefig("training_summary.png", dpi=150)
    plt.show()
    print("Plot saved → training_summary.png")
