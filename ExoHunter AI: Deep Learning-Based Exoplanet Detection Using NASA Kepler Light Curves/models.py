"""
Light Curve Classifier — Model Definitions
===========================================
Two sequence models for binary transit classification:

  1. LSTMmodel    — unidirectional 2-layer LSTM
  2. biLSTMmodel  — bidirectional 2-layer LSTM  (doubles hidden state → 128-d FC input)

Input shape  : (batch, seq_len, 1)   ← single flux feature per time-step
Output shape : (batch, 1)            ← sigmoid probability of transit
"""

import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Unidirectional LSTM
# ---------------------------------------------------------------------------
class LSTMmodel(nn.Module):
    """
    Unidirectional 2-layer LSTM → fully-connected head.

    Architecture
    ------------
    LSTM  : input(1) → hidden(64), 2 layers, dropout 0.3
    FC    : 64 → 32 (ReLU, Dropout 0.3) → 1 (Sigmoid)
    """

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            batch_first=True,   # (batch, seq, feature)
            dropout=0.3,
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid(),       # binary output ∈ (0, 1)
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor  shape (batch, seq_len, 1)

        Returns
        -------
        Tensor  shape (batch, 1)
        """
        out, _ = self.lstm(x)       # (batch, seq_len, 64)
        out = out[:, -1, :]         # last time-step  → (batch, 64)
        out = self.fc(out)          # classifier head → (batch, 1)
        return out


# ---------------------------------------------------------------------------
# 2. Bidirectional LSTM
# ---------------------------------------------------------------------------
class biLSTMmodel(nn.Module):
    """
    Bidirectional 2-layer LSTM → fully-connected head.

    Because forward + backward passes are concatenated, the effective
    hidden dimension doubles: 64 × 2 = 128.

    Architecture
    ------------
    BiLSTM : input(1) → hidden(64 × 2 = 128), 2 layers, dropout 0.3
    FC     : 128 → 32 (ReLU, Dropout 0.3) → 1 (Sigmoid)
    """

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,     # forward + backward → 128-d output
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 32),     # 64 * 2 directions = 128
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor  shape (batch, seq_len, 1)

        Returns
        -------
        Tensor  shape (batch, 1)
        """
        out, _ = self.lstm(x)       # (batch, seq_len, 128)
        out = out[:, -1, :]         # last time-step  → (batch, 128)
        out = self.fc(out)          # classifier head → (batch, 1)
        return out


# ---------------------------------------------------------------------------
# Instantiation + quick check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import torch

    lstm_model   = LSTMmodel()
    bilstm_model = biLSTMmodel()

    print("LSTM model created")
    print(lstm_model)
    print()
    print("BiLSTM model created")
    print(bilstm_model)

    # Parameter counts
    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nLSTM   trainable params : {count_params(lstm_model):,}")
    print(f"BiLSTM trainable params : {count_params(bilstm_model):,}")

    # Dummy forward pass — batch=4, seq_len=200, features=1
    dummy = torch.randn(4, 200, 1)
    print(f"\nDummy input  : {dummy.shape}")
    print(f"LSTM   output: {lstm_model(dummy).shape}")
    print(f"BiLSTM output: {bilstm_model(dummy).shape}")
