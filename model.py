import torch.nn as nn
import torch

class CNN_GRU(nn.Module):
    def __init__(self, input_size, output_size, units):
        super(CNN_GRU, self).__init__()

        self.conv1d = nn.Conv1d(
            in_channels=input_size,
            out_channels=output_size,
            kernel_size=3,
            stride=1,
            padding=0
        )

        self.gru = nn.GRU(
            input_size=output_size,
            hidden_size=units,
            num_layers=1,
            dropout=0.2,
            batch_first=True
        )

        self.relu = nn.ReLU()

        self.linear = nn.Linear(
            in_features=units,
            out_features=4  # 클래스 수에 맞게 조절
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, features, seq_len)
        x = self.conv1d(x)
        x = self.relu(x)

        x = x.permute(0, 2, 1)  # (batch, seq_len, features)
        out, h_n = self.gru(x)

        x = self.linear(out[:, -1, :])
        x = self.softmax(x)
        return x
