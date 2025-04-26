import torch.nn as nn
import torch

class CNN_BiGRU(nn.Module):
    def __init__(self, input_size, output_size, units, num_classes):
        super().__init__()
        self.conv1d = nn.Conv1d(input_size, output_size, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        self.bigru = nn.GRU(
            input_size=output_size,
            hidden_size=units,
            batch_first=True,
            bidirectional=True
        )

        self.linear = nn.Linear(units * 2, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = x.permute(0, 2, 1)
        out, _ = self.bigru(x)
        x = self.linear(out[:, -1, :])
        return self.softmax(x)
