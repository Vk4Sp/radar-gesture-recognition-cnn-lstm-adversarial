import torch
import torch.nn as nn


class SoliModel(nn.Module):
    def __init__(self, num_classes=11):
        super(SoliModel, self).__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # (32x32)
            nn.ReLU(),
            nn.MaxPool2d(2),  # (16x16)

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (8x8)
        )

        # flatten size = 32 * 8 * 8 = 2048
        self.feature_dim = 32 * 8 * 8

        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.feature_dim +1,
            hidden_size=128,
            batch_first=True
        )

        # Classifier
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x,spread):
        # x: (batch, T, 1, 32, 32)
        batch_size, T, C, H, W = x.shape

        # reshape for CNN
        x = x.view(batch_size * T, C, H, W)

        x = self.cnn(x)

        # flatten
        x = x.view(batch_size, T, -1)
        x = torch.cat([x, spread], dim=2)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # take last time step
        out = lstm_out[:, -1, :]

        # classification
        out = self.fc(out)

        return out