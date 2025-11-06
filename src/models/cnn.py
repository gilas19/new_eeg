import torch
import torch.nn as nn


class EEG_CNN(nn.Module):
    def __init__(self, n_channels, n_timepoints, n_classes=2, dropout=0.3):
        super(EEG_CNN, self).__init__()

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=25, padding=12),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=15, padding=7),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            nn.Conv1d(128, 256, kernel_size=10, padding=5),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, n_channels, n_timepoints)
            conv_output = self.temporal_conv(dummy_input)
            flattened_size = conv_output.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.temporal_conv(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
