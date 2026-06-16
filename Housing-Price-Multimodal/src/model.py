import torch
import torch.nn as nn
import torchvision.models as models

class HousingModel(nn.Module):
    def __init__(self):
        super(HousingModel, self).__init__()

        # CNN for images
        self.cnn = models.resnet18(weights=None)
        self.cnn.fc = nn.Linear(512, 128)

        # Tabular network
        self.tabular = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

        # Fusion layer
        self.regressor = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, image, tabular):
        img_feat = self.cnn(image)
        tab_feat = self.tabular(tabular)

        combined = torch.cat((img_feat, tab_feat), dim=1)
        output = self.regressor(combined)

        return output.squeeze()