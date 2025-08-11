import torch
import torch.nn as nn
from torchvision.models import resnet18

class Model(nn.Module):
    def __init__(self, n_classes):
        super(Model, self).__init__()

        resnet_model = resnet18(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(resnet_model.children())[:-1])  # Remove the last fully connected layer
        for param in self.backbone.parameters():
            param.requires_grad = False  # Freeze the backbone

        self.fc = nn.Linear(resnet_model.fc.in_features, n_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)  # Flatten the output from the backbone
        x = self.fc(x)
        return x