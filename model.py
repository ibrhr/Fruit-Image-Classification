import torch
import torch.nn as nn
import torchvision

class FruitClassifier(nn.Module):
    """ResNet-50 backbone (no FC) + Dropout→Linear head for 196 fruit classes."""
    
    def __init__(self, num_classes :int = 196, pretrained=True):
        super().__init__()
        backbone = torchvision.models.resnet50(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(backbone.fc.in_features, num_classes)
        )
        
        
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x
    
    
    