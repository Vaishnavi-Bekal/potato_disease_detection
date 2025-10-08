import torch
import torch.nn as nn
import torchvision

class FeatureBackbone(nn.Module):
    def __init__(self, model_name='vgg16', pretrained=True):
        super().__init__()
        if model_name == 'vgg16':
            model = torchvision.models.vgg16(pretrained=pretrained)
            self.features = model.features
            self.out_dim = 512
        elif model_name == 'mobilenet_v2':
            model = torchvision.models.mobilenet_v2(pretrained=pretrained)
            self.features = model.features
            self.out_dim = 1280
        else:
            raise ValueError("Unsupported model")

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)

class CombinedModel(nn.Module):
    def __init__(self, n_classes=3, pretrained=True):
        super().__init__()
        self.vgg = FeatureBackbone('vgg16', pretrained)
        self.mob = FeatureBackbone('mobilenet_v2', pretrained)
        total_dim = self.vgg.out_dim + self.mob.out_dim

        self.head = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, n_classes)
        )

    def forward(self, x):
        v = self.vgg(x)
        m = self.mob(x)
        x = torch.cat([v, m], dim=1)
        return self.head(x)
