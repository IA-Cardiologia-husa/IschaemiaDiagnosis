import torch
import torch.nn as nn
from torch.utils.data import Dataset

# ----------------------------------------------------------
# 🧩 MODELO MULTIMODAL
# ----------------------------------------------------------
    
class SimpleMultimodalClassifier(nn.Module):
    def __init__(self, num_clinical_features, num_classes=2):
        super(SimpleMultimodalClassifier, self).__init__()
       
        self.image_features = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )
       
        self.clinical_net = nn.Sequential(
            nn.Linear(num_clinical_features, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
       
        self.classifier = nn.Sequential(
            nn.Linear(128 + 64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        self.img_norm = nn.BatchNorm1d(128)
        self.clinical_norm = nn.BatchNorm1d(64)
 
    def forward(self, x):
        images, clinical = x
        img_features = self.image_features(images)
        img_features = nn.AdaptiveAvgPool3d(1)(img_features).view(img_features.size(0), -1)
        img_features = self.img_norm(img_features)
        clinical_features = self.clinical_net(clinical)
        clinical_features = self.clinical_norm(clinical_features)
        combined = torch.cat([img_features, clinical_features], dim=1)
        return self.classifier(combined)