from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import models
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from transformers import CLIPProcessor, CLIPModel


# MLP head to be added after the CLIP model
class MLPHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class ResNetWithMLP(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNetWithMLP, self).__init__()
        # Load a pretrained ResNet (e.g., ResNet18)
        self.resnet = models.resnet18(pretrained=True)
        
        # Freeze the ResNet layers if you don't want to train them
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Replace the last fully connected layer with a custom MLP head
        num_features = self.resnet.fc.in_features  # Get the number of features in the last layer
        self.resnet.fc = MLPHead(input_dim=num_features, output_dim=num_classes)

    def forward(self, x):
        return self.resnet(x)

# CLIP model with MLP head for binary classification
class CLIPWithMLP(nn.Module):
    def __init__(self, clip_model, mlp_head):
        super(CLIPWithMLP, self).__init__()
        self.clip_model = clip_model
        self.mlp_head = mlp_head

        # Freeze CLIP's parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, image):
        # Extract image features from CLIP
        image_features = self.clip_model.get_image_features(pixel_values=image)
        # Pass through the MLP head
        outputs = self.mlp_head(image_features)
        return outputs


