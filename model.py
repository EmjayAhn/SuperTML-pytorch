import torch.nn as nn
from torchvision import models

model_res = models.resnet18(pretrained=True)
num_features = model_res.fc.in_features
model_res.fc = nn.Linear(num_features, 3)
