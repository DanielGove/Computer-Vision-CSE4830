from torch import nn
import torch.nn.functional as F

from RI_CNN import *

class MrModel(nn.Module):
    def __init__(self, num_classes=10):
        super(MrModel, self).__init__()
        # Define the convolutional block
        self.features = nn.Sequential(
            self.conv_block(3, 32),
            self.conv_block(32, 64),
            self.conv_block(64, 128),
            self.conv_block(128, 256),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Define the classifier block
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 10)
        )

    def conv_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    

class MrSpecialModel(nn.Module):
    def __init__(self, num_classes=10):
        super(MrSpecialModel, self).__init__()
        # Define the convolutional block
        self.features = nn.Sequential(
            self.vector_transformation_block(3, 16),
            self.vector_conv_block(16, 32),
            self.vector_conv_block(32, 64),
            self.vector_conv_block(64, 128),
            Vector2Magnitude(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Define the classifier block
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 10)
        )

    def vector_transformation_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            VectorTransformConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            VectorBatchNorm2d(out_channels),
            VectorRelu(),
            VectorMaxPool2d(2, 1)
        )

    def vector_conv_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            VectorConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            VectorBatchNorm2d(out_channels),
            VectorRelu(),
            VectorMaxPool2d(2, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x