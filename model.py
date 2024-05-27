from torch import nn
import torch.nn.functional as F

from RI_CNN import *

# Standard model with conventional convolutional layers
class MrEngineerMan(nn.Module):
    def __init__(self, img_height=384, img_width=512, num_boxes=67, num_classes=11):
        super(MrEngineerMan, self).__init__()
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.img_height = img_height
        self.img_width = img_width

        # Standard Convolutional Layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # This will reduce the dimension by half each time it's applied

        # For better gradients
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.batch_norm3 = nn.BatchNorm2d(64)

        # Calculate the output size after convolutions and pooling
        out_size = self.calculate_conv_output_size()

        # Boundary Box Prediction
        self.fc1_bbox = nn.Linear(out_size, 256)
        self.fc2_bbox = nn.Linear(256, num_boxes * 4)

        # Class Prediction
        self.fc1_class = nn.Linear(out_size, 256)
        self.fc2_class = nn.Linear(256, num_boxes * num_classes)

        # Confidence score prediction
        self.fc1_conf = nn.Linear(out_size, 256)
        self.fc2_conf = nn.Linear(256, num_boxes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.pool(x))  # Apply pooling and activation
        x = self.batch_norm1(x)
        x = self.conv2(x)
        x = F.relu(self.pool(x))  # Apply second pooling and activation
        x = self.batch_norm2(x)
        x = self.conv3(x)
        x = F.relu(self.pool(x))  # Third pooling and activation
        x = self.batch_norm3(x)

        x = x.view(x.size(0), -1)  # Flatten the output for dense layers

        bbox = F.relu(self.fc1_bbox(x))
        cls = F.relu(self.fc1_class(x))
        conf = F.relu(self.fc1_conf(x))
        
        bbox = F.relu(self.fc2_bbox(bbox)).view(8, self.num_boxes, -1)
        cls = F.relu(self.fc2_class(cls)).view(8, self.num_boxes, -1)
        conf = F.relu(self.fc2_conf(conf)).unsqueeze(-1)

        cls = F.softmax(cls)
        conf = F.sigmoid(conf)
        
        return torch.cat((bbox, conf, cls), dim=-1)

    def calculate_conv_output_size(self):
        size = (self.img_height // 2 // 2 // 2, self.img_width // 2 // 2 // 2)
        return size[0] * size[1] * 64
    
# Special model with Vector based convolutional layers
class SpecialEngineerMan(nn.Module):
    def __init__(self, img_height=384, img_width=512, num_boxes=67, num_classes=10):
        super(SpecialEngineerMan, self).__init__()
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.img_height = img_height
        self.img_width = img_width

        # Standard Convolutional Layers
        self.conv1 = VectorTransformConv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = VectorConv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = VectorConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = VectorMaxPool2d(2, 2)  # This will reduce the dimension by half each time it's applied

        # For better gradients
        self.batch_norm1 = VectorBatchNorm2d(16)
        self.batch_norm2 = VectorBatchNorm2d(32)
        self.batch_norm3 = nn.BatchNorm2d(64)

        # Calculate the output size after convolutions and pooling
        out_size = self.calculate_conv_output_size()

        # Dense Layers
        self.fc1 = nn.Linear(out_size, 512)
        self.fc2 = nn.Linear(512, num_boxes * (4 + 1 + num_classes))  # 4 for bbox, 1 for confidence, num_classes for class probs

    def forward(self, x):
        x = self.conv1(x)
        x = vector_relu(self.pool(x))  # Apply pooling and activation
        x = self.batch_norm1(x)
        x = self.conv2(x)
        x = vector_relu(self.pool(x))  # Apply second pooling and activation
        x = self.batch_norm2(x)
        x = self.conv3(x)
        x = vector_relu(self.pool(x))  # Third pooling and activation

        # Drop the phase and normalize
        x = x[..., 0]
        x = self.batch_norm3(x)

        # The decision head
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x.view(-1, self.num_boxes, 5 + self.num_classes)  # Reshape for output format

    def calculate_conv_output_size(self):
        size = (self.img_height // 2 // 2 // 2, self.img_width // 2 // 2 // 2)
        return size[0] * size[1] * 64