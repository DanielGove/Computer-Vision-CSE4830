from torch import nn

class MrEngineerMan(nn.Module):
    def __init__(self, num_boxes=67, num_classes=10):
        super(MrEngineerMan, self).__init__()
        self.num_boxes = num_boxes
        self.num_classes = num_classes

        # Standard Convolutional Layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Dense Layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, num_boxes * (4 + 1 + num_classes))  # 4 for bbox, 1 for confidence, num_classes for class probs
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)
        return x.view(-1, self.num_boxes, 5 + self.num_classes)