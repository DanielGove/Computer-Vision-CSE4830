import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from model_clf import MrModel, MrSpecialModel
from dataloader_clf import create_data_loaders


# Parameters
num_classes = 10
num_epochs = 20
batch_size = 32
learning_rate = 0.001

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset loading and splitting
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Model setup
model1 = MrModel(num_classes=num_classes).to(device)
model2 = MrSpecialModel(num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(model1.parameters(), lr=learning_rate)
optimizer2 = optim.Adam(model2.parameters(), lr=learning_rate)

# Training function
def train_and_evaluate(model, optimizer, train_loader, val_loader, num_epochs):
    model_metrics = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        model_metrics['train_loss'].append(avg_train_loss)

        # Validation
        model.eval()
        total = 0
        correct = 0
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        model_metrics['val_loss'].append(avg_val_loss)
        model_metrics['val_accuracy'].append(val_accuracy)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

    return model_metrics

# Train both models
train_loader, val_loader = create_data_loaders(batch_size=32)
metrics1 = train_and_evaluate(model1, optimizer1, train_loader, val_loader, num_epochs)
train_loader, val_loader = create_data_loaders(batch_size=16) # 16: Memory Issues
metrics2 = train_and_evaluate(model2, optimizer2, train_loader, val_loader, num_epochs)

import json
with open('model1_metrics.json', 'w') as f:
    json.dump(metrics1, f)
with open('model2_metrics.json', 'w') as f:
    json.dump(metrics2, f)
