import torch
import torch.optim as optim
from model_clf import MrModel, MrSpecialModel
from dataloader_clf import create_data_loaders

# Hyperparameters
num_epochs = 10
learning_rate = 0.001

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DataLoaders
train_loader, val_loader = create_data_loaders(batch_size=16)

# Initialize the model
# model = MrModel().to(device)
model = MrSpecialModel().to(device)

# Loss and Optimizer
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def train():
    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            if (i+1) % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():,.4f}')

        average_loss = total_loss / len(train_loader)
        print(f"Ending Epoch {epoch+1} with Average Loss {average_loss:.4f}")

def test():
    model.eval()  # Set model to evaluation mode
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')

if __name__ == '__main__':
    train()
    test()