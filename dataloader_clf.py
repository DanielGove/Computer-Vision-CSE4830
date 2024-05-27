import os
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

class EuroSATDataset(Dataset):
    def __init__(self, directory, transform=None):
        """
        Args:
            directory (string): Path to the dataset directory.
            transform (callable, optional): Optional transform to be applied.
        """
        self.directory = directory
        self.transform = transform
        self.samples = []  # Store (image_path, label_index)

        label_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        self.label_to_index = {label_dir: i for i, label_dir in enumerate(label_dirs)}

        for label_dir in label_dirs:
            class_dir = os.path.join(directory, label_dir)
            for image_name in os.listdir(class_dir):
                self.samples.append((os.path.join(class_dir, image_name), self.label_to_index[label_dir]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize to ImageNet stats
])

def create_data_loaders(directory="/home/dan/projects/cse4830/RI-CNN/EuroSAT_RGB", batch_size=64, train_split=0.8):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = EuroSATDataset(directory, transform=transform)
    total_count = len(dataset)

    train_count = int(train_split * total_count)
    test_count = total_count - train_count

    train_dataset, test_dataset = random_split(dataset, [train_count, test_count])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader