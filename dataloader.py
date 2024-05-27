import torch
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image
import os

class VHRDataset(Dataset):
    def __init__(self, positive_img_dir, negative_img_dir, gt_dir, transform=None):
        """
        Args:
            positive_img_dir (string): Directory with all the positive images.
            negative_img_dir (string): Directory with all the negative images.
            gt_dir (string): Directory with all the ground truth files for positive images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.positive_img_dir = positive_img_dir
        self.negative_img_dir = negative_img_dir
        self.gt_dir = gt_dir
        self.transform = transform
        
        # List of all images and their types (positive/negative)
        self.imgs = [(os.path.join(positive_img_dir, f), 'positive') for f in os.listdir(positive_img_dir) if f.endswith('.jpg')]
        self.imgs += [(os.path.join(negative_img_dir, f), 'negative') for f in os.listdir(negative_img_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path, img_type = self.imgs[idx]
        image = Image.open(img_path).convert('RGB')
        
        boxes = []
        labels = []
        confidences = []
        
        if img_type == 'positive':
            gt_path = os.path.join(self.gt_dir, os.path.basename(img_path).replace('.jpg', '.txt'))
            with open(gt_path, 'r') as file:
                for line in file:
                    if line == '\n':
                        continue
                    x1, y1, x2, y2, a = line.strip().split(',')
                    x1, x2 = x1.strip('('), x2.strip('(')
                    y1, y2 = y1.strip(')').strip(), y2.strip(')').strip()
                    boxes.append([int(x1), int(y1), int(x2), int(y2)])
                    labels.append(int(a))
                    confidences.append(1)
        else:
            confidences.append(0)

        sample = {'image': image, 'boxes': boxes, 'confidences': confidences, 'labels': labels}
        if self.transform:
            sample = self.transform(sample)

        return sample

class MyTransform:
    """Apply transformations to the image and adjust bounding boxes."""
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, boxes, confidences, labels = sample['image'], sample['boxes'], sample['confidences'], sample['labels']
        w, h = image.size
        new_h, new_w = self.output_size if isinstance(self.output_size, tuple) else (self.output_size, self.output_size)

        # Resize the image
        image = image.resize((new_w, new_h), Image.BILINEAR)

        # Scale the bounding boxes
        new_boxes = []
        for box in boxes:
            scaled_box = [
                box[0] / w, box[1] / h,
                box[2] / w, box[3] / h
            ]
            new_boxes.append(scaled_box)

        # Convert image eto tensor and normalize
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return {'image': image, 'boxes': torch.tensor(new_boxes, dtype=torch.float32), 'confidences': torch.tensor(confidences, dtype=torch.int), 'labels': torch.tensor(labels, dtype=torch.int8)}

# Used to combine several samples into a single batched tensor to be processed by the model.
def collate_fn(batch):
    images = [item['image'] for item in batch]
    boxes = [item['boxes'] for item in batch]
    confidences = [item['confidences'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Pad boxes and labels to the maximum length so that they are of uniform size
    max_boxes_len = max(len(b) for b in boxes) # 25
    max_boxes = 67
    assert max_boxes_len <= max_boxes, f"Oh shit, {max_boxes_len} labels"

    # Padd the detections.
    padded_boxes = torch.zeros((len(batch), max_boxes, 4), dtype=torch.float32)
    padded_labels = torch.zeros((len(batch), max_boxes), dtype=torch.int64)
    padded_confidences = torch.zeros((len(batch), max_boxes), dtype=torch.float32)

    for i in range(len(batch)):
        num_boxes = len(boxes[i])
        if num_boxes > 0:
            padded_boxes[i, :num_boxes] = torch.tensor(boxes[i], dtype=torch.float32)
            padded_labels[i, :num_boxes] = torch.tensor(labels[i], dtype=torch.int64)
            padded_confidences[i, :num_boxes] = torch.tensor(confidences[i], dtype=torch.float32)
    
    # One-hot encode the labels.
    padded_one_hot_labels = F.one_hot(padded_labels, num_classes=11)

    targets = torch.cat((padded_boxes, padded_confidences.unsqueeze(-1), padded_one_hot_labels), dim=-1)
    images = torch.stack(images)
    return images, targets

dataset = VHRDataset(
    positive_img_dir='/home/dan/projects/cse4830/RI-CNN/NWPU VHR-10 dataset/positive image set',
    negative_img_dir='/home/dan/projects/cse4830/RI-CNN/NWPU VHR-10 dataset/negative image set',
    gt_dir='/home/dan/projects/cse4830/RI-CNN/NWPU VHR-10 dataset/ground truth',
    transform = MyTransform((512, 384))
)

total_size = len(dataset)
train_size = int(0.6 * total_size)
val_size = int(0.2 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    

class MrDataHead(DataLoader):
    def __init__(self, dataset=dataset, batch_size=4, shuffle=True, collate_fn=collate_fn):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

train_loader = MrDataHead(dataset=train_dataset, batch_size=4, shuffle=True)
val_loader = MrDataHead(dataset=val_dataset, batch_size=4, shuffle=False)
test_loader = MrDataHead(dataset=test_dataset, batch_size=4, shuffle=False)