import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
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

        sample = {'image': image, 'boxes': boxes, 'labels': labels}
        if self.transform:
            sample = self.transform(sample)

        return sample

class MyTransform:
    """Apply transformations to the image and adjust bounding boxes."""
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, boxes, labels = sample['image'], sample['boxes'], sample['labels']
        w, h = image.size
        new_h, new_w = self.output_size if isinstance(self.output_size, tuple) else (self.output_size, self.output_size)

        # Resize the image
        image = image.resize((new_w, new_h), Image.BILINEAR)

        # Scale the bounding boxes
        new_boxes = []
        for box in boxes:
            scaled_box = [
                box[0] * new_w / w, box[1] * new_h / h,
                box[2] * new_w / w, box[3] * new_h / h
            ]
            new_boxes.append(scaled_box)

        # Convert image eto tensor and normalize
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return {'image': image, 'boxes': torch.tensor(new_boxes, dtype=torch.float32), 'labels': torch.tensor(labels, dtype=torch.int8)}

# Used to combine several samples into a single batched tensor to be processed by the model.
def collate_fn(batch):
    images = [item['image'] for item in batch]
    boxes = [item['boxes'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Pad boxes and labels to the maximum length so that they are of uniform size
    max_boxes_len = max(len(b) for b in boxes) # 25
    max_boxes = 67
    assert max_boxes_len <= max_boxes, f"Oh shit, {max_boxes_len} labels"
    padded_boxes = []
    padded_labels = []
    for b, l in zip(boxes, labels):
        if b.ndim == 2:
            b = torch.nn.functional.pad(b, (0, 0, 0, max_boxes - b.size(0)))
            l = torch.nn.functional.pad(l, (0, max_boxes - l.size(0)))
        else:
            b = torch.zeros((max_boxes, 4))
            l = torch.zeros((max_boxes,  ))
    
        padded_boxes.append(b)
        padded_labels.append(l)
    
    # Convert padded lists to tensors
    padded_boxes = torch.stack(padded_boxes)
    padded_labels = torch.stack(padded_labels)

    return {'image': torch.stack(images), 'boxes': padded_boxes, 'labels': padded_labels}

dataset = VHRDataset(
    positive_img_dir='/home/dan/projects/cse4830/RI-CNN/NWPU VHR-10 dataset/positive image set',
    negative_img_dir='/home/dan/projects/cse4830/RI-CNN/NWPU VHR-10 dataset/negative image set',
    gt_dir='/home/dan/projects/cse4830/RI-CNN/NWPU VHR-10 dataset/ground truth',
    transform = MyTransform((512, 384))
)

class MrDataHead(DataLoader):
    def __init__(self, dataset=dataset, batch_size=4, shuffle=True, collate_fn=collate_fn):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)