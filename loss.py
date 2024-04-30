import torch.nn as nn

class DetectionLoss(nn.Module):
    def __init__(self):
        super(DetectionLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='sum')
        self.ce_loss = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, predictions, targets):
        num_boxes = predictions.shape[1]

        # Split predictions
        pred_boxes = predictions[:, :, 0:4]
        pred_confidence = predictions[:, :, 4]
        pred_classes = predictions[:, :, 5:]

        # Split targets
        target_boxes = targets[:, :, 0:4]
        target_confidence = predictions[:, :, 4]
        target_classes = targets[:, :, 5:].argmax(dim=2)

        # Box loss - only for boxes with an object
        box_loss = self.mse_loss(pred_boxes * target_confidence.unsqueeze(-1), target_boxes * target_confidence.unsqueeze(-1)) / num_boxes

        # Confidence loss - for all boxes
        confidence_loss = self.bce_loss(pred_confidence, target_confidence) / num_boxes

        # Class loss - only for boxes with an object
        class_loss = self.ce_loss(pred_classes, target_classes) * target_confidence
        class_loss = class_loss.sum() / num_boxes

        # Combine losses
        total_loss = box_loss + confidence_loss + class_loss
        return total_loss