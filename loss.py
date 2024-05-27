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
        pred_confidence = nn.functional.sigmoid(predictions[:, :, 4])
        pred_classes = predictions[:, :, 5:]

        # Split targets
        target_boxes = targets[:, :, 0:4]
        target_confidence = predictions[:, :, 4]
        target_classes = targets[:, :, 5:].argmax(dim=2)

        # Box loss - only for boxes with an object
        box_loss = self.mse_loss(pred_boxes * target_confidence.unsqueeze(-1), target_boxes * target_confidence.unsqueeze(-1)) / num_boxes

        # Confidence loss - for all boxes
        confidence_loss = self.bce_loss(pred_confidence, target_confidence) / num_boxes

        #print(pred_classes.shape)

        # Class loss - only for boxes with an object
        class_loss = self.ce_loss(pred_classes.view(-1, pred_classes.size(-1)), target_classes.view(-1))
        class_loss = (class_loss * target_confidence).sum() / num_boxes


        # Combine losses
        print(box_loss.item(), class_loss.item(), confidence_loss.item())
        total_loss = box_loss + confidence_loss + class_loss
        return total_loss