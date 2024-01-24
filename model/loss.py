import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=1, C=5):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5
    
    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        
        b_start = self.C + 1
        b_end = self.C + 5

        iou_b1 = intersection_over_union(predictions[..., b_start:b_end], target[..., b_start:b_end])
        iou_b2 = intersection_over_union(predictions[..., b_start:b_end], target[..., b_start:b_end])

        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = target[..., self.C].unsqueeze(3)

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        box_predictions = exists_box * predictions[..., b_start:b_end]
        
        box_targets = exists_box * target[..., b_start:b_end]

        # Take sqrt of width, height of boxes to ensure that
        # they are not negative

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6))
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(torch.flatten(box_predictions, end_dim=-2), torch.flatten(box_targets, end_dim=-2))

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #


        pred_box = predictions[..., b_start:b_end]

        object_loss = self.mse(
            torch.flatten(exists_box * iou_maxes), 
            torch.flatten(exists_box * predictions[..., self.C:self.C+1])
        )

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., self.C:self.C+1]), 
            torch.flatten((1 - exists_box) * target[..., self.C:self.C+1])
        )

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :self.C], end_dim=-2), 
            torch.flatten(exists_box * target[..., :self.C], end_dim=-2)
        )

        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        return loss

