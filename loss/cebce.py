from .vanilla_clf_stable import VanillaClassifierStableV0, BinaryVanillaClassifierStableV0
import torch
from torch import nn

    
class CEBCESegmenterStableV2(VanillaClassifierStableV0):
    def __init__(self, args) -> None:
        super().__init__(args)

        self.loss_fn = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        losses = torch.zeros(self.args.seg_n_classes).to(pred.device)

        B, C, H, W = tuple(pred.size())

        for cidx in range(C):
            c_pred = pred[:, cidx]
            c_target = target[:, cidx]

            c_loss = self.loss_fn(c_pred, c_target)

            losses[cidx] = c_loss
        
        batch_weight = torch.ones_like(losses).to(pred.device)
        
        loss_reg = torch.mul(losses, batch_weight).sum()

        logits = self.act(pred)

        entropy = logits * target

        loss = (-1 / (B * H * W)) * torch.sum(entropy)

        return loss + loss_reg