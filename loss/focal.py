from .vanilla_clf_stable import VanillaClassifierStableV0
import torch
import torch.nn.functional as F

class FocalClassifierV0(VanillaClassifierStableV0):
    def __init__(self, args) -> None:
        super().__init__(args)
        
        self.gamma = args.gamma
    
    def forward(self, pred, target) -> torch.Tensor:

        logits = self.act(pred)

        B, C = tuple(logits.size())

        entropy = torch.pow(1 - logits, self.gamma) * logits * F.one_hot(target, num_classes=C).float()

        return (-1 / B) * torch.sum(entropy)

class FocalSegmenterV0(VanillaClassifierStableV0):
    def __init__(self, args) -> None:
        super().__init__(args)
        
        self.gamma = args.gamma

    def forward(self, pred, target) -> torch.Tensor:
        logits = self.act(pred)

        B, C, H, W = tuple(logits.size())

        entropy = torch.pow(1 - logits, self.gamma) * logits * target

        return (-1 / (B * H * W)) * torch.sum(entropy)