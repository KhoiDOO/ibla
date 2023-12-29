from .vanilla_clf import VanillaClassifierV0
import torch

class VanillaSegmenterV0(VanillaClassifierV0):
    def __init__(self, args) -> None:
        super().__init__(args)

    def forward(self, pred, target) -> torch.Tensor:
        pred_soft = self.act(pred)

        logits = torch.log(pred_soft)

        B, C, H, W = tuple(logits.size())

        entropy = logits * target

        return (-1 / (B * H * W)) * torch.sum(entropy)

class VanillaSegmenterV1(VanillaSegmenterV0):
    def __init__(self, args) -> None:
        super().__init__(args)
    
    def forward(self, pred, target) -> torch.Tensor:

        cls_loss = {}

        pred_soft = self.act(pred)

        logits = torch.log(pred_soft)

        B, C, H, W = tuple(logits.size())

        _logits = logits.permute(0, 2, 3, 1).flatten(0, -2)
        _target = target.permute(0, 2, 3, 1).flatten(0, -2)

        for cidx in range(C):
            c_logits = _logits[_target[:, cidx] == 1]
            c_target = _target[_target[:, cidx] == 1]

            entropy = torch.sum(c_logits * c_target)

            cls_loss[cidx] = entropy

        return (-1 / (B * H * W)) * sum(list(cls_loss.values()))