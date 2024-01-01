from .vanilla_clf_stable import VanillaClassifierStableV0
import torch
import torch.nn.functional as F

class CBFocalClassifierV0(VanillaClassifierStableV0):
    def __init__(self, args) -> None:
        super().__init__(args)
        
        self.gamma = args.gamma
    
    def forward(self, pred, target) -> torch.Tensor:

        cls_loss = {}

        logits = self.act(pred)

        B = list(target.size())[0]
        
        beta = (B - 1)/B
        
        for b_logits, b_target in zip(logits, target):
            target_idx = b_target.item()
            _b_logits = b_logits[target_idx]
            if target_idx in cls_loss:
                cls_loss[target_idx].append(_b_logits * torch.pow(1 - _b_logits, self.gamma))
            else:
                cls_loss[target_idx] = [_b_logits * torch.pow(1 - _b_logits, self.gamma)]
        
        sum_cls_loss = {
            _cls : ((1 - beta)/(1 - beta ** len(cls_loss[_cls]))) * sum(cls_loss[_cls]) for _cls in cls_loss
        }

        return (-1 / B) * sum(list(sum_cls_loss.values()))

class CBFocalSegmenterV0(VanillaClassifierStableV0):
    def __init__(self, args) -> None:
        super().__init__(args)
        
        self.gamma = args.gamma
        
    def forward(self, pred, target) -> torch.Tensor:

        cls_loss = {}

        logits = self.act(pred)

        B, C, H, W = tuple(logits.size())

        _logits = logits.permute(0, 2, 3, 1).flatten(0, -2)
        _target = target.permute(0, 2, 3, 1).flatten(0, -2)
        
        N, _ = tuple(_target.shape)
        
        beta = (N - 1)/N

        for cidx in range(C):
            c_logits = _logits[_target[:, cidx] == 1]
            c_target = _target[_target[:, cidx] == 1]
            
            N_c, _ = tuple(c_target.shape)

            entropy = ((1 - beta)/(1 - beta ** N_c)) * torch.sum(torch.pow(1 - c_logits, self.gamma) * c_logits * c_target)

            cls_loss[cidx] = entropy

        return (-1 / (B * H * W)) * sum(list(cls_loss.values()))