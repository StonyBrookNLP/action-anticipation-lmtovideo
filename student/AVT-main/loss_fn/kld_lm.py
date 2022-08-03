
import torch
import torch.nn as nn


def softmax_with_T(logits: torch.Tensor, device, t: int = 1):
    """
    softmax with temperature T for soft targets
    """
    log_soft = torch.log_softmax(logits/t, dim=1).to(device)
    return log_soft


class KLDiv(nn.KLDivLoss):
    def __init__(self, device):
        # reduction is applied later
        self.device = device
        super().__init__(reduction='none', log_target=True)

    def forward(self, inp: torch.Tensor, tgt: torch.Tensor):
        """
        Loss between inp (logits/action from AVT Student) and tgt (logits from LM Teacher)
        Note: both converted to soft targets here.
        Args:
            inp: (B, C)
            tgt: (B, C)
        """
        assert inp.ndim == tgt.ndim
        assert inp.shape == tgt.shape
        T = 10

        # # select a few important classes from inp and tgt
        # k = 50
        # # take top k classes for distillation
        # tgt_topk, topk_ind = torch.topk(tgt, k=k, sorted=False)
        # tgt_redist = tgt_topk.to(self.device)
        # topk_ind = topk_ind.to(self.device)
        # inp_redist = torch.gather(inp, dim=1, index=topk_ind)
        #
        # # n = inp.shape[1]
        # # tgt_sorted_ind = torch.argsort(tgt)
        # # tgt_redist_ind = torch.cat([tgt_sorted_ind[:, :k], tgt_sorted_ind[:, int((n - k) / 2):int((n + k) / 2)],
        # #                             tgt_sorted_ind[:, n - k:n]], dim=-1).to(self.device)
        # # tgt_redist = torch.gather(tgt.to(self.device), dim=1, index=tgt_redist_ind)
        # # inp_redist = torch.gather(inp, dim=1, index=tgt_redist_ind)
        # #
        # res = super().forward(softmax_with_T(inp_redist, self.device, T), softmax_with_T(tgt_redist, self.device, T))
        # #
        res = super().forward(softmax_with_T(inp, self.device, T), softmax_with_T(tgt, self.device, T))
        return res * (T**2)
