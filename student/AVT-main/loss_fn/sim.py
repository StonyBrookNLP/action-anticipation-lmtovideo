import torch.nn as nn


class CosineSim(nn.CosineSimilarity):
    def __init__(self, device):
        self.device = device
        super().__init__(dim=1)

    def forward(self, inp, tgt):
        """
        Args:
            inp: (*, C)
            tgt: (*, C)
            Will normalize the input before the sim
        """
        cosine_sim = super().forward(inp, tgt.to(self.device))
        return 1-cosine_sim
