import torch
import torch.nn as nn
import torch.nn.functional as F

class GetIndex(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        return x[self.index]