import torch.nn as nn
import numpy as np
import torch
class SpecificNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(SpecificNorm, self).__init__()
        self.mean = np.array([0.485, 0.456, 0.406])
        self.mean = torch.from_numpy(self.mean).float().cpu()
        self.mean = self.mean.view([1, 3, 1, 1])
        self.std = np.array([0.229, 0.224, 0.225])
        self.std = torch.from_numpy(self.std).float().cpu()
        self.std = self.std.view([1, 3, 1, 1])

    def forward(self, x):
        mean = self.mean.expand([1, 3, x.shape[2], x.shape[3]])
        std = self.std.expand([1, 3, x.shape[2], x.shape[3]])
        return (x - mean) / std