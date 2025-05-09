import torch.nn as nn
import math

class MLPpropose(nn.Module):
    def __init__(self, trace_length):
        super().__init__()
        self.trace_length = trace_length
        self.network = nn.Sequential(
            nn.Linear(self.trace_length, math.floor((self.trace_length) * 3 / 4)),
            nn.SELU(),
            nn.Linear(math.floor((self.trace_length) * 3 / 4), math.floor((self.trace_length)/2)),
            nn.SELU(),
            nn.Linear(math.floor((self.trace_length)/2), math.floor((self.trace_length)* 3 / 4)),
            nn.SELU(),
            nn.Linear(math.floor((self.trace_length) * 3 / 4), self.trace_length),
        )
        
    def forward(self, x):
        output = self.network(x)
        return output