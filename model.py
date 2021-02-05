#!/usr/bin/env python3
import torch.jit
import torch.nn as nn


# 80Layers CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.convs = nn.ModuleList([nn.Conv2d(256, 256, 3, padding=1) for _ in range(80)])

    def forward(self, x):
        for c in self.convs:
            x = c.forward(x)
        return x


model = Net()
input_data = torch.empty([1, 256, 32, 32])
traced_model = torch.jit.trace(model, input_data)
traced_model.save("model.ts")
