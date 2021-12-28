#!/usr/bin/env python3
import torch.jit
import torch.nn as nn

model = nn.TransformerEncoderLayer(256, 8)
input_data = torch.empty([1, 32, 256])
traced_model = torch.jit.trace(model, input_data)
traced_model.save("model.ts")
