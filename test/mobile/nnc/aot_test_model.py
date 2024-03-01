import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def forward(self, x):
        return torch.add(x, 10)

model = NeuralNetwork()
script = torch.jit.script(model)
torch.jit.save(script, "aot_test_model.pt")
