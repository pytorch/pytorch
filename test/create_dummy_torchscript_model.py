# Usage: python create_dummy_model.py <name_of_the_file>
import sys
import torch
from torch import nn


class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


if __name__ == '__main__':
    jit_module = torch.jit.script(NeuralNetwork())
    torch.jit.save(jit_module, sys.argv[1])
    orig_module = nn.Sequential(
        nn.Linear(28 * 28, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10),
    )
    torch.save(orig_module, sys.argv[1] + ".orig")
