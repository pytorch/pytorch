import torch.nn as nn


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 20)
        self.relu = nn.ReLU()
