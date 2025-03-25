# Owner(s): ["module: unknown"]

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(20, 20)

    def forward(self, input):
        out = self.linear(input[:, 10:30])
        return out.sum()


def main():
    data = torch.randn(10, 50).cuda()
    model = Model().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    for _ in range(10):
        optimizer.zero_grad()
        loss = model(data)
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    main()
