import argparse

import torch


class Module(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 10, 5, 1)

    def forward(self, x):
        y = self.conv(x)
        return y


def run_model(level):
    m = Module().eval()
    d = torch.rand(1, 1, 112, 112)
    with torch.backends.onednn.verbose(level):
        m(d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose-level", default=0, type=int)
    args = parser.parse_args()
    try:
        run_model(args.verbose_level)
    except Exception as e:
        print(e)
