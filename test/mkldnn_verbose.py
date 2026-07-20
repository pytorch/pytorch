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
    with torch.backends.mkldnn.verbose(level):
        m(d)


def run_acl_bf16_linear(level):
    # Keep Inductor import out of the default verbose helper subprocess.
    from torch._inductor import config as inductor_config

    dtype = torch.bfloat16
    with torch.no_grad(), inductor_config.patch({"freezing": True}):
        x = torch.rand(size=(1024, 1024), dtype=dtype)
        linear = torch.nn.Linear(1024, 1024).to(dtype).eval()
        linear = torch.compile(linear)
        linear(x)
        with torch.backends.mkldnn.verbose(level):
            linear(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose-level", default=0, type=int)
    parser.add_argument("--model", choices=["conv", "acl-bf16-linear"], default="conv")
    args = parser.parse_args()
    try:
        if args.model == "acl-bf16-linear":
            run_acl_bf16_linear(args.verbose_level)
        else:
            run_model(args.verbose_level)
    except Exception as e:
        print(e)
