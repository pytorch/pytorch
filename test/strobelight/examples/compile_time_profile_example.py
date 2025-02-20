# mypy: allow-untyped-defs
import torch


# if __name__ == "__main__":
# You can pass TORCH_COMPILE_STROBELIGHT=True instead.
# StrobelightCompileTimeProfiler.enable()


def fn(x, y, z):
    return x * y + z


@torch.compile(fullgraph=True)
def mywork(x, y, z):
    return fn(x, y, z)


x = torch.randn(10, 10)
y = torch.randn(10, 10)
z = torch.randn(10, 10)
mywork(x, y, z)
