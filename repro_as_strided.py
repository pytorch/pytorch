import torch

def fn():
    x = torch.arange(4.0, dtype=torch.float32)
    y = x.as_strided_((2, 2), (2, 1))
    observer = torch.max(y)
    return y, x.size(), y.size(), x.stride(), y.stride(), observer

def show(name, out):
    y, x_size, y_size, x_stride, y_stride, observer = out
    print(name)
    print("y:", y)
    print("x_size:", tuple(x_size))
    print("y_size:", tuple(y_size))
    print("x_stride:", tuple(x_stride))
    print("y_stride:", tuple(y_stride))
    print("max:", observer)

print("torch", torch.__version__)

eager = fn()
show("eager", eager)

compiled = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=True)
compiled_out = compiled()
show("compiled", compiled_out)
