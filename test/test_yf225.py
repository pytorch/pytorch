import torch
import torch._dynamo.config
from torch._inductor.utils import run_and_get_triton_code
from torch.testing import FileCheck
from torch._dynamo import disable

# TORCH_LOGS="+dynamo,aot,inductor" TORCH_COMPILE_DEBUG=1 python test/test_yf225.py

device = "cuda" if torch.cuda.is_available() else "cpu"

@disable()
def g1_mutation_tuple(d, e):
    d.relu_()
    return d, e

@disable()
def g1_mutation_tensor(d, e):
    d.relu_()
    return d + e

@disable()
def g2(a, b):
    return torch.cat(torch.chunk(a * b, 2))

global_a = torch.randn(4, 4, device=device)

@disable()
def g2_read_global_var(a, b):
    return torch.cat(torch.chunk(a * b.div(torch.selu(global_a)), 2))

def global3(a, b):
    return a + b


class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(1))  # torch.randn(4, 4))
        self.register_buffer('buf', torch.randn(1))  # torch.randn(4, 4))

    @disable()
    def f_read_param_mutate_param(self, c):
        self.buf.relu_()
        return c * c * self.weight

    def f2(self, x, y):
        return x + y

    def subfunc0(self, x, y):
        return x + y

    def subfunc01(self, x, y):
        return x + y

    def subfunc1(self, x, y):
      x.relu_()
      self.buf.relu_()
      y = torch.cat(torch.chunk(y, 2))
      return x, y

    def subfunc2(self, x, y):
        # z = torch.relu(x) + g1_mutation_tuple(x, y)[0]
        # z = z + g1_mutation_tensor(x, x)
        # z = z + g2(x, y)
        # z = x + y
        # z = z + g2_read_global_var(x, y)
        # z = z + self.f_read_param_mutate_param(x)
        # z = z + torch.tanh(self.weight)
        # z = z + self.buf
        # z = z + global_a
        # z = z + self.f2(x, y)
        # z = z + global3(x, y)
        z = x + y
        return z

    def forward(self, x, y):
        # z = self.subfunc0(x, y)
        # y = y.relu()
        # z = self.subfunc01(z, y)
        # return z
        x, y = self.subfunc1(x, y)
        y.relu_()
        y = y.relu()
        z = self.subfunc2(x, y)
        z.relu_()
        z = z.relu()
        return z


from torch._lazy_scheduler import Segment, LazyScheduler

m = TestModule()
# TODO: implement submodule method tagging
# Segment._func_to_segment_mapping[m.subfunc0] = "subfunc0"
# Segment._func_to_segment_mapping[m.subfunc01] = "subfunc01"
Segment._func_to_segment_mapping[m.subfunc1] = "subfunc1"
Segment._func_to_segment_mapping[m.subfunc2] = "subfunc2"
m = m.to(device)
x = torch.randn(4, 4, device=device)
y = torch.randn(4, 4, device=device)

lazy_scheduler = LazyScheduler([
    # TODO: figure out why we are scheduling some graphs twice
    # TODO: need to check that actual scheduling is done in the right order
    # "subfunc0",
    # "subfunc01",
    "subfunc1",
    "subfunc2",
])
compiled_m = torch.compile(m, backend=lazy_scheduler.compile, fullgraph=False)

# ref = m(x, y)
actual = compiled_m(x, y)
print(f"first iter done, result: {actual}")
actual = compiled_m(x, y)
print(f"second iter done, result: {actual}")
# # assert torch.allclose(ref, actual)


# TODO: can we use module hook to implement the segment tagging logic?
#
# class Segment:
#     _instance = None
#     def __call__(self, fn, tag):
#         def _fn(*args, **kwargs):
#             with tag_segment(tag):  # TODO: does this work with Dynamo? what if `fn` is not single graph?
#                 return fn(*args, **kwargs)
#         return _fn
#
# def hook(module, args):
#     module.subfunc1 = Segment()(module.subfunc1, "subfunc1")
#     module.subfunc2 = Segment()(module.subfunc2, "subfunc2")
#
# m.register_forward_pre_hook(hook)
