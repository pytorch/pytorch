import torch
from functorch.compile import min_cut_rematerialization_partition
from torch._dynamo.backends.common import aot_autograd
import types

class TestNonDepSegmentModule(torch.nn.Module):
  """
  Dependency chain:
  func1 ->
          -> mul -> output
  func2 ->
  """
  def __init__(self):
    super().__init__()

  def func1(self, x, y):
    return torch.matmul(x, y)

  def func2(self, x, y):
    return torch.add(x, y)

  def forward(self, x, y):
    z1 = self.func1(x, y)
    z2 = self.func2(x, y)
    z = z1 * z2
    return z


def dummy_compile(gm, args):
    print(f"here1")
    return gm

backend = aot_autograd(
    fw_compiler=dummy_compile,
    bw_compiler=dummy_compile,
    partition_fn=min_cut_rematerialization_partition,
)

model = TestNonDepSegmentModule()

model2 = TestNonDepSegmentModule()

print(f"id(model.func1.__code__): {id(model.func1.__code__)}")
print(f"id(model2.func1.__code__): {id(model2.func1.__code__)}")

model.func1 = torch.compile(model.func1, backend=backend)
# model.forward = torch.compile(model.forward)

out = model(torch.randn(3, 3, requires_grad=True), torch.randn(3, 3, requires_grad=True))
out.sum().backward()

"""
Observation:
1. If `model.forward = torch.compile(model.forward)` is not enabled, prints "here1" twice (it means inner compile result is taking effect).
2. If `model.forward = torch.compile(model.forward)` is enabled, doesn't print "here1" (it means inner compile result is overwritten).
"""
