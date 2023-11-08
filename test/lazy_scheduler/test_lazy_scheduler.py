"""
pytest -vs test/lazy_scheduler/test_lazy_scheduler.py::TestLazyScheduler::test_backward_simple_no_segment
"""

import torch
from torch.testing._internal.common_utils import TestCase as TorchTestCase
from torch._dynamo import disable
import functools
from torch._inductor.compile_fx import compile_fx


class TestCase(TorchTestCase):
  def setUp(self):
    torch._dynamo.reset()
    super().setUp()

  def tearDown(self):
    super().tearDown()
    torch._dynamo.reset()


class TestLazyScheduler(TestCase):
  def test_backward_simple_no_segment(self):
    class TestModule(torch.nn.Module):
      def __init__(self):
        super().__init__()

      def func1(self, x, y):
        z = torch.matmul(x, y)
        return z

      def forward(self, x, y):
        z = self.func1(x, y)
        z = z * z
        return z

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from torch._lazy_scheduler import LazyScheduler

    m = TestModule()
    m = m.to(device)
    x = torch.randn(4, 4, requires_grad=True, device=device)
    y = torch.randn(4, 4, requires_grad=True, device=device)

    actual_e = m(x, y)
    actual_e.sum().backward()
    print(f"eager: first iter done")
    actual_e = m(x, y)
    actual_e.sum().backward()
    print(f"eager: second iter done")

    lazy_scheduler = LazyScheduler()
    compiled_m_ls = torch.compile(
      m,
      backend=functools.partial(compile_fx, inner_compile=lazy_scheduler.compile),
      fullgraph=False
    )

    actual_ls = compiled_m_ls(x, y)
    print(f"actual_ls: {actual_ls}")
    actual_ls.sum().backward()
    print(f"compiled_ls: first iter done")
    actual_ls = compiled_m_ls(x, y)
    print(f"actual_ls: {actual_ls}")
    actual_ls.sum().backward()
    print(f"compiled_ls: second iter done")

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests
    run_tests()
