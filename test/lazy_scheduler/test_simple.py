"""
pytest -vs test/lazy_scheduler/test_simple.py::TestSimple::test_custom_autograd_func_returning_subclass

pytest -vs test/lazy_scheduler/test_simple.py::TestSimple::test_inductor_inner_compile_returning_subclass
"""

import torch
from torch.testing._internal.common_utils import TestCase as TorchTestCase
from torch._dynamo import disable
import functools
import itertools
from typing import Optional, Dict, Callable
from torch._subclasses.fake_tensor import FakeTensorMode
from collections import defaultdict, OrderedDict
import weakref
import threading
from torch._inductor.compile_fx import compile_fx, compile_fx_inner

class TestCase(TorchTestCase):
  def setUp(self):
    torch._dynamo.reset()
    super().setUp()

  def tearDown(self):
    super().tearDown()
    torch._dynamo.reset()



class LazyGraphModule(torch.nn.Module):
  def __init__(self, gm, compiled_fn):
    super().__init__()
    self.gm = gm
    self.compiled_fn = compiled_fn

  def __call__(self, *args):
    assert self.compiled_fn is not None
    # TODO: this is hacky, but how do we know the actual number of outputs (excluding tangents)?
    out, _ = self.compiled_fn(list(args))
    return OneTensor(out)

# NOTE: this matches compile_fx_inner signature
def custom_compile(
  gm: torch.fx.GraphModule,
  example_inputs,
  cudagraphs = None,
  num_fixed = 0,
  is_backward = False,
  graph_id = None,
  cpp_wrapper = False,
  aot_mode = False,
  is_inference = False,
  boxed_forward_device_index = None,
  user_visible_outputs = frozenset(),
  layout_opt = None,
  extern_node_serializer = None,
):
  compiled_fn = compile_fx_inner(
    gm,
    example_inputs,
    cudagraphs=cudagraphs,
    num_fixed=num_fixed,
    is_backward=is_backward,
    graph_id=graph_id,
    cpp_wrapper=cpp_wrapper,
    aot_mode=aot_mode,
    is_inference=is_inference,
    boxed_forward_device_index=boxed_forward_device_index,
    user_visible_outputs=user_visible_outputs,
    layout_opt=layout_opt,
    extern_node_serializer=extern_node_serializer,
  )
  lazy_gm = LazyGraphModule(
    gm,
    compiled_fn,
  )
  return lazy_gm


from torch.testing._internal.two_tensor import OneTensor

class TestLazyScheduler(TestCase):
  def test_custom_autograd_func_returning_subclass(self):
    class TestAutogradFunc(torch.autograd.Function):
      @staticmethod
      def forward(self, input):
        out = torch.matmul(input, input)
        return OneTensor(out)

      @staticmethod
      def backward(self, grad_output):
        return grad_output

    def f(x):
      return TestAutogradFunc.apply(x)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.randn(4, 4, requires_grad=True, device=device)
    out = f(x)
    out.sum().backward()
    # the type of `x.grad` here will be a OneTensor, both in eager mode and with compile
    assert out.grad_fn is not None
    print(x.grad)

    xc = torch.randn(4, 4, requires_grad=True, device=device)
    out_c = torch.compile(f)(xc)
    out_c.sum().backward()
    # the type of `x.grad` here will be a OneTensor, both in eager mode and with compile
    assert out_c.grad_fn is not None
    print(xc.grad)

  def test_inductor_inner_compile_returning_subclass(self):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def f(x):
      return torch.matmul(x, x)

    xc = torch.ones(4, 4, requires_grad=True, device=device)
    compiled_f = torch.compile(
      f,
      backend=functools.partial(compile_fx, inner_compile=custom_compile),
    )
    out_c = compiled_f(xc)
    # breakpoint()
    out_c.sum().backward()
    # the type of `x.grad` here will be a OneTensor, both in eager mode and with compile
    assert out_c.grad_fn is not None
    print(xc.grad)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests
    run_tests()
