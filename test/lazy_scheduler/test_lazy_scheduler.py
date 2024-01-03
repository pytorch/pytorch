"""
pytest -vs test/lazy_scheduler/test_lazy_scheduler.py::TestLazyScheduler::test_single_segment
pytest -vs test/lazy_scheduler/test_lazy_scheduler.py::TestLazyScheduler::test_split_module_above_aotautograd
pytest -vs test/lazy_scheduler/test_lazy_scheduler.py::TestLazyScheduler::test_compile_fx_with_segment_info
"""

import torch
import torch.utils._pytree as pytree
from torch.testing._internal.common_utils import TestCase as TorchTestCase
from torch._dynamo import disable
import functools
import itertools
from typing import Optional, Dict, Callable, List
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._subclasses.async_tensor import AsyncTensor
from collections import defaultdict, OrderedDict
import weakref
import threading
from torch.utils._python_dispatch import return_and_correct_aliasing
from torch._dynamo.backends.common import aot_autograd
from torch._inductor.compile_fx import compile_fx as inductor_compile_fx
from torch._inductor.compile_fx import compile_fx_inner as inductor_compile_fx_inner

fake_mode = FakeTensorMode()


class AsyncFuncHandle:
  """
  We use this class to represent the function that needs to be scheduled.
  It also has methods for checking whether the function has been scheduled or completed.
  """
  _gm_to_handle_mapping: Dict[torch.fx.GraphModule, "AsyncFuncHandle"] = {}

  def __init__(self, compiled_fn, args, outs_async, scheduler):
    self.cuda_event = torch.cuda.Event()
    self.compiled_fn: Callable = compiled_fn
    self.args = args
    self.outs_async = outs_async
    self.outs = None
    self.is_going_to_be_scheduled = False
    self._scheduler = weakref.ref(scheduler)

  def schedule(self):
    # make sure to schedule only once
    if self.is_going_to_be_scheduled:
      return
    self.is_going_to_be_scheduled = True
    gm = self._scheduler()._handle_to_gm_map[self]
    AsyncTensor.wait_until_materialized(self.args)
    args_materialized = pytree.tree_map_only(AsyncTensor, lambda x: x._materialized_tensor, pytree.tree_map(lambda x: x.detach(), self.args))
    self.outs = self.compiled_fn(list(args_materialized))
    self.cuda_event.record()

  def wait_for_completion(self):
    self.cuda_event.synchronize()
    for out, out_async in zip(self.outs, self.outs_async):
      # Set the output AsyncTensor's underlying materialized tensor
      # to be the actual output tensor.
      out_async.materialize_with_value(out)

  def is_completed(self):
    return self.cuda_event.query()

  def scheduler(self):
    scheduler = self._scheduler()
    assert scheduler is not None
    return scheduler


# NOTE: this is only for threading outputs through multiple submodules when doing module splitting above AOTAutograd (but after Dynamo).
class _LazilyCompiledModule(torch.nn.Module):
  def __init__(self, submod, compiler):
    super().__init__()
    self.submod = submod
    self.compiler = compiler
    self.compiled = False

  def __call__(self, *args):
    if not self.compiled:
      new_submod = self.compiler(self.submod, args)
      del self.submod
      self.submod = new_submod
      self.compiled = True
      self.compiler = None
    x = self.submod(*args)
    return x


def split_module_based_on_segment_info(gm: torch.fx.GraphModule):
  known_segments = []
  for node in gm.graph.nodes:
    if len(known_segments) == 0 or node.meta["segment"] != known_segments[-1]:
      known_segments.append(node.meta["segment"])

  def split_callback(node):
    return known_segments.index(node.meta["segment"])

  qualname_map = {}
  gm_after_split = torch.fx.passes.split_module.split_module(
    m=gm,
    root_m=None,
    split_callback=split_callback,
    qualname_map=qualname_map,
    keep_original_order=True,
  )
  return gm_after_split


class LazySchedulerGraphModule(torch.nn.Module):
  """
  This module wraps around a GraphModule.
  Its __call__ method doesn't execute the graph module immediately.
  Instead, it calls the scheduler's maybe_run method, which decides
  whether to run the graph module based on the schedule.
  """
  def __init__(self, scheduler, gm, compiled_fn):
    super().__init__()
    self.scheduler = scheduler
    self.gm = gm
    self.compiled_fn = compiled_fn

  def __call__(self, *args):
    assert self.compiled_fn is not None
    return self.scheduler.maybe_run(self.gm, self.compiled_fn, *args)


class LazyScheduler:
  """
  LazyScheduler is used to decide when to schedule the execution of a graph module (based on the schedule).
  """
  def __init__(self):
    self._gm_to_handle_map = OrderedDict()
    self._handle_to_gm_map = OrderedDict()

  def _compile_fx_with_segment_info(
    self,
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    segment_assignment_fn=None,
    **kwargs,
  ):
    if segment_assignment_fn is not None:
      segment_assignment_fn(gm)
    # Assumes `gm` already has segment info in each of its nodes
    gm_after_split = split_module_based_on_segment_info(gm)
    for name, sub_gm in gm_after_split.named_children():
      lazy_sub_gm = _LazilyCompiledModule(
        sub_gm,
        functools.partial(inductor_compile_fx, **kwargs)
      )
      setattr(gm_after_split, name, lazy_sub_gm)
    # Trigger compile_fx in all submodules
    gm_after_split(*example_inputs)
    return gm_after_split

  def compile_fx(
    self,
    # NOTE: matches positional args in compile_fx signature
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    segment_assignment_fn=None,
    **kwargs,
  ):
    return self._compile_fx_with_segment_info(gm, example_inputs, segment_assignment_fn, **kwargs)

  def compile_fx_inner(
    self,
    # NOTE: assumes first arg is GraphModule in compile_fx_inner signature
    gm: torch.fx.GraphModule,
    *args,
    **kwargs,
  ):
    """
    Compiles a graph module using Inductor compile_fx_inner,
    and wraps the output compiled_fn in a LazySchedulerGraphModule to be called later.
    """
    assert isinstance(gm, torch.fx.GraphModule)
    compiled_fn = inductor_compile_fx_inner(gm, *args, **kwargs)
    lazy_gm = LazySchedulerGraphModule(
      self,
      gm,
      compiled_fn,
    )
    return lazy_gm

  def maybe_run(self, gm, compiled_fn, *args):
    """
    Decides whether to run the graph module based on the schedule.

    Always immediately returns AsyncTensor as output, and the AsyncTensor will be populated
    when the graph module is eventually executed.
    """
    # Create the handle and the async tensors
    args_fake = []
    for arg in args:
      if isinstance(arg, AsyncTensor):
        args_fake.append(arg._fake)
      elif isinstance(arg, torch.Tensor):
        args_fake.append(fake_mode.from_tensor(arg))
    with fake_mode:
      outs_fake = gm(*args_fake)

    outs_async = tuple(AsyncTensor(fake_tensor=out_fake) for out_fake in outs_fake)
    if gm in self._gm_to_handle_map:
      cur_handle = self._gm_to_handle_map[gm]
    else:
      cur_handle = AsyncFuncHandle(compiled_fn, args=args, outs_async=outs_async, scheduler=self)
      self._gm_to_handle_map[gm] = cur_handle
      self._handle_to_gm_map[cur_handle] = gm
    for out_async in outs_async:
      out_async.set_handle(cur_handle)

    # NOTE: add more complex logic here (e.g. check against the schedule, etc.)
    # cur_handle.schedule()
    # cur_handle.wait_for_completion()
    return cur_handle.outs_async


class TestCase(TorchTestCase):
  def setUp(self):
    torch._dynamo.reset()
    super().setUp()

  def tearDown(self):
    super().tearDown()
    torch._dynamo.reset()


class TestModule(torch.nn.Module):
  def __init__(self):
    super().__init__()

  def func1(self, x, z):
    return torch.matmul(x, z)

  def func2(self, x, y):
    z = torch.add(x, y)
    return z

  def forward(self, x, y):
    z = self.func1(x, y)
    z = self.func2(x, z)
    z = z * z
    return z


class TestLazyScheduler(TestCase):
  def _validate(self, fn, backend, *args, skip_check=False):
    cloned_args = []
    for arg in args:
      cloned_args.append(arg.clone().detach().requires_grad_(arg.requires_grad))

    # Eager, 1st iter
    torch.manual_seed(0)
    expected = fn(*args)
    expected.sum().backward()

    # Eager, 2nd iter
    torch.manual_seed(0)
    expected = fn(*args)
    expected.sum().backward()

    compiled_fn = torch.compile(fn, fullgraph=False, backend=backend)

    # Compiled, 1st iter
    torch.manual_seed(0)
    result = compiled_fn(*cloned_args)
    r_sum = result.sum()
    r_sum.backward()

    # Compiled, 2nd iter
    torch.manual_seed(0)
    result = compiled_fn(*cloned_args)
    result.sum().backward()

    if not skip_check:
      self.assertEqual(
        result,
        expected,
        msg="Output mismatch between torch.compile and eager versions",
      )
      for arg, cloned_arg in zip(args, cloned_args):
        self.assertEqual(
          arg.grad,
          cloned_arg.grad,
          msg=f"Gradient mismatch between torch.compile and eager versions. arg.grad: {arg.grad}, cloned_arg.grad: {cloned_arg.grad}",
        )

  def test_single_segment(self):
    # Check that output and gradients are correct when there is only one segment in the model.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = TestModule()
    m = m.to(device)
    x = torch.randn(4, 4, requires_grad=True, device=device)
    y = torch.randn(4, 4, requires_grad=True, device=device)
    lazy_scheduler = LazyScheduler()

    def segment_assignment_fn(gm):
      for i, node in enumerate(gm.graph.nodes):
        node.meta["segment"] = "seg1"

    self._validate(
      m,
      functools.partial(
        lazy_scheduler.compile_fx,
        inner_compile=lazy_scheduler.compile_fx_inner,
        segment_assignment_fn=segment_assignment_fn
      ),
      x,
      y,
    )

  def test_split_module_above_aotautograd(self):
    # Check that GraphModule produced by Dynamo is correctly split
    # (each submodule only contains one NN method from original module, which maps to one segment)
    # before entering AOTAutograd.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = TestModule()
    m = m.to(device)
    x = torch.randn(4, 4, requires_grad=True, device=device)
    y = torch.randn(4, 4, requires_grad=True, device=device)
    lazy_scheduler = LazyScheduler()

    def segment_assignment_fn(gm):
      for _, node in enumerate(gm.graph.nodes):
        assert "nn_module_method" in node.meta
        # One NN module method maps to one segment
        node.meta["segment"] = str(node.meta["nn_module_method"])

    def compile_fx_count_submods(
      gm: torch.fx.GraphModule,
      example_inputs: List[torch.Tensor],
      segment_assignment_fn=None,
      **kwargs,
    ):
      gm_after_split = lazy_scheduler.compile_fx(gm, example_inputs, segment_assignment_fn, **kwargs)
      # Test that one submodule is created for each NN module method: forward, func1, func2.
      self.assertEqual(len(list(gm_after_split.named_children())), 3)
      for _, submod in enumerate(gm_after_split.named_children()):
        self.assertTrue(isinstance(submod, _LazilyCompiledModule))
      return gm_after_split

    def compile_fx_inner_assert_single_nn_method(
      gm: torch.fx.GraphModule,
      *args,
      **kwargs,
    ):
      # All nodes in a fwd or bwd GraphModule should belong to the same NN module method.
      nn_module_method_name = None
      for i, node in enumerate(gm.graph.nodes):
        if node.op == "call_function":
          assert "nn_module_method" in node.meta
          if nn_module_method_name is None:
            nn_module_method_name = node.meta["nn_module_method"]
          else:
            self.assertEqual(node.meta["nn_module_method"], nn_module_method_name)
      return lazy_scheduler.compile_fx_inner(gm, *args, **kwargs)

    self._validate(
      m,
      functools.partial(
        compile_fx_count_submods,
        inner_compile=compile_fx_inner_assert_single_nn_method,
        segment_assignment_fn=segment_assignment_fn
      ),
      x,
      y,
    )

def test_segment_tagging(self):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = TestModule()
    m = m.to(device)
    x = torch.randn(4, 4, requires_grad=True, device=device)
    y = torch.randn(4, 4, requires_grad=True, device=device)
    lazy_scheduler = LazyScheduler()

    segment_dict = {}

    # This is roughly how the register_segment function will look like
    def register_segment(method, name):
      global segment_dict
      segment_dict[method] = name

    register_segment(m.func2, "func2")

    def segment_assignment_fn(gm):
      for _, node in enumerate(gm.graph.nodes):
        assert "nn_module_method" in node.meta
        if node.meta["nn_module_method"] in segment_dict:
          node.meta["segment"] = str(node.meta["nn_module_method"])
        else:
          # TODO: add unnamed segment

    def compile_fx_count_submods(
      gm: torch.fx.GraphModule,
      example_inputs: List[torch.Tensor],
      segment_assignment_fn=None,
      **kwargs,
    ):
      gm_after_split = lazy_scheduler.compile_fx(gm, example_inputs, segment_assignment_fn, **kwargs)
      # func2 is the middle function, so should produce 3 submodules
      self.assertEqual(len(list(gm_after_split.named_children())), 3)
      for _, submod in enumerate(gm_after_split.named_children()):
        self.assertTrue(isinstance(submod, _LazilyCompiledModule))
      return gm_after_split

    self._validate(
      m,
      functools.partial(
        compile_fx_count_submods,
        inner_compile=compile_fx_inner_assert_single_nn_method,
        segment_assignment_fn=segment_assignment_fn
      ),
      x,
      y,
    )

"""
TODO:
1. Enable test_segment_tagging to check segment tagging is working
2. Add scheduling logic, add unit test to check it's working
"""

if __name__ == "__main__":
  from torch._dynamo.test_case import run_tests
  run_tests()
