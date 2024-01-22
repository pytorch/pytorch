"""
# pytest -vs test/lazy_scheduler/test_lazy_scheduler.py::TestLazyScheduler::test_single___unnamed_segment
pytest -vs test/lazy_scheduler/test_lazy_scheduler.py::TestLazyScheduler::test_single_segment_prefix_fwd
pytest -vs test/lazy_scheduler/test_lazy_scheduler.py::TestLazyScheduler::test_single_segment_prefix_fwd_bwd
pytest -vs test/lazy_scheduler/test_lazy_scheduler.py::TestLazyScheduler::test_split_module_dep_segments
pytest -vs test/lazy_scheduler/test_lazy_scheduler.py::TestLazyScheduler::test_split_module_non_dep_segments
pytest -vs test/lazy_scheduler/test_lazy_scheduler.py::TestLazyScheduler::test_segment_tagging
pytest -vs test/lazy_scheduler/test_lazy_scheduler.py::TestLazyScheduler::test_explicit_schedule
pytest -vs test/lazy_scheduler/test_lazy_scheduler.py::TestLazyScheduler::test_explicit_schedule_reordering
pytest -vs test/lazy_scheduler/test_lazy_scheduler.py::TestLazyScheduler::test_register_segment_hook
"""

import types
import torch
import torch.utils._pytree as pytree
from torch.testing._internal.common_utils import TestCase as TorchTestCase
from torch._dynamo import disable
import functools
import itertools
from typing import Any, Optional, Dict, Callable, List
from torch._subclasses.async_tensor import AsyncTensor, fake_mode
from collections import defaultdict, OrderedDict
import weakref
import threading
from torch.utils._python_dispatch import return_and_correct_aliasing
from torch._dynamo.backends.common import aot_autograd
from torch._inductor.compile_fx import compile_fx as inductor_compile_fx
from torch._inductor.compile_fx import compile_fx_inner as inductor_compile_fx_inner
from torch._dynamo.output_graph import GraphCompileReason
from torch._lazy_scheduler import LazyScheduler, Segment


class TestCase(TorchTestCase):
  def setUp(self):
    torch._dynamo.reset()
    super().setUp()

  def tearDown(self):
    super().tearDown()
    torch._dynamo.reset()


class TestDepSegmentModule(torch.nn.Module):
  """
  Dependency chain:
  func1 -> func2 -> mul -> output
  """
  def __init__(self):
    super().__init__()

  def func1(self, x, y):
    return torch.matmul(x, y)

  def func2(self, x, y):
    return torch.add(x, y)

  def forward(self, x, y):
    z1 = self.func1(x, y)
    z2 = self.func2(x, z1)
    z = z1 * z2
    return z


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


class TestLazyScheduler(TestCase):
  def _validate(self, orig_eager_fn, lazy_scheduler_gen, expected_exec_order, inps, skip_check=False, test_eager=True, test_compile=True):
    def _clone_inps():
      cloned_inps = []
      for inp in inps:
        cloned_inps.append(inp.clone().detach().requires_grad_(inp.requires_grad))
      return cloned_inps

    def _compare_output_and_grad(orig_fn, lazy_scheduler):
      inps_no_ls = _clone_inps()
      inps_ls = _clone_inps()

      # Original function, 1st iter
      torch.manual_seed(0)
      expected = orig_fn(*inps_no_ls)
      expected.sum().backward()

      # Original function, 2nd iter
      torch.manual_seed(0)
      expected = orig_fn(*inps_no_ls)
      expected.sum().backward()

      # LazyScheduler function, 1st iter
      torch.manual_seed(0)
      result = lazy_scheduler(*inps_ls)
      result.sum().backward()

      # LazyScheduler function, 2nd iter
      torch.manual_seed(0)
      # Reset state so that we only track the execution order of the 2nd iter
      lazy_scheduler.reset_recorded_exec_order()
      result = lazy_scheduler(*inps_ls)
      result.sum().backward()

      if not skip_check:
        self.assertEqual(
          result,
          expected,
          msg="Output mismatch between torch.compile and eager versions",
        )
        for inp, cloned_inp in zip(inps_no_ls, inps_ls):
          self.assertEqual(
            inp.grad,
            cloned_inp.grad,
            msg=f"Gradient mismatch between torch.compile and eager versions. inp.grad: {inp.grad}, cloned_inp.grad: {cloned_inp.grad}",
          )

    def _compare_exec_order(recorded_exec_order, expected_exec_order):
      err_msg = f"""
Expected execution order to be:
{expected_exec_order},

but got:
{lazy_scheduler._recorded_exec_order}
"""
      self.assertEqual(len(recorded_exec_order), len(expected_exec_order), msg=err_msg)
      self.assertEqual(lazy_scheduler._recorded_exec_order, expected_exec_order, msg=err_msg)

    if test_eager:
      lazy_scheduler = lazy_scheduler_gen()
      _compare_output_and_grad(orig_eager_fn, lazy_scheduler)
      # NOTE: In eager mode LazyScheduler, unnamed segment is not annotated with prefix "__unnamed_" (they are not annotated at all).
      # TODO: need to differentiate between "known backward segment's unknown forward segment" and "truly unknown segment".
      # expected_exec_order_without_unnamed_segments = [s for s in expected_exec_order if not s.startswith("__unnamed_")]
      # _compare_exec_order(lazy_scheduler._recorded_exec_order, expected_exec_order_without_unnamed_segments)
      _compare_exec_order(lazy_scheduler._recorded_exec_order, expected_exec_order)

    if test_compile:
      lazy_scheduler = lazy_scheduler_gen()
      _compare_output_and_grad(
        torch.compile(orig_eager_fn, fullgraph=False, backend="inductor"),
        torch.compile(lazy_scheduler, fullgraph=False, backend="inductor"),
      )
      _compare_exec_order(lazy_scheduler._recorded_exec_order, expected_exec_order)

  def test_single_segment_prefix_fwd(self):
    # Check that output and gradients are correct when there is
    # only one unnamed segment in the model.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = TestDepSegmentModule()
    m = m.to(device)
    x = torch.randn(4, 4, requires_grad=True, device=device)
    y = torch.randn(4, 4, requires_grad=True, device=device)

    lazy_scheduler_gen = lambda: LazyScheduler(
      m,
      segments=[
        Segment("func1_fwd", m.func1),
      ],
      schedule=[
        "func1_fwd",
      ],
    )

    self._validate(
      m,
      lazy_scheduler_gen,
      # lazy_scheduler._compile_fx,
      expected_exec_order=[
        "func1_fwd",
        "__unregistered_func1_bwd",
      ],
      inps=[x, y],
    )

  def test_single_segment_prefix_fwd_bwd(self):
    # Check that output and gradients are correct when there is
    # only one unnamed segment in the model.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = TestDepSegmentModule()
    m = m.to(device)
    x = torch.randn(4, 4, requires_grad=True, device=device)
    y = torch.randn(4, 4, requires_grad=True, device=device)

    lazy_scheduler_gen = lambda: LazyScheduler(
      m,
      segments=[
        Segment("func1_fwd", m.func1),
        Segment("func1_bwd", m.func1, is_backward=True),
      ],
      schedule=[
        "func1_fwd",
        "func1_bwd",
      ],
    )

    self._validate(
      m,
      lazy_scheduler_gen,
      # lazy_scheduler._compile_fx,
      expected_exec_order=[
        "func1_fwd",
        "func1_bwd",
      ],
      inps=[x, y],
    )

  def test_split_module_dep_segments(self):
    # Check that GraphModule produced by Dynamo is correctly split
    # (each submodule only contains one segment)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = TestDepSegmentModule()
    m = m.to(device)
    x = torch.randn(4, 4, requires_grad=True, device=device)
    y = torch.randn(4, 4, requires_grad=True, device=device)
    # lazy_scheduler = LazyScheduler(schedule=[])

    # def segment_prefix_assignment_fn(gm):
    #   for node in gm.graph.nodes:
    #     assert "nn_module_method" in node.meta
    #     # One NN module method maps to one named segment
    #     node.meta["segment_prefix"] = str(node.meta["nn_module_method"])

    # torch._dynamo.config.lazy_scheduler_compile_fn = functools.partial(
    #   lazy_scheduler._compile_fn,
    #   segment_prefix_assignment_fn=segment_prefix_assignment_fn
    # )

    lazy_scheduler_gen = lambda: LazyScheduler(
      m,
      segments=[
        Segment("func1_fwd", m.func1),
        Segment("func2_fwd", m.func2),
        Segment("forward_fwd", m.forward),
        Segment("func1_bwd", m.func1, is_backward=True),
        Segment("func2_bwd", m.func2, is_backward=True),
        Segment("forward_bwd", m.forward, is_backward=True),
      ],
      schedule=[
        "func1_fwd",
        "func2_fwd",
        "forward_fwd",
        "forward_bwd",
        "func2_bwd",
        "func1_bwd",
      ],
    )

    self._validate(
      m,
      lazy_scheduler_gen,
      # lazy_scheduler._compile_fx,
      expected_exec_order=[
        "func1_fwd",
        "func2_fwd",
        "forward_fwd",
        "forward_bwd",
        "func2_bwd",
        "func1_bwd",
      ],
      inps=[x, y],
    )

  def test_split_module_non_dep_segments(self):
    # Check that GraphModule produced by Dynamo is correctly split
    # (each submodule only contains one segment)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = TestNonDepSegmentModule()
    m = m.to(device)
    x = torch.randn(4, 4, requires_grad=True, device=device)
    y = torch.randn(4, 4, requires_grad=True, device=device)
    # lazy_scheduler = LazyScheduler(schedule=[])

    # def segment_prefix_assignment_fn(gm):
    #   for node in gm.graph.nodes:
    #     assert "nn_module_method" in node.meta
    #     # One NN module method maps to one named segment
    #     node.meta["segment_prefix"] = str(node.meta["nn_module_method"])

    # torch._dynamo.config.lazy_scheduler_compile_fn = functools.partial(
    #   lazy_scheduler._compile_fn,
    #   segment_prefix_assignment_fn=segment_prefix_assignment_fn
    # )

    lazy_scheduler_gen = lambda: LazyScheduler(
      m,
      segments={
        Segment("func1_fwd", m.func1),
        Segment("func2_fwd", m.func2),
        Segment("forward_fwd", m.forward),
        Segment("func1_bwd", m.func1, is_backward=True),
        Segment("func2_bwd", m.func2, is_backward=True),
        Segment("forward_bwd", m.forward, is_backward=True),
      },
      schedule=[
        "func1_fwd",
        "func2_fwd",
        "forward_fwd",
        "forward_bwd",
        "func2_bwd",
        "func1_bwd",
      ],
    )

    self._validate(
      m,
      lazy_scheduler_gen,
      # lazy_scheduler._compile_fx,
      expected_exec_order=[
        "func1_fwd",
        "func2_fwd",
        "forward_fwd",
        "forward_bwd",
        "func2_bwd",
        "func1_bwd",
      ],
      inps=[x, y],
    )

  def test_segment_tagging(self):
    def _run_test(orig_fn, ls_fn, expected_exec_order):
      # torch._dynamo.config.lazy_scheduler_compile_fn = functools.partial(
      #   lazy_scheduler._compile_fn,
      #   segment_prefix_assignment_fn=functools.partial(segment_prefix_assignment_fn, segment_dict=segment_dict),
      # )

      # self._validate(
      #   m,
      #   lazy_scheduler._compile_fx,
      #   x,
      #   y,
      # )

      x = torch.randn(4, 4, requires_grad=True, device=device)
      y = torch.randn(4, 4, requires_grad=True, device=device)

      self._validate(
        orig_fn,
        ls_fn,
        # lazy_scheduler._compile_fx,
        expected_exec_order=expected_exec_order,
        inps=[x, y],
      )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Tag fwd segments
    m = TestNonDepSegmentModule()
    m = m.to(device)
    lazy_scheduler_gen = lambda: LazyScheduler(m, segments=[Segment("func2_fwd", m.func2)])
    _run_test(m, lazy_scheduler_gen, expected_exec_order=['__unnamed_0_fwd', 'func2_fwd', '__unnamed_1_fwd', '__unnamed_1_bwd', '__unnamed_2_bwd', '__unnamed_0_bwd'])

    m = TestNonDepSegmentModule()
    m = m.to(device)
    lazy_scheduler_gen = lambda: LazyScheduler(
      m,
      segments=[
        Segment("func1_fwd", m.func1),
        Segment("func2_fwd", m.func2),
      ],
    )
    _run_test(m, lazy_scheduler_gen, expected_exec_order=['func1_fwd', 'func2_fwd', '__unnamed_0_fwd', '__unnamed_0_bwd', '__unnamed_1_bwd', '__unnamed_2_bwd'])

    # Tag bwd segments
    m = TestNonDepSegmentModule()
    m = m.to(device)
    lazy_scheduler_gen = lambda: LazyScheduler(m, segments=[Segment("func2_bwd", m.func2, is_backward=True)])
    _run_test(m, lazy_scheduler_gen, expected_exec_order=['__unnamed_0_fwd', '__unnamed_1_fwd', '__unnamed_2_fwd', '__unnamed_2_bwd', 'func2_bwd', '__unnamed_0_bwd'])

    m = TestNonDepSegmentModule()
    m = m.to(device)
    lazy_scheduler_gen = lambda: LazyScheduler(
      m,
      segments=[
        Segment("func1_bwd", m.func1, is_backward=True),
        Segment("func2_bwd", m.func2, is_backward=True),
      ],
    )
    _run_test(m, lazy_scheduler_gen, expected_exec_order=['__unnamed_0_fwd', '__unnamed_1_fwd', '__unnamed_2_fwd', '__unnamed_2_bwd', 'func2_bwd', 'func1_bwd'])

  def test_explicit_schedule(self):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = TestNonDepSegmentModule()
    m = m.to(device)
    x = torch.randn(4, 4, requires_grad=True, device=device)
    y = torch.randn(4, 4, requires_grad=True, device=device)

    lazy_scheduler = LazyScheduler(
      m,
      segments=[
        Segment("func1_fwd", m.func1),
        Segment("func2_fwd", m.func2),
        Segment("func1_bwd", m.func1, is_backward=True),
        Segment("func2_bwd", m.func2, is_backward=True),
      ],
      # This is the explicit schedule (i.e. execution order)
      schedule=["func1_fwd", "func2_fwd", "func2_bwd", "func1_bwd"],
    )

    # register_segment(segment_dict, m.func1, "func1")
    # register_segment(segment_dict, m.func2, "func2")

    # torch._dynamo.config.lazy_scheduler_compile_fn = functools.partial(
    #   lazy_scheduler._compile_fn,
    #   segment_prefix_assignment_fn=functools.partial(segment_prefix_assignment_fn, segment_dict=segment_dict),
    # )

    self._validate(
      m,
      lazy_scheduler,
      # lazy_scheduler._compile_fx,
      expected_exec_order=['func1_fwd', 'func2_fwd', '__unnamed_0_fwd', '__unnamed_0_bwd', 'func2_bwd', 'func1_bwd'],
      inps=[x, y],
    )

  def test_explicit_schedule_reordering(self):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = TestNonDepSegmentModule()
    m = m.to(device)
    x = torch.randn(4, 4, requires_grad=True, device=device)
    y = torch.randn(4, 4, requires_grad=True, device=device)

    # # This is roughly how the official register_segment function will look like
    # def register_segment(segment_dict, method, name):
    #   segment_dict[method] = name

    # segment_dict = {}
    # register_segment(segment_dict, m.func1, "func1")
    # register_segment(segment_dict, m.func2, "func2")

    # torch._dynamo.config.lazy_scheduler_compile_fn = functools.partial(
    #   lazy_scheduler._compile_fn,
    #   segment_prefix_assignment_fn=functools.partial(segment_prefix_assignment_fn, segment_dict=segment_dict),
    # )

    lazy_scheduler = LazyScheduler(
      m,
      segments=[
        Segment("func1_fwd", m.func1),
        Segment("func2_fwd", m.func2),
        Segment("func1_bwd", m.func1, is_backward=True),
        Segment("func2_bwd", m.func2, is_backward=True),
      ],
      # This is the explicit schedule (i.e. execution order).
      # Notice we force func2_fwd to run before func1_fwd.
      schedule=["func2_fwd", "func1_fwd", "func2_bwd", "func1_bwd"],
    )

    self._validate(
      m,
      lazy_scheduler,
      # lazy_scheduler._compile_fx,
      expected_exec_order=['func2_fwd', 'func1_fwd', '__unnamed_0_fwd', '__unnamed_0_bwd', 'func2_bwd', 'func1_bwd'],
      inps=[x, y],
    )

  def DISABLED_test_register_segment_hook(self):
    # Use segment hook instead of explicit schedule to specify the execution order
    """
    class SDDModule(nn.Module):
      def forward(self, x):
        return dist.all_to_all(x, …)

    class OverArch(nn.Module):
      def func1(self, x):
        return torch.relu(x)

      def func2(self, x):
        return torch.matmul(x, x)

      def forward(self, x):
        x = self.func1(x)
        x = self.func2(x)
        return x

    class Model(nn.Module):
      def __init__(self):
        super().__init__()
        self.sdd = SDDModule()
        self.overarch = OverArch()

      def __forward__(self, x):
        self.sdd(x)
        output = self.overarch(x)
        return output

    model = Model()

    # Eager mode
    model_ls = LazyScheduler(
      model,
      # Create NN module method to segment mapping.
      segments=[
        Segment("sdd_fwd", model.sdd.forward, nth_call=0),
        Segment("overarch_func2_bwd", model.overarch.func2, is_backward=True, nth_call=0),
      ],
      # Run "sdd_fwd" right before "overarch_func2_bwd".
      schedule=["sdd_fwd", "overarch_func2_bwd"],
    )
    output = model_ls(inputs)

    # Compile mode
    model_c = LazyScheduler(
      model,
      segments=[
        Segment("sdd_fwd", model.sdd.forward, nth_call=0),
        Segment("overarch_func2_bwd", model.overarch.func2, is_backward=True, nth_call=0),
      ],
      schedule=["sdd_fwd", "overarch_func2_bwd"],
      compile_options={
        "fullgraph": False,
        "backend": "inductor",
      }
    )
    output = model_c(inputs)
    """
    pass

"""
TODO:
- Eager mode prototype, merge with compile mode prototype. Combine the unit tests. (use https://docs.google.com/document/d/1vv0H5IMGwUMyzmJKnksJOnRSult1B4YlbBSs_MeAvXM/edit?usp=sharing as design source of truth)
- Support user calling a method multiple times and only tag a specific call as segment (i.e. make `nth_call=X` work)
- Unit test: graph break within segment (i.e. multiple graphs per segment)
- Unit test: in-place op in named segment
- For named segments, show its segment ID (prefix + fwd/bwd + nth_call) in profiler annotation in GPU trace
- Integration with DDPOptimizer
- Integration with FSDP (graph break, not tracing)
- Integration with activation checkpointing
- Implement "fall back when there is side effect in delayed region detected by Dynamo”
- what if a segment is in the schedule but is never run due to dynamic control flow change? we should gracefully fall back to no-scheduler mode
- Try on Ads model: https://docs.google.com/document/d/1tFLUh4Xe4_eGKOtgpj08kfNDhy7Fqp-dSq0d7lejdZU/edit#bookmark=id.wds06wiqwjh2 figure out integration point with trainer loop
- (Later) Integration with compiled autograd
- Logging for better user debugging (what is scheduled and when, and the compile output). Look at the generated graph and the original code.
- Also log memory usage, how much memory I am keeping above.

NOTE:
- Even if the segment is only in the backward, its corresponding forward segment will also be carved out (graph-break'ed).

AsyncTensor specific:
- Support kwargs in AsyncTensor ops
- Support tuple / list / etc. in AsyncTensor op args
- Support all factory functions in AsyncTensor dispatch
"""

if __name__ == "__main__":
  from torch._dynamo.test_case import run_tests
  run_tests()


"""
FAQ
Q1: What happens if we have a user-defined segment deep down in a submodule?
Answer: everything before the defined segment will be in their own segment. Everything after is in another segment.
"""
