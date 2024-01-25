"""
CUDA_VISIBLE_DEVICES=7 pytest -vs test/lazy_scheduler/test_lazy_scheduler.py::TestLazyScheduler::test_single_segment_prefix_fwd && \
CUDA_VISIBLE_DEVICES=7 pytest -vs test/lazy_scheduler/test_lazy_scheduler.py::TestLazyScheduler::test_single_segment_prefix_fwd_bwd && \
CUDA_VISIBLE_DEVICES=7 pytest -vs test/lazy_scheduler/test_lazy_scheduler.py::TestLazyScheduler::test_nested_segments_dep && \
CUDA_VISIBLE_DEVICES=7 pytest -vs test/lazy_scheduler/test_lazy_scheduler.py::TestLazyScheduler::test_nested_segments_non_dep && \
CUDA_VISIBLE_DEVICES=7 pytest -vs test/lazy_scheduler/test_lazy_scheduler.py::TestLazyScheduler::test_explicit_schedule_reordering && \
CUDA_VISIBLE_DEVICES=7 pytest -vs test/lazy_scheduler/test_lazy_scheduler.py::TestLazyScheduler::test_segment_compiled_with_different_backend && \
CUDA_VISIBLE_DEVICES=7 pytest -vs test/lazy_scheduler/test_lazy_scheduler.py::TestLazyScheduler::test_graph_break_within_segment
 >output.log 2>&1
"""

import types
import copy
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


def extract_ops_from_gm(gm):
    ops = []
    for node in gm.graph.nodes:
      if node.op == "call_function":
        ops.append(node.target.__name__)
      elif node.op == "call_method":
        ops.append(node.target)
    return ops


def check_segment(lazy_scheduler, segment_name_to_expected_ops_in_gms, is_compile):
  segment_to_gms_map = lazy_scheduler._segment_to_gms_map
  for segment_name, expected_ops_in_gms in segment_name_to_expected_ops_in_gms.items():
    assert segment_name in segment_to_gms_map, f"{segment_name} should exist in segment_to_gms_map but it doesn't."
    assert len(segment_to_gms_map[segment_name]) == len(expected_ops_in_gms)
    for gm, expected_ops_dict_or_list in zip(segment_to_gms_map[segment_name], expected_ops_in_gms):
      if isinstance(expected_ops_dict_or_list, dict):
        expected_ops = expected_ops_dict_or_list["compiled"] if is_compile else expected_ops_dict_or_list["eager"]
      else:
        expected_ops = expected_ops_dict_or_list
      ops_from_gm = extract_ops_from_gm(gm)
      print(f"ops_from_gm: {ops_from_gm}")
      assert ops_from_gm == expected_ops


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


def check_segment_for_TestDepSegmentModule_fwd_bwd(lazy_scheduler, is_compile=False):
  return check_segment(
    lazy_scheduler,
    {
      "func1_fwd": [{
        "eager": ['mm.default', 't.default', 't.default'],
        "compiled": ['mm.default', 'permute.default', 'permute.default'],
      }],
      "func2_fwd": [['add.Tensor']],
      "forward_fwd": [['mul.Tensor']],
      "forward_bwd": [['mul.Tensor', 'mul.Tensor']],
      "func2_bwd": [[]],
      "func1_bwd": [['mm.default', 'mm.default']],
    },
    is_compile=is_compile,
  )


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


def check_segment_for_TestNonDepSegmentModule_fwd_bwd(lazy_scheduler, is_compile=False):
  return check_segment(
    lazy_scheduler,
    {
      "func1_fwd": [{
        "eager": ['mm.default', 't.default', 't.default'],
        "compiled": ['mm.default', 'permute.default', 'permute.default'],
      }],
      "func2_fwd": [['add.Tensor']],
      "forward_fwd": [['mul.Tensor']],
      "forward_bwd": [['mul.Tensor', 'mul.Tensor']],
      "func2_bwd": [[]],
      "func1_bwd": [['mm.default', 'mm.default']],
    },
    is_compile=is_compile,
  )


class TestLazyScheduler(TestCase):
  def _validate(self, eager_module, lazy_scheduler_gen, expected_exec_order, inps, fwd_only=False, skip_check=False, test_eager=True, test_compile=True, additional_check=None):
    def _clone_inps():
      cloned_inps = []
      for inp in inps:
        cloned_inps.append(inp.clone().detach().requires_grad_(inp.requires_grad))
      return cloned_inps

    def _compare_output_and_grad(eager_module, lazy_scheduler_gen, is_compile=False, additional_check=None):
      inps_no_ls = _clone_inps()
      inps_ls = _clone_inps()

      if is_compile:
        baseline_fn = torch.compile(eager_module, fullgraph=False, backend="inductor")
      else:
        baseline_fn = eager_module

      num_iterations = 1  # 2

      # Original function, 2 iterations
      for i in range(num_iterations):
        print(f"------------------ eager iter: {i} ------------------")
        torch.manual_seed(0)
        expected = baseline_fn(*inps_no_ls)
        if not fwd_only:
          expected.sum().backward()

      # LazyScheduler function, 2 iterations
      lazy_scheduler = None
      for i in range(num_iterations):
        print(f"------------------ LazyScheduler iter: {i} ------------------")
        torch.manual_seed(0)
        # TODO: currently we re-create LazyScheduler for every iteration, so that the scheduler state is always fresh.
        # If this is a lot of runtime overhead, we can explore reusing old LazyScheduler instance.
        lazy_scheduler = lazy_scheduler_gen(eager_module, is_compile=is_compile)
        result = lazy_scheduler(*inps_ls)
        if not fwd_only:
          result.sum().backward()
        print(f"here1 lazy_scheduler._recorded_exec_order: {lazy_scheduler._recorded_exec_order}")
        print(f"here1 id(lazy_scheduler): {id(lazy_scheduler)}")
        print(f"here1 lazy_scheduler._registered_segment_prefixes: {lazy_scheduler._registered_segment_prefixes}")
        print(f"here1 lazy_scheduler._segment_to_gms_map: {lazy_scheduler._segment_to_gms_map}")
        print(f"here1 id(lazy_scheduler._segment_to_gms_map): {id(lazy_scheduler._segment_to_gms_map)}")
        if hasattr(eager_module, "func1"):
          print(f"here1 id(eager_module.func1): {id(eager_module.func1)}")
        if hasattr(eager_module, "func2"):
          print(f"here1 id(eager_module.func2): {id(eager_module.func2)}")
        if hasattr(eager_module, "forward"):
          print(f"here1 id(eager_module.forward): {id(eager_module.forward)}")

      if not skip_check:
        self.assertEqual(
          result,
          expected,
          msg="Output mismatch between torch.compile and eager versions",
        )
        if not fwd_only:
          for inp, cloned_inp in zip(inps_no_ls, inps_ls):
            self.assertEqual(
              inp.grad,
              cloned_inp.grad,
              msg=f"Gradient mismatch between torch.compile and eager versions. inp.grad: {inp.grad}, cloned_inp.grad: {cloned_inp.grad}",
            )

      if additional_check is not None:
        if not isinstance(additional_check, list):
          additional_check = [additional_check]
        for check in additional_check:
          check(lazy_scheduler, is_compile=is_compile)

      recorded_exec_order = lazy_scheduler._recorded_exec_order
      err_msg = f"""
Expected execution order to be:
{expected_exec_order},

but got:
{recorded_exec_order}
"""
      # NOTE: We don't care about the execution order of unnamed segments.
      recorded_exec_order_without_unnamed_segments = [s for s in recorded_exec_order if not s.startswith("__unnamed_")]
      self.assertEqual(len(recorded_exec_order_without_unnamed_segments), len(expected_exec_order), msg=err_msg)
      self.assertEqual(recorded_exec_order_without_unnamed_segments, expected_exec_order, msg=err_msg)

    if test_eager:
      _compare_output_and_grad(copy.deepcopy(eager_module), lazy_scheduler_gen, is_compile=False, additional_check=additional_check)
      torch._dynamo.reset()
      print(f"Eager mode test done!")

    if test_compile:
      _compare_output_and_grad(copy.deepcopy(eager_module), lazy_scheduler_gen, is_compile=True, additional_check=additional_check)
      torch._dynamo.reset()
      print(f"Compile mode test done!")

  def test_single_segment_prefix_fwd(self):
    # Check that output and gradients are correct when there is
    # only one unnamed segment in the model.
    device = "cuda"
    m = TestDepSegmentModule()
    m = m.to(device)
    x = torch.randn(4, 4, requires_grad=True, device=device)
    y = torch.randn(4, 4, requires_grad=True, device=device)

    def lazy_scheduler_gen(module, is_compile=False):
      return LazyScheduler(
        module,
        segments=[
          Segment("func1_fwd", module.func1),
        ],
        schedule=[
          "func1_fwd",
        ],
        compile_options=None if not is_compile else {
          "fullgraph": False,
          "backend": "inductor",
        }
      )

    self._validate(
      m,
      lazy_scheduler_gen,
      expected_exec_order=[
        "func1_fwd",
        "__unregistered_func1_bwd",
      ],
      inps=[x, y],
    )

  def test_single_segment_prefix_fwd_bwd(self):
    # Check that output and gradients are correct when there is
    # only one unnamed segment in the model.
    device = "cuda"
    m = TestDepSegmentModule()
    m = m.to(device)
    x = torch.randn(4, 4, requires_grad=True, device=device)
    y = torch.randn(4, 4, requires_grad=True, device=device)

    def lazy_scheduler_gen(module, is_compile=False):
      return LazyScheduler(
        module,
        segments=[
          Segment("func1_fwd", module.func1),
          Segment("func1_bwd", module.func1),
        ],
        schedule=[
          "func1_fwd",
          "func1_bwd",
        ],
        compile_options=None if not is_compile else {
          "fullgraph": False,
          "backend": "inductor",
        }
      )

    self._validate(
      m,
      lazy_scheduler_gen,
      expected_exec_order=[
        "func1_fwd",
        "func1_bwd",
      ],
      inps=[x, y],
    )

  def test_nested_segments_dep(self):
    # Check that GraphModule produced by Dynamo is correctly split
    # (each submodule only contains one segment)
    device = "cuda"
    m = TestDepSegmentModule()
    m = m.to(device)
    x = torch.randn(4, 4, requires_grad=True, device=device)
    y = torch.randn(4, 4, requires_grad=True, device=device)

    expected_exec_order = [
      "func1_fwd",
      "func2_fwd",
      "forward_fwd",
      "forward_bwd",
      "func2_bwd",
      "func1_bwd",
    ]

    def lazy_scheduler_gen(module, is_compile=False):
      return LazyScheduler(
        module,
        segments=[
          Segment("func1_fwd", module.func1),
          Segment("func2_fwd", module.func2),
          Segment("forward_fwd", module.forward),
          Segment("func1_bwd", module.func1),
          Segment("func2_bwd", module.func2),
          Segment("forward_bwd", module.forward),
        ],
        schedule=expected_exec_order,
        compile_options=None if not is_compile else {
          "fullgraph": False,
          "backend": "inductor",
        },
      )

    self._validate(
      m,
      lazy_scheduler_gen,
      expected_exec_order=expected_exec_order,
      inps=[x, y],
      additional_check=check_segment_for_TestDepSegmentModule_fwd_bwd,
    )

  def test_nested_segments_non_dep(self):
    # Check that GraphModule produced by Dynamo is correctly split
    # (each submodule only contains one segment)
    device = "cuda"
    m = TestNonDepSegmentModule()
    m = m.to(device)
    x = torch.randn(4, 4, requires_grad=True, device=device)
    y = torch.randn(4, 4, requires_grad=True, device=device)

    expected_exec_order = [
      "func1_fwd",
      "func2_fwd",
      "forward_fwd",
      "forward_bwd",
      "func2_bwd",
      "func1_bwd",
    ]

    def lazy_scheduler_gen(module, is_compile=False):
      return LazyScheduler(
        module,
        segments=[
          Segment("func1_fwd", module.func1),
          Segment("func2_fwd", module.func2),
          Segment("forward_fwd", module.forward),
          Segment("func1_bwd", module.func1),
          Segment("func2_bwd", module.func2),
          Segment("forward_bwd", module.forward),
        ],
        schedule=expected_exec_order,
        compile_options=None if not is_compile else {
          "fullgraph": False,
          "backend": "inductor",
        },
      )

    self._validate(
      m,
      lazy_scheduler_gen,
      expected_exec_order=expected_exec_order,
      inps=[x, y],
      additional_check=check_segment_for_TestNonDepSegmentModule_fwd_bwd,
    )

  def test_explicit_schedule_reordering(self):
    device = "cuda"
    m = TestNonDepSegmentModule()
    m = m.to(device)
    x = torch.randn(4, 4, requires_grad=True, device=device)
    y = torch.randn(4, 4, requires_grad=True, device=device)

    expected_exec_order = [
      "func2_fwd",
      "func1_fwd",
      "forward_fwd",
      "forward_bwd",
      "func2_bwd",
      "func1_bwd",
    ]

    def lazy_scheduler_gen(module, is_compile=False):
      return LazyScheduler(
        module,
        segments=[
          Segment("func1_fwd", module.func1),
          Segment("func2_fwd", module.func2),
          Segment("forward_fwd", module.forward),
          Segment("func1_bwd", module.func1),
          Segment("func2_bwd", module.func2),
          Segment("forward_bwd", module.forward),
        ],
        schedule=expected_exec_order,
        compile_options=None if not is_compile else {
          "fullgraph": False,
          "backend": "inductor",
        },
      )

    self._validate(
      m,
      lazy_scheduler_gen,
      expected_exec_order=expected_exec_order,
      inps=[x, y],
      additional_check=check_segment_for_TestNonDepSegmentModule_fwd_bwd,
    )

  def test_segment_compiled_with_different_backend(self):
    def _run_test(lazy_scheduler_gen, expected_exec_order, additional_check, fwd_only=False):
      device = "cuda"
      m = TestNonDepSegmentModule()
      m = m.to(device)
      x = torch.randn(4, 4, requires_grad=True, device=device)
      y = torch.randn(4, 4, requires_grad=True, device=device)

      self._validate(
        m,
        lazy_scheduler_gen,
        expected_exec_order=expected_exec_order,
        inps=[x, y],
        fwd_only=fwd_only,
        additional_check=additional_check,
      )

    def segment_use_dynamo_eager_fwd_only():
      expected_exec_order = [
        "func2_fwd",
        "func1_fwd",
        "forward_fwd",
      ]
      def check_segment_fwd_only(lazy_scheduler, is_compile=False):
        return check_segment(
          lazy_scheduler,
          {
            "func1_fwd": [['matmul']],
            "func2_fwd": [['add.Tensor']],
            "forward_fwd": [['mul.Tensor']],
          },
          is_compile=is_compile,
        )
      def _lazy_scheduler_gen(module, is_compile=False):
        return LazyScheduler(
          module,
          segments=[
            Segment("func1_fwd", module.func1, backend="eager"),
            Segment("func2_fwd", module.func2, backend="aot_eager"),
            Segment("forward_fwd", module.forward, backend="inductor"),
          ],
          schedule=expected_exec_order,
          compile_options=None if not is_compile else {
            "fullgraph": False,
            "backend": "inductor",
          },
        )
      return _lazy_scheduler_gen, expected_exec_order, check_segment_fwd_only

    def segment_use_dynamo_aot_eager_fwd_bwd():
      expected_exec_order = [
        "func2_fwd",
        "func1_fwd",
        "forward_fwd",
        "forward_bwd",
        "func2_bwd",
        "func1_bwd",
      ]
      def check_segment_fwd_bwd(lazy_scheduler, is_compile=False):
        return check_segment(
          lazy_scheduler,
          {
            "func1_fwd": [['mm.default', 't.default', 't.default']],
            "func2_fwd": [['add.Tensor']],
            "forward_fwd": [['mul.Tensor']],
            "forward_bwd": [['mul.Tensor', 'mul.Tensor']],
            "func2_bwd": [[]],
            "func1_bwd": [['mm.default', 'mm.default']],
          },
          is_compile=is_compile,
        )
      def _lazy_scheduler_gen(module, is_compile=False):
        return LazyScheduler(
          module,
          segments=[
            Segment("func1_fwd", module.func1, backend="aot_eager"),
            Segment("func2_fwd", module.func2, backend="aot_eager"),
            Segment("forward_fwd", module.forward, backend="inductor"),
            Segment("func1_bwd", module.func1, backend="aot_eager"),
            Segment("func2_bwd", module.func2, backend="aot_eager"),
            Segment("forward_bwd", module.forward, backend="inductor"),
          ],
          schedule=expected_exec_order,
          compile_options=None if not is_compile else {
            "fullgraph": False,
            "backend": "inductor",
          },
        )
      return _lazy_scheduler_gen, expected_exec_order, check_segment_fwd_bwd

    _run_test(*segment_use_dynamo_eager_fwd_only(), fwd_only=True)
    _run_test(*segment_use_dynamo_aot_eager_fwd_bwd(), fwd_only=False)


  def test_graph_break_within_segment(self):
    """
    - Unit test: graph break within segment (i.e. multiple graphs per segment), either in the delayed segment or in the anchored segment
      - In the delayed segment case, also add output usage within the graph break eager region, to trigger the immediate materialization of AsyncTensor output
      - Make sure to assert that each GM contains the ops you expect.
    """
    # TODO: maybe unify with other `_run_test` functions
    def _run_test(lazy_scheduler_gen=None, expected_exec_order=None, mod_class=None, additional_check=None):
      device = "cuda"
      m = mod_class()
      m = m.to(device)
      x = torch.randn(4, 4, requires_grad=True, device=device)
      y = torch.randn(4, 4, requires_grad=True, device=device)

      self._validate(
        m,
        lazy_scheduler_gen,
        expected_exec_order=expected_exec_order,
        inps=[x, y],
        additional_check=additional_check,
      )

    expected_exec_order = [
      "func2_fwd",
      "func1_fwd",
      "forward_fwd",
      "forward_bwd",
      "func2_bwd",
      "func1_bwd",
    ]
    def check_segment_fwd_bwd(lazy_scheduler, is_compile=False):
      return check_segment(
        lazy_scheduler,
        {
          "func1_fwd": [['mm.default', 't.default', 't.default']],
          "func2_fwd": [['add.Tensor']],
          "forward_fwd": [['mul.Tensor']],
          "forward_bwd": [['mul.Tensor', 'mul.Tensor']],
          "func2_bwd": [[]],
          "func1_bwd": [['mm.default', 'mm.default']],
        },
        is_compile=is_compile,
      )
    def _lazy_scheduler_gen(module, is_compile=False):
      return LazyScheduler(
        module,
        segments=[
          Segment("func1_fwd", module.func1, backend="aot_eager"),
          Segment("func2_fwd", module.func2, backend="aot_eager"),
          Segment("forward_fwd", module.forward, backend="inductor"),
          Segment("func1_bwd", module.func1, backend="aot_eager"),
          Segment("func2_bwd", module.func2, backend="aot_eager"),
          Segment("forward_bwd", module.forward, backend="inductor"),
        ],
        schedule=expected_exec_order,
        compile_options=None if not is_compile else {
          "fullgraph": False,
          "backend": "inductor",
        },
      )

    def segment_has_graph_break_not_using_async_tensor_output():
      class TestModule(torch.nn.Module):
        def __init__(self):
          super().__init__()

        def func1(self, x, y):
          y2 = torch.matmul(x, y)
          print("y2")  # guaranteed graph-break
          return torch.matmul(x, y2)

        def func2(self, x, y):
          y3 = torch.add(x, y)
          print("y3")  # guaranteed graph-break
          return torch.add(x, y3)

        def forward(self, x, y):
          z1 = self.func1(x, y)
          z2 = self.func2(x, y)
          z = z1 * z2
          return z

      return {
        "mod_class": TestModule,
        "additional_check": None,
      }

    def segment_has_graph_break_using_async_tensor_output():
      global_dict = {}
      class TestModule(torch.nn.Module):
        def __init__(self):
          super().__init__()

        def func1(self, x, y):
          global global_dict
          y2 = torch.matmul(x, y)
          global_dict["y2_sum"] = y2.sum()
          print(global_dict["y2_sum"])  # guaranteed graph-break
          return torch.matmul(x, y2)

        def func2(self, x, y):
          global global_dict
          y3 = torch.add(x, y)
          global_dict["y3_sum"] = y3.sum()
          print(global_dict["y3_sum"])  # guaranteed graph-break
          return torch.add(x, y3)

        def forward(self, x, y):
          z1 = self.func1(x, y)
          z2 = self.func2(x, y)
          z = z1 * z2
          return z

      def check_async_tensor_is_early_scheduled(lazy_scheduler, is_compile=False):
        # TODO how to check that AsyncTensor is early scheduled? maybe check the recorded execution order?
        pass

      return {
        "mod_class": TestModule,
        "additional_check": check_async_tensor_is_early_scheduled,
      }

    _run_test(_lazy_scheduler_gen, expected_exec_order, **segment_has_graph_break_not_using_async_tensor_output())
    _run_test(_lazy_scheduler_gen, expected_exec_order, **segment_has_graph_break_using_async_tensor_output())


  def DISABLED_example_usage(self):
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
        Segment("overarch_func2_bwd", model.overarch.func2, nth_call=0),
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
        Segment("overarch_func2_bwd", model.overarch.func2, nth_call=0),
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
Design doc: https://docs.google.com/document/d/1vv0H5IMGwUMyzmJKnksJOnRSult1B4YlbBSs_MeAvXM/edit?usp=sharing
- Unit test: graph break within segment (i.e. multiple graphs per segment), either in the delayed segment or in the anchored segment
  - In the delayed segment case, also add output usage within the graph break eager region, to trigger the immediate materialization of AsyncTensor output
  - Make sure to assert that each GM contains the ops you expect.
- Unit test: in-place op in named segment
- Make debug_mode work
- Support user calling a method multiple times and only tag a specific call as segment (i.e. make `nth_call=X` work)
- For named segments, show its segment ID (prefix + fwd/bwd + nth_call) in profiler annotation in GPU trace
- Integration with DDPOptimizer
- Integration with FSDP (graph break version, not tracing)
- Integration with (selective) activation checkpointing
- What if a segment is in the schedule but is never run due to dynamic control flow change? we should either throw error or gracefully fall back to no-scheduler mode
- Try on Ads model: https://docs.google.com/document/d/1tFLUh4Xe4_eGKOtgpj08kfNDhy7Fqp-dSq0d7lejdZU/edit#bookmark=id.wds06wiqwjh2 figure out integration point with trainer loop
- (Later) Integration with compiled autograd
- Logging for better user debugging (what is scheduled and when, and the compilation output). Look at the generated graph and the original code.
- Also log memory usage, how much memory I am keeping above.

NOTE:
- Even if the segment is only in the backward, its corresponding forward segment will also be carved out (graph-break'ed).

AsyncTensor specific:
- Support kwargs in AsyncTensor ops
- Support tuple / list / etc. in AsyncTensor op args
- Support all factory functions in AsyncTensor dispatch
- Add unit tests to cover all common usage
"""

if __name__ == "__main__":
  from torch._dynamo.test_case import run_tests
  run_tests()


"""
FAQ
Q1: What happens if we have a user-defined segment deep down in a submodule?
Answer: everything before the defined segment will be in their own segment. Everything after is in another segment.
"""
