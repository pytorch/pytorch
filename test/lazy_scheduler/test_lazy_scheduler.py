"""
CUDA_VISIBLE_DEVICES=7 pytest -vs test/lazy_scheduler/test_lazy_scheduler.py::TestLazyScheduler::test_single_segment_prefix_fwd && \
CUDA_VISIBLE_DEVICES=7 pytest -vs test/lazy_scheduler/test_lazy_scheduler.py::TestLazyScheduler::test_single_segment_prefix_fwd_bwd && \
CUDA_VISIBLE_DEVICES=7 pytest -vs test/lazy_scheduler/test_lazy_scheduler.py::TestLazyScheduler::test_nested_segments_dep && \
CUDA_VISIBLE_DEVICES=7 pytest -vs test/lazy_scheduler/test_lazy_scheduler.py::TestLazyScheduler::test_nested_segments_non_dep && \
CUDA_VISIBLE_DEVICES=7 pytest -vs test/lazy_scheduler/test_lazy_scheduler.py::TestLazyScheduler::test_explicit_schedule_reordering_fwd_segments && \
CUDA_VISIBLE_DEVICES=7 pytest -vs test/lazy_scheduler/test_lazy_scheduler.py::TestLazyScheduler::test_explicit_schedule_reordering_fwd_and_bwd_segments && \
CUDA_VISIBLE_DEVICES=7 pytest -vs test/lazy_scheduler/test_lazy_scheduler.py::TestLazyScheduler::test_segment_compiled_with_different_backend && \
CUDA_VISIBLE_DEVICES=7 pytest -vs test/lazy_scheduler/test_lazy_scheduler.py::TestLazyScheduler::test_debug_mode_msg

>output.log 2>&1

buck2 run @mode/opt -c fbcode.enable_gpu_sections=true caffe2/test/lazy_scheduler:test_lazy_scheduler >output.txt 2>&1

How to run Ads APS model:
CUDA_VISIBLE_DEVICES=6,7 buck2 run @mode/opt -c fbcode.enable_gpu_sections=true //aps_models/ads/icvr:icvr_launcher -- mode=local_ctr_cvr_rep
"""

import unittest
import contextlib
import copy
import torch
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.testing._internal.common_utils import TestCase as TorchTestCase
from typing import Optional
from torch._lazy_scheduler import LazyScheduler, Segment
from torch.utils._triton import has_triton
from torch._inductor.utils import fresh_inductor_cache
from torch.testing._internal.common_distributed import (
  run_with_native_funcol,
)
import torch.distributed._functional_collectives as funcol
import torch.distributed as dist

# ======== REMOVE WHEN READY TO MERGE ========
import argparse
import os
import subprocess
import sys
import urllib
import urllib.parse
import uuid

from typing import Optional

PERFETTO_UI_ROOT_URL = (
    "https://interncache-all.fbcdn.net/manifold/perfetto-artifacts/tree/ui/index.html"
)
MANIFOLD_FOLDER = "perfetto_internal_traces/tree/shared_trace"
DEFAULT_TTL_SEC = 28 * 24 * 60 * 60


def upload_trace_file(local_path: str, overwrite: bool = False) -> Optional[str]:
    file_name = os.path.basename(local_path)
    manifold_path = os.path.join(
        MANIFOLD_FOLDER, f"{os.getlogin()}_{str(uuid.uuid4())}_{file_name}"
    )
    cmd = [
        "manifold",
        "put",
        local_path,
        manifold_path,
        "--ttl",
        str(DEFAULT_TTL_SEC),
        "--userData",
        "false",
    ]
    ret = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )
    if ret.returncode == 0:
        print("Upload trace successfully.")
        return manifold_path
    else:
        print("[ERROR] Upload failed, maybe the trace file exists.")
        return None


def print_perfetto_ui_url(manifold_path: str) -> None:
    url = (
        PERFETTO_UI_ROOT_URL
        + "#!/?url=https://interncache-all.fbcdn.net/manifold/"
        + urllib.parse.quote_plus(manifold_path)
    )
    print(f"The trace is accessible at:\n{url}")
# ======== REMOVE WHEN READY TO MERGE ========


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
      # print(f"ops_from_gm: {ops_from_gm}")
      assert ops_from_gm == expected_ops


class TestDepSegmentModule(torch.nn.Module):
  """
  Dependency chain:
  func1 -> func2 -> mul -> output
  """
  def __init__(self):
    super().__init__()
    self.param = torch.nn.Parameter(torch.randn(4, 4))

  def func1(self, x, y):
    return torch.matmul(x, y)

  def func2(self, x, y):
    return torch.add(x, y)

  def forward(self, x, y):
    z1 = self.func1(x, y)
    z2 = self.func2(x, z1)
    z1 = torch.matmul(z1, self.param)
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
      "forward_fwd": [{
        "eager": ['mm.default', 'mul.Tensor', 't.default', 't.default'],
        "compiled": ['mm.default', 'mul.Tensor', 'permute.default', 'permute.default'],
      }],
      "forward_bwd": [['mul.Tensor', 'mul.Tensor', 'mm.default', 'mm.default']],
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
  def __init__(self, hidden_size=4):
    super().__init__()
    self.param = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))

  def func1(self, x, y):
    return torch.matmul(x, y)

  def func2(self, x, y):
    return torch.add(x, y)

  def forward(self, x, y):
    z1 = self.func1(x, y)
    z2 = self.func2(x, y)
    z1 = torch.matmul(z1, self.param)
    z = z1 * z2
    return z


def check_segment_for_TestNonDepSegmentModule_fwd_only(lazy_scheduler, is_compile=False):
  return check_segment(
    lazy_scheduler,
    {
      "func1_fwd": [{
        "eager": ['mm.default', 't.default', 't.default'],
        "compiled": ['mm.default', 'permute.default', 'permute.default'],
      }],
      "func2_fwd": [['add.Tensor']],
      "forward_fwd": [{
        "eager": ['mm.default', 'mul.Tensor', 't.default', 't.default'],
        "compiled": ['mm.default', 'mul.Tensor', 'permute.default', 'permute.default'],
      }],
    },
    is_compile=is_compile,
  )


def check_segment_for_TestNonDepSegmentModule_fwd_bwd(lazy_scheduler, is_compile=False):
  return check_segment(
    lazy_scheduler,
    {
      "func1_fwd": [{
        "eager": ['mm.default', 't.default', 't.default'],
        "compiled": ['mm.default', 'permute.default', 'permute.default'],
      }],
      "func2_fwd": [['add.Tensor']],
      "forward_fwd": [{
        "eager": ['mm.default', 'mul.Tensor', 't.default', 't.default'],
        "compiled": ['mm.default', 'mul.Tensor', 'permute.default', 'permute.default'],
      }],
      "forward_bwd": [['mul.Tensor', 'mul.Tensor', 'mm.default', 'mm.default']],
      "func2_bwd": [[]],
      "func1_bwd": [['mm.default', 'mm.default']],
    },
    is_compile=is_compile,
  )


def _validate(
  self,
  eager_module,
  lazy_scheduler_gen,
  expected_execution_order,
  inps,
  *,
  fwd_only=False,
  skip_output_check=False,
  skip_param_grad_check=False,
  skip_input_grad_check=False,
  test_eager=True,
  test_compile=True,
  upload_profiler_trace=False,
  test_non_debug_mode_expected_error_regex=None,
  test_debug_mode_expected_msg_substrs=None,
  additional_check=None,
  num_iters=2,
):
  def _clone_inps():
    cloned_inps = []
    for inp in inps:
      cloned_inps.append(inp.clone().detach().requires_grad_(inp.requires_grad))
    return cloned_inps

  def _compare_output_and_grad(eager_module, lazy_scheduler_gen, debug_mode, is_compile=False, additional_check=None):
    inps_no_ls = _clone_inps()
    inps_ls = _clone_inps()

    if is_compile:
      baseline_fn = torch.compile(eager_module, fullgraph=False, backend="inductor")
    else:
      baseline_fn = eager_module

    # Original function, run several iterations
    for i in range(num_iters):
      print(f"------------------ eager iter: {i} ------------------")
      torch.manual_seed(0)
      expected = baseline_fn(*inps_no_ls)
      if not fwd_only:
        expected.sum().backward()

    eager_module_clone = copy.deepcopy(eager_module)
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
      # LazyScheduler function, run several iterations
      for i in range(num_iters):
        print(f"------------------ LazyScheduler iter: {i} ------------------")
        torch.manual_seed(0)
        lazy_scheduler = lazy_scheduler_gen(eager_module_clone, is_compile=is_compile)
        result = lazy_scheduler(*inps_ls)
        if not fwd_only:
          result.sum().backward()
    if upload_profiler_trace:
      if is_compile:
        profiler_trace_path = "compiled_trace.json"
      else:
        profiler_trace_path = "eager_trace.json"
      prof.export_chrome_trace(profiler_trace_path)
      if not os.path.exists(profiler_trace_path):
        raise Exception(f"[ERROR] The trace file doesn't exist: {profiler_trace_path}")
      manifold_path = upload_trace_file(profiler_trace_path)
      if manifold_path:
          print_perfetto_ui_url(manifold_path)

    if debug_mode:
      lazy_scheduler.debug()

    if not skip_output_check:
      self.assertEqual(result, expected, msg="Output mismatch between torch.compile and eager versions")
    if not fwd_only:
      if not skip_param_grad_check:
        self.assertEqual(
          eager_module.param.grad, eager_module_clone.param.grad,
          msg=f"Gradient mismatch between torch.compile and eager versions. eager_module.param.grad: {eager_module.param.grad}, eager_module_clone.param.grad: {eager_module_clone.param.grad}",
        )
      if not skip_input_grad_check:
        for inp, cloned_inp in zip(inps_no_ls, inps_ls):
          self.assertEqual(
            inp.grad, cloned_inp.grad,
            msg=f"Gradient mismatch between torch.compile and eager versions. inp.grad: {inp.grad}, cloned_inp.grad: {cloned_inp.grad}",
          )

    if additional_check is not None:
      if not isinstance(additional_check, list):
        additional_check = [additional_check]
      for check in additional_check:
        check(lazy_scheduler, is_compile=is_compile)

    # NOTE: We don't care about the execution order of unnamed or unregistered segments.
    recorded_execution_order = lazy_scheduler.get_recorded_execution_order()
    err_msg = f"""
Expected execution order to be:
{expected_execution_order},

but got:
{recorded_execution_order}

If recorded execution order is empty, it means all segments have fallen back to non-lazy execution.
One possible reason is that all segments are skipped by Dynamo due to skipfiles rules
(e.g. in FBCODE, we skip all files in FBCODE_SKIP_DIRS (e.g. `torchrec/distributed/`) in Dynamo trace_rules.py).
Please make sure eager module definitions are not in the skipped directories.
To further investigate, please enable TORCH_COMPILE_DEBUG=1 and search for "inline in skipfiles" in Dynamo logs.
"""
    self.assertEqual(len(recorded_execution_order), len(expected_execution_order), msg=err_msg)
    self.assertEqual(recorded_execution_order, expected_execution_order, msg=err_msg)

  def _test_eager_and_compile(debug_mode):
    if test_eager:
      _compare_output_and_grad(copy.deepcopy(eager_module), lazy_scheduler_gen, debug_mode, is_compile=False, additional_check=additional_check)
      torch._dynamo.reset()
      print(f"Eager mode test done!")

    if test_compile:
      _compare_output_and_grad(copy.deepcopy(eager_module), lazy_scheduler_gen, debug_mode, is_compile=True, additional_check=additional_check)
      torch._dynamo.reset()
      print(f"Compile mode test done!")

  if test_debug_mode_expected_msg_substrs is not None:
    try:
      _test_eager_and_compile(debug_mode=True)
    except RuntimeError as e:
      self.assertTrue(
        all(substr in str(e) for substr in test_debug_mode_expected_msg_substrs),
        msg=f"Debug mode test failed. Expected substring: {test_debug_mode_expected_msg_substrs}. Actual error message: {str(e)}",
      )
    else:
      raise AssertionError("Debug mode did not throw expected RuntimeError.")

  ctx = contextlib.nullcontext()
  if test_non_debug_mode_expected_error_regex is not None:
    ctx = self.assertRaisesRegex(Exception, test_non_debug_mode_expected_error_regex)
  with ctx:  # debug_mode=False
    _test_eager_and_compile(debug_mode=False)

class TestLazyScheduler(TorchTestCase):
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
        },
      )

    _validate(
      self,
      m,
      lazy_scheduler_gen,
      expected_execution_order=[
        "func1_fwd",
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
        },
      )

    _validate(
      self,
      m,
      lazy_scheduler_gen,
      expected_execution_order=[
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

    expected_execution_order = [
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
        schedule=expected_execution_order,
        compile_options=None if not is_compile else {
          "fullgraph": False,
          "backend": "inductor",
        },
      )

    _validate(
      self,
      m,
      lazy_scheduler_gen,
      expected_execution_order=expected_execution_order,
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

    expected_execution_order = [
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
        schedule=expected_execution_order,
        compile_options=None if not is_compile else {
          "fullgraph": False,
          "backend": "inductor",
        },
      )

    _validate(
      self,
      m,
      lazy_scheduler_gen,
      expected_execution_order=expected_execution_order,
      inps=[x, y],
      additional_check=check_segment_for_TestNonDepSegmentModule_fwd_bwd,
    )

  def test_explicit_schedule_reordering_fwd_segments(self):
    device = "cuda"
    m = TestNonDepSegmentModule()
    m = m.to(device)
    x = torch.randn(4, 4, requires_grad=True, device=device)
    y = torch.randn(4, 4, requires_grad=True, device=device)

    expected_execution_order = [
      "func2_fwd",
      "func1_fwd",
      "forward_fwd",
    ]

    def lazy_scheduler_gen(module, is_compile=False):
      return LazyScheduler(
        module,
        segments=[
          Segment("func1_fwd", module.func1),
          Segment("func2_fwd", module.func2),
          Segment("forward_fwd", module.forward),
        ],
        schedule=expected_execution_order,
        compile_options=None if not is_compile else {
          "fullgraph": False,
          "backend": "inductor",
        },
      )

    _validate(
      self,
      m,
      lazy_scheduler_gen,
      expected_execution_order=expected_execution_order,
      inps=[x, y],
      additional_check=check_segment_for_TestNonDepSegmentModule_fwd_only,
    )

  def test_explicit_schedule_reordering_fwd_and_bwd_segments(self):
    device = "cuda"
    hidden_size = 10240
    m = TestNonDepSegmentModule(hidden_size=hidden_size)  # hidden_size=4
    m = m.to(device)
    # x = torch.randn(4, 4, requires_grad=True, device=device)
    # y = torch.randn(4, 4, requires_grad=True, device=device)
    x = torch.randn(hidden_size, hidden_size, requires_grad=True, device=device)
    y = torch.randn(hidden_size, hidden_size, requires_grad=True, device=device)

    expected_execution_order = [
      "func2_fwd",
      "func1_fwd",
      "forward_fwd",
      "forward_bwd",
      "func1_bwd",
      "func2_bwd",
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
        schedule=expected_execution_order,
        compile_options=None if not is_compile else {
          "fullgraph": False,
          "backend": "inductor",
        },
      )

    _validate(
      self,
      m,
      lazy_scheduler_gen,
      expected_execution_order=expected_execution_order,
      inps=[x, y],
      additional_check=check_segment_for_TestNonDepSegmentModule_fwd_bwd,
    )

  def test_segment_compiled_with_different_backend(self):
    def _run_test(lazy_scheduler_gen, expected_execution_order, additional_check, fwd_only=False):
      device = "cuda"
      m = TestNonDepSegmentModule()
      m = m.to(device)
      x = torch.randn(4, 4, requires_grad=True, device=device)
      y = torch.randn(4, 4, requires_grad=True, device=device)

      _validate(
        self,
        m,
        lazy_scheduler_gen,
        expected_execution_order=expected_execution_order,
        inps=[x, y],
        fwd_only=fwd_only,
        additional_check=additional_check,
      )

    def segment_use_dynamo_eager_fwd_only():
      expected_execution_order = [
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
            "forward_fwd": [['mm.default', 'mul.Tensor', 'permute.default', 'permute.default']]
          },
          is_compile=is_compile,
        )
      def lazy_scheduler_gen(module, is_compile=False):
        return LazyScheduler(
          module,
          segments=[
            Segment("func1_fwd", module.func1, backend="eager"),
            Segment("func2_fwd", module.func2, backend="aot_eager"),
            Segment("forward_fwd", module.forward, backend="inductor"),
          ],
          schedule=expected_execution_order,
          compile_options=None if not is_compile else {
            "fullgraph": False,
            "backend": "inductor",
          },
          )
      return lazy_scheduler_gen, expected_execution_order, check_segment_fwd_only

    def segment_use_dynamo_aot_eager_fwd_bwd():
      expected_execution_order = [
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
            "forward_fwd": [['mm.default', 'mul.Tensor', 'permute.default', 'permute.default']],
            "forward_bwd": [['mul.Tensor', 'mul.Tensor', 'mm.default', 'mm.default']],
            "func2_bwd": [[]],
            "func1_bwd": [['mm.default', 'mm.default']],
          },
          is_compile=is_compile,
        )
      def lazy_scheduler_gen(module, is_compile=False):
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
          schedule=expected_execution_order,
          compile_options=None if not is_compile else {
            "fullgraph": False,
            "backend": "inductor",
          },
          )
      return lazy_scheduler_gen, expected_execution_order, check_segment_fwd_bwd

    _run_test(*segment_use_dynamo_eager_fwd_only(), fwd_only=True)
    _run_test(*segment_use_dynamo_aot_eager_fwd_bwd(), fwd_only=False)

  def test_debug_mode_msg(self):
    device = "cuda"
    expected_execution_order = [
      "func2_fwd",
      "func1_fwd",
      "forward_fwd",
    ]

    def lazy_scheduler_gen(module, is_compile=False):
      return LazyScheduler(
        module,
        segments=[
          # SegmentA (also called "delayed_segment", because it's delayed in the schedule)
          Segment("func1_fwd", module.func1, backend="aot_eager"),
          # SegmentB
          Segment("func2_fwd", module.func2, backend="aot_eager"),
          Segment("forward_fwd", module.forward, backend="inductor"),
        ],
        schedule=expected_execution_order,
        compile_options=None if not is_compile else {
          "fullgraph": False,
          "backend": "inductor",
        },
      )

    def _run_test(
      mod_class, lazy_scheduler_gen, expected_execution_order, additional_check=None, extra_input_no_require_grad=False,
      test_non_debug_mode_expected_error_regex="Tensor-likes are not close",
      test_debug_mode_expected_msg_substrs=None,
    ):
      m = mod_class()
      m = m.to(device)
      x = torch.randn(4, 4, requires_grad=True, device=device)
      y = torch.randn(4, 4, requires_grad=True, device=device)
      if extra_input_no_require_grad:
        k = torch.randn(4, 4, requires_grad=False, device=device)

      _validate(
        self,
        m,
        lazy_scheduler_gen,
        expected_execution_order=expected_execution_order,
        inps=[x, y, k] if extra_input_no_require_grad else [x, y],
        additional_check=additional_check,
        test_non_debug_mode_expected_error_regex=test_non_debug_mode_expected_error_regex,
        test_debug_mode_expected_msg_substrs=test_debug_mode_expected_msg_substrs,
      )

    class TestModule_read_shared_buf_mutated_by_segmentB_in_delayed_segment(torch.nn.Module):
      def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(4, 4))
        self.register_buffer('shared_buf', torch.zeros(4, 4))

      def func1(self, x, y):
        y2 = torch.matmul(x, y)
        y2 = y2 + self.shared_buf
        return torch.matmul(x, y2)

      def func2(self, x, y):
        y3 = torch.add(x, y)
        self.shared_buf.add_(1.5)
        return torch.add(x, y3)

      def forward(self, x, y):
        z1 = self.func1(x, y)
        z2 = self.func2(x, y)
        z1 = torch.matmul(z1, self.param)
        z = z1 * z2
        return z

    class TestModule_read_shared_buf_mutated_by_delayed_segment_in_segmentB(torch.nn.Module):
      def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(4, 4))
        self.register_buffer('shared_buf', torch.zeros(4, 4))

      def func1(self, x, y):
        y2 = torch.matmul(x, y)
        self.shared_buf.add_(1.5)
        return torch.matmul(x, y2)

      def func2(self, x, y):
        y3 = torch.add(x, y)
        y3 = y3 + self.shared_buf
        return torch.add(x, y3)

      def forward(self, x, y):
        z1 = self.func1(x, y)
        z2 = self.func2(x, y)
        z1 = torch.matmul(z1, self.param)
        z = z1 * z2
        return z

    glb_tensor = torch.zeros(4, 4, device=device)
    class TestModule_read_global_tensor_mutated_by_segmentB_in_delayed_segment(torch.nn.Module):
      def __init__(self):
        global glb_tensor
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(4, 4))
        glb_tensor = torch.zeros(4, 4, device=device)

      def func1(self, x, y):
        y2 = torch.matmul(x, y)
        y2 = y2 + glb_tensor
        return torch.matmul(x, y2)

      def func2(self, x, y):
        global glb_tensor
        y3 = torch.add(x, y)
        glb_tensor.add_(1.5)
        return torch.add(x, y3)

      def forward(self, x, y):
        z1 = self.func1(x, y)
        z2 = self.func2(x, y)
        z1 = torch.matmul(z1, self.param)
        z = z1 * z2
        return z

    glb_tensor = torch.zeros(4, 4, device=device)
    class TestModule_read_global_tensor_mutated_by_delayed_segment_in_segmentB(torch.nn.Module):
      def __init__(self):
        global glb_tensor
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(4, 4))
        glb_tensor = torch.zeros(4, 4, device=device)

      def func1(self, x, y):
        global glb_tensor
        y2 = torch.matmul(x, y)
        glb_tensor.add_(1.5)
        return torch.matmul(x, y2)

      def func2(self, x, y):
        y3 = torch.add(x, y)
        y3 = y3 + glb_tensor
        return torch.add(x, y3)

      def forward(self, x, y):
        z1 = self.func1(x, y)
        z2 = self.func2(x, y)
        z1 = torch.matmul(z1, self.param)
        z = z1 * z2
        return z

    class TestModule_read_input_tensor_mutated_by_segmentB_in_delayed_segment(torch.nn.Module):
      def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(4, 4))

      def func1(self, x, y, k):
        y2 = torch.matmul(x, y)
        y2 = y2 + k
        return torch.matmul(x, y2)

      def func2(self, x, y, k):
        y3 = torch.add(x, y)
        k.add_(1.5)
        return torch.add(x, y3)

      def forward(self, x, y, k):
        z1 = self.func1(x, y, k)
        z2 = self.func2(x, y, k)
        z1 = torch.matmul(z1, self.param)
        z = z1 * z2
        return z

    class TestModule_read_input_tensor_mutated_by_delayed_segment_in_segmentB(torch.nn.Module):
      def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(4, 4))

      def func1(self, x, y, k):
        y2 = torch.matmul(x, y)
        k.add_(1.5)
        return torch.matmul(x, y2)

      def func2(self, x, y, k):
        y3 = torch.add(x, y)
        y3 = y3 + k
        return torch.add(x, y3)

      def forward(self, x, y, k):
        z1 = self.func1(x, y, k)
        z2 = self.func2(x, y, k)
        z1 = torch.matmul(z1, self.param)
        z = z1 * z2
        return z

    _run_test(TestModule_read_shared_buf_mutated_by_segmentB_in_delayed_segment, lazy_scheduler_gen, expected_execution_order,
      test_debug_mode_expected_msg_substrs=["Delayed segments: ['func1_fwd']", "Segments that contain in-place mutation ops: ['func2']"])
    _run_test(TestModule_read_shared_buf_mutated_by_delayed_segment_in_segmentB, lazy_scheduler_gen, expected_execution_order,
      test_debug_mode_expected_msg_substrs=["Delayed segments: ['func1_fwd']", "Segments that contain in-place mutation ops: ['func1']"])
    _run_test(TestModule_read_global_tensor_mutated_by_segmentB_in_delayed_segment, lazy_scheduler_gen, expected_execution_order,
      test_debug_mode_expected_msg_substrs=["Delayed segments: ['func1_fwd']", "Segments that contain in-place mutation ops: ['func2']"])
    _run_test(TestModule_read_global_tensor_mutated_by_delayed_segment_in_segmentB, lazy_scheduler_gen, expected_execution_order, test_non_debug_mode_expected_error_regex="['func1_fwd', 'func2_fwd', 'forward_fwd']",
      test_debug_mode_expected_msg_substrs=["end of `func1_fwd` (exclusive) to end of `func2_fwd` (inclusive) depends on output of `func1_fwd`", "Delayed segments: ['func1_fwd']", "Segments that contain in-place mutation ops: ['func1']"])
    _run_test(TestModule_read_input_tensor_mutated_by_segmentB_in_delayed_segment, lazy_scheduler_gen, expected_execution_order, extra_input_no_require_grad=True,
      test_debug_mode_expected_msg_substrs=["Delayed segments: ['func1_fwd']", "Segments that contain in-place mutation ops: ['func2']"])
    _run_test(TestModule_read_input_tensor_mutated_by_delayed_segment_in_segmentB, lazy_scheduler_gen, expected_execution_order, extra_input_no_require_grad=True, test_non_debug_mode_expected_error_regex="['func1_fwd', 'func2_fwd', 'forward_fwd']",
      test_debug_mode_expected_msg_substrs=["end of `func1_fwd` (exclusive) to end of `func2_fwd` (inclusive) depends on output of `func1_fwd`", "Delayed segments: ['func1_fwd']", "Segments that contain in-place mutation ops: ['func1']"])
    # TODO add a case to trigger "end of `func2_fwd` (exclusive) to end of `func3_fwd` (inclusive) depends on output of `func1_fwd`" for schedule [2, 1, 3]

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
        Segment("sdd_fwd", model.sdd.forward),
        Segment("overarch_func2_bwd", model.overarch.func2),
      ],
      # Run "sdd_fwd" right before "overarch_func2_bwd".
      schedule=[],
      # Actual execution order is:
      # "_____overarch_func1_fwd", "______overarch_func2_fwd", "sdd_fwd", "overarch_func2_bwd", "______overarch_func1_bwd"
    )
    output = model_ls(inputs)

    # Compile mode
    model_c = LazyScheduler(
      model,
      segments=[
        Segment("sdd_fwd", model.sdd.forward, backend="eager"),
        Segment("overarch_func2_bwd", model.overarch.func2),
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

class TestLazySchedulerMultiProc(TorchTestCase):
  @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
  @fresh_inductor_cache()
  @run_with_native_funcol
  def setUp(self):
    self.rank = 0
    self.world_size = 2
    torch.cuda.set_device("cuda:0")

    store = FakeStore()
    dist.init_process_group(
        backend="fake",
        world_size=self.world_size,
        rank=self.rank,
        store=store,
    )

  def tearDown(self):
    dist.destroy_process_group()

  def test_comm_in_segA_then_wait_in_segB(self):
    hidden_size = 4
    class TestCommInSegAThenWaitInSegBModule(torch.nn.Module):
      def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))

      def all_reduce_issue(self, x):
        ar0 = funcol.all_reduce(x, "avg", "0")
        return ar0

      def all_reduce_wait(self, ar0):
        ar0 = funcol.wait_tensor(ar0)
        return ar0

      def func3(self, x):
        z = torch.matmul(x, self.param)
        return z

      def forward(self, x):
        ar_out = self.all_reduce_issue(x)
        wait_out = self.all_reduce_wait(ar_out)
        func3_out = self.func3(x)
        out = wait_out * func3_out
        return out

    device = "cuda"

    def _test_with_schedule_and_expected_execution_order(schedule, expected_execution_order):
      def lazy_scheduler_gen(module, is_compile=False):
        return LazyScheduler(
          module,
          segments=[
            Segment("all_reduce_issue_fwd", module.all_reduce_issue),
            Segment("all_reduce_wait_fwd", module.all_reduce_wait),
            Segment("func3_fwd", module.func3),
            Segment("forward_fwd", module.forward),
          ],
          schedule=schedule,
          compile_options=None if not is_compile else {
            "fullgraph": False,
            "backend": "inductor",
          },
        )

      m = TestCommInSegAThenWaitInSegBModule()
      m = m.to(device)
      x = torch.randn(hidden_size, hidden_size, device=device)
      _validate(
        self,
        m,
        lazy_scheduler_gen,
        expected_execution_order=expected_execution_order,
        inps=[x],
        skip_input_grad_check=True,
      )

    # Case 1: if we put `func3_fwd` before `all_reduce_wait_fwd` in the schedule,
    # we notice that `func3_fwd` is executed before `all_reduce_wait_fwd` which is correct.
    schedule = [
      "all_reduce_issue_fwd",
      "func3_fwd",
      "all_reduce_wait_fwd",
      "forward_fwd",
    ]
    _test_with_schedule_and_expected_execution_order(
      schedule=schedule,
      expected_execution_order=schedule,
    )

    # Case 2: if we put `forward_fwd` before `all_reduce_wait_fwd` in the schedule,
    # `all_reduce_wait_fwd` will still be executed first because it produces input to `forward_fwd`.
    schedule = [
      "all_reduce_issue_fwd",
      "func3_fwd",
      "forward_fwd",
      "all_reduce_wait_fwd",
    ]
    expected_execution_order = [
      "all_reduce_issue_fwd",
      "func3_fwd",
      "all_reduce_wait_fwd",
      "forward_fwd",
    ]
    _test_with_schedule_and_expected_execution_order(
      schedule=schedule,
      expected_execution_order=expected_execution_order,
    )


"""
TODO:
Design doc: https://docs.google.com/document/d/1vv0H5IMGwUMyzmJKnksJOnRSult1B4YlbBSs_MeAvXM/edit?usp=sharing
- Try on Ads model: https://docs.google.com/document/d/1tFLUh4Xe4_eGKOtgpj08kfNDhy7Fqp-dSq0d7lejdZU/edit#bookmark=id.wds06wiqwjh2 figure out integration point with trainer loop

- For named segments, show its segment ID (prefix + fwd/bwd + nth_call) in profiler annotation in GPU trace
- Logging for better user debugging (what is scheduled and when, and the compilation output). Look at the generated graph and the original code.
- Also log memory usage, how much memory I am keeping above.
- Integration with DDPOptimizer
- Integration with FSDP (graph break version, not tracing)
- Support user calling a method multiple times and only tag a specific call as segment (i.e. make `nth_call=X` work)
- Integration with (selective) activation checkpointing
- (Later) Integration with compiled autograd
- To specify where to insert the delayed segment, currently we need to tag the function immediately before and also the function immediately after the insertion point.
  We should be able to improve on this to only need to tag the function immediately after the insertion point.
- Support graph break within segment (i.e. multiple graphs per segment), either in the delayed segment or in the anchored segment
  - In the delayed segment case, also add output usage within the graph break eager region, to trigger the immediate materialization of AsyncTensor output
  - Make sure to assert that each GM contains the ops you expect.
- What if a segment is in the schedule but is never run due to dynamic control flow change? we should either throw error or gracefully fall back to no-scheduler mode

NOTE:
- Even if the segment is only in the backward, its corresponding forward segment will also be carved out (graph-break'ed).

NOTE re. comm in segA and wait in segB:
- Do we need to modify Dynamo to support AsyncCollectiveTensor as output from graph / input to graph?
  - The discussion on having AsyncCollectiveTensor as output from the graph in Composability meeting (https://docs.google.com/document/d/1QTR3t3KdRu5JT1lvAuJLsPd3LruCfv0LecpsgR8eWhg/edit?kh_source=GDOCS#bookmark=id.hqvxodlpkooe)
  - Current situation:
    - For sparse data distribution (SDD) op reordering via LazyScheduler, I need to schedule the all_to_all op in segment A, have the AsyncCollectiveTensor as output from segment A, and then schedule its corresponding wait_tensor in a later segment B.
    - Currently I require user to explicitly use a wait_tensor op in the user program, and I do an FX pass to remove the auto-generated wait_tensor op and instead use the user-written wait_tensor op (to make sure all_to_all is in seg A and its wait_tensor op is in seg B).
    - Currently the graph output of seg A is always torch.Tensor instead of AsyncCollectiveTensor. This seems ok, because I don't use the output of seg A until after I schedule the wait_tensor op in seg B.

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
