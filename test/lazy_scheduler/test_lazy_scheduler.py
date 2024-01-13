"""
pytest -vs test/lazy_scheduler/test_lazy_scheduler.py::TestLazyScheduler::test_single_unnamed_segment
pytest -vs test/lazy_scheduler/test_lazy_scheduler.py::TestLazyScheduler::test_split_module_dep_segments
pytest -vs test/lazy_scheduler/test_lazy_scheduler.py::TestLazyScheduler::test_split_module_non_dep_segments
pytest -vs test/lazy_scheduler/test_lazy_scheduler.py::TestLazyScheduler::test_segment_tagging
pytest -vs test/lazy_scheduler/test_lazy_scheduler.py::TestLazyScheduler::test_explicit_schedule
pytest -vs test/lazy_scheduler/test_lazy_scheduler.py::TestLazyScheduler::test_explicit_schedule_reordering
pytest -vs test/lazy_scheduler/test_lazy_scheduler.py::TestLazyScheduler::test_register_segment_hook
"""

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
import traceback


def get_segment_prefix_from_gm(gm):
  segment_prefix = None
  for node in gm.graph.nodes:
    if node.op != "placeholder" and node.op != "output":
      assert "segment_prefix" in node.meta
      if not segment_prefix:
        segment_prefix = node.meta["segment_prefix"]
      else:
        assert segment_prefix == node.meta["segment_prefix"], f"{segment_prefix} vs. {node.meta['segment_prefix']}"
  return segment_prefix


def compute_segment_name(segment_prefix, is_backward):
  return f"{segment_prefix}_{'bwd' if is_backward else 'fwd'}"


class AsyncFuncHandle:
  """
  We use this class to represent the function that needs to be scheduled.
  It also has methods for checking whether the function has been scheduled or completed.
  """
  def __init__(self, compiled_fn, segment, args, outs_async, scheduler):
    self.cuda_event = torch.cuda.Event()
    self.compiled_fn: Callable = compiled_fn
    self.args = args
    self.outs_async = outs_async
    self.outs = None
    self.segment = segment
    self.is_going_to_be_scheduled = False
    self._scheduler = weakref.ref(scheduler)

  def schedule(self):
    # make sure to schedule only once
    if self.is_going_to_be_scheduled:
      return
    self.is_going_to_be_scheduled = True
    AsyncTensor.wait_until_materialized(self.args)
    args_materialized = pytree.tree_map_only(AsyncTensor, lambda x: x._materialized_tensor, pytree.tree_map(lambda x: x.detach(), self.args))
    self._scheduler().add_to_recorded_execution_order(self.segment)
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


def split_module_based_on_segment_info(gm: torch.fx.GraphModule):
  known_segments = []
  for node in gm.graph.nodes:
    if len(known_segments) == 0 or node.meta["segment_prefix"] != known_segments[-1]:
      known_segments.append(node.meta["segment_prefix"])

  def split_callback(node):
    return known_segments.index(node.meta["segment_prefix"])

  qualname_map = {}
  gm_after_split = torch.fx.passes.split_module.split_module(
    m=gm,
    root_m=None,
    split_callback=split_callback,
    qualname_map=qualname_map,
    keep_original_order=True,
  )

  # Check invariant: a named segment should only contain ops from the same NN module method
  for _, sub_gm in gm_after_split.named_children():
    nn_module_method = None
    for node in sub_gm.graph.nodes:
      if node.op != "placeholder" and node.op != "output":
        assert "segment_prefix" in node.meta
        if not node.meta["segment_prefix"].startswith("unnamed_"):
          if not nn_module_method:
            nn_module_method = node.meta["nn_module_method"]
          else:
            assert nn_module_method == node.meta["nn_module_method"], f"{nn_module_method} vs. {node.meta['nn_module_method']}"

  return gm_after_split


# TODO: maybe merge LazySchedulerGraphModule and AsyncFuncHandle
class LazySchedulerGraphModule(torch.nn.Module):
  """
  This module wraps around a GraphModule.
  Its __call__ method doesn't execute the graph module immediately.
  Instead, it calls the scheduler's maybe_run method, which decides
  whether to run the graph module based on the schedule.
  """
  def __init__(self, scheduler, segment, gm, compiled_fn):
    super().__init__()
    self.scheduler = scheduler
    self.segment = segment
    self.gm = gm
    self.compiled_fn = compiled_fn

  def __call__(self, *args):
    assert self.compiled_fn is not None
    return self.scheduler.maybe_run(self.gm, self.compiled_fn, self.segment, *args)


class LazyScheduler:
  """
  LazyScheduler is used to decide when to schedule the execution of a graph module (based on the schedule).
  """
  def __init__(self, schedule):
    # If `schedule` is empty list, it means we don't enforce the execution order.
    self._schedule = schedule
    self._gm_to_handle_map = OrderedDict()
    self._handle_to_gm_map = OrderedDict()
    self._segment_to_gms_map = defaultdict(list)
    self._recorded_execution_order = []

  def add_to_recorded_execution_order(self, segment):
    if len(self._recorded_execution_order) > 0 and self._recorded_execution_order[-1] == segment:
      return
    self._recorded_execution_order.append(segment)

  def _compile_fx_inner(
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

    assert "segment_prefix" in kwargs
    segment_prefix = kwargs["segment_prefix"]
    del kwargs["segment_prefix"]

    is_backward = kwargs.get("is_backward", False)

    segment = compute_segment_name(segment_prefix, is_backward)

    # NOTE: `gm` in this function is the post-AOTAutograd fwd or bwd GraphModule,
    # each node in `gm` originally does not have segment info, and we need to re-populate it here.
    for node in gm.graph.nodes:
      node.meta["segment"] = segment

    assert "inner_compile_orig" in kwargs
    inner_compile_orig = kwargs["inner_compile_orig"]
    del kwargs["inner_compile_orig"]
    # Call the user-specified original compiler
    compiled_fn = inner_compile_orig(gm, *args, **kwargs)

    lazy_gm = LazySchedulerGraphModule(
      self,
      segment,
      gm,
      compiled_fn,
    )

    # Build segment -> GMs mapping
    self._segment_to_gms_map[segment].append(gm)

    return lazy_gm

  def _compile_fx(
    self,
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    **kwargs,
  ):
    segment_prefix = get_segment_prefix_from_gm(gm)
    # `inner_compile` can be passed in via `torch.compile(m, functools.partial(inductor_compile_fx, inner_compile=...))`.
    # It's the custom compiler for each fwd or bwd GraphModule.
    inner_compile_orig = kwargs.get("inner_compile", inductor_compile_fx_inner)

    kwargs.update({
      "inner_compile": functools.partial(self._compile_fx_inner, segment_prefix=segment_prefix, inner_compile_orig=inner_compile_orig)
    })
    return inductor_compile_fx(gm, example_inputs, **kwargs)

  def _compile_fn(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor], backend_compile_fn, segment_prefix_assignment_fn):
    # Do segment prefix assignment, and then split the graph module based on segment prefix.
    for node in gm.graph.nodes:
      assert "nn_module_method" in node.meta

    segment_prefix_assignment_fn(gm)
    split_gm = split_module_based_on_segment_info(gm)

    submod_compiler = SubmoduleReplacer(split_gm, backend_compile_fn)
    submod_compiler.run(*example_inputs)
    split_gm.recompile()

    return split_gm

  def maybe_run(self, gm, compiled_fn, segment, *args):
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
      cur_handle = AsyncFuncHandle(compiled_fn, segment, args=args, outs_async=outs_async, scheduler=self)
      self._gm_to_handle_map[gm] = cur_handle
      self._handle_to_gm_map[cur_handle] = gm
    for out_async in outs_async:
      out_async.set_handle(cur_handle)

    # First, try to schedule all graphs from all segments that are before the incoming graph in the schedule.
    # The incoming graph can be scheduled only if:
    # 1. All preceding graphs have their handles created.
    # 2. All preceding graphs have been scheduled.
    all_preceding_graph_handles = []
    all_preceding_graph_handles_are_created = True
    reached_current_graph = False
    # TODO: for now, we always check the schedule from the beginning.
    # We can optimize this by keeping track of which segments have been scheduled already.
    _next_segment_index = 0
    while _next_segment_index < len(self._schedule):
      segment = self._schedule[_next_segment_index]
      if segment not in self._segment_to_gms_map:
        all_preceding_graph_handles_are_created = False
        break
      else:
        for g in self._segment_to_gms_map[segment]:
          if str(g.graph) == str(gm.graph):  # TODO: is there a better way to check graph equivalence?
            reached_current_graph = True
            break
          if g not in self._gm_to_handle_map:
            all_preceding_graph_handles_are_created = False
            break
          else:
            all_preceding_graph_handles.append(self._gm_to_handle_map[g])
      if reached_current_graph or (not all_preceding_graph_handles_are_created):
        break
      else:
        _next_segment_index += 1

    if not all_preceding_graph_handles_are_created:
      # If not all preceding graph handles are created, then we don't schedule the current graph yet.
      return cur_handle.outs_async
    else:
      # If all preceding graph handles are created, then we schedule all of them,
      # and then schedule the current graph.
      for handle in all_preceding_graph_handles:
        handle.schedule()
      cur_handle.schedule()
      return cur_handle.outs_async

class SubmoduleReplacer(torch.fx.interpreter.Interpreter):
  # This is a copy of DDPOptimizer `SubmoduleReplacer` class.
  def __init__(self, module, compiler):
    super().__init__(module)
    self.compiler = compiler

  def lazily_compiled_submod(self, input_mod):
    """
    Create a wrapper around submodules which:
    - lazily compiles each of the partitioned submodules using the user-provided compiler
    - unpacks singleton tuples/lists into flat arg
    """

    class LazilyCompiledModule(torch.nn.Module):
      def __init__(self, submod, compiler, unwrap_singleton_tuple):
        super().__init__()
        self.submod = submod
        self.compiler = compiler
        self.compiled = False
        self.unwrap_singleton_tuple = unwrap_singleton_tuple

      def forward(self, *args):
        if not self.compiled:
          # First compile with args as example_inputs
          # These args will be fakeified if using Inductor/AOTAutograd
          new_submod = self.compiler(self.submod, args)
          del self.submod
          self.submod = new_submod
          self.compiled = True
          self.compiler = None

        x = self.submod(*args)
        # we must let 'input_mod' return a tuple, to make AOT happy.
        # (aot_autograd compile_fn literally requires that the output of a graph it compiles is a tuple).
        # however, we don't acutally want this tuple to be returned, since the fx logic that calls the submod
        # will again wrap outputs from the submod in a tuple.  So we unwrap it, and count on it being re-wrapped
        if self.unwrap_singleton_tuple and isinstance(x, (tuple, list)):
          return x[0]
        return x

    unwrap_singleton_tuple = False
    for sn in input_mod.graph.nodes:
      if sn.op == "output":
        if not isinstance(sn.args[0], tuple):
          unwrap_singleton_tuple = True
          sn.args = (sn.args,)

    input_mod.recompile()
    input_mod.compile_subgraph_reason = GraphCompileReason(
      "LazyScheduler intentional graph-break (See Note [LazyScheduler] TODO)."
      " Set `torch._dynamo.config.lazy_scheduler_compile_fn = None` to disable.",
      [
        # it's close to useless to get a real stacktrace here, and quite verbose.
        traceback.FrameSummary(__file__, 0, SubmoduleReplacer),
      ],
    )

    wrapper = LazilyCompiledModule(
      input_mod,
      self.compiler,
      unwrap_singleton_tuple,
    )
    return wrapper

  # We replace the submodules with lazy submodules which compile
  # the corresponding submodules when they are run with real values
  # Always returns `None` - we do not need to propagate values in order
  # to replace submodules.
  def run_node(self, n: torch.fx.Node) -> Any:
    if n.op == "call_module":
      real_mod = self.fetch_attr(n.target)

      assert len(n.kwargs) == 0, "We assume only args for these modules"

      lazily_compiled_submod = self.lazily_compiled_submod(real_mod)

      # We update the original (outer) graph with a call into the compiled module
      # instead of the uncompiled one.
      self.module.delete_submodule(n.target)
      n.target = "compiled_" + n.target
      self.module.add_submodule(n.target, lazily_compiled_submod)


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


def segment_prefix_assignment_fn(gm, segment_dict):
  next_unnamed_segment_id = 0
  in_unnamed_segment = False
  for _, node in enumerate(gm.graph.nodes):
    assert "nn_module_method" in node.meta
    if node.meta["nn_module_method"] in segment_dict:
      if in_unnamed_segment:
        in_unnamed_segment = False
        next_unnamed_segment_id += 1
      node.meta["segment_prefix"] = segment_dict[node.meta["nn_module_method"]]
    else:
      if not in_unnamed_segment:
        in_unnamed_segment = True
      node.meta["segment_prefix"] = f"unnamed_{next_unnamed_segment_id}"


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

  def test_single_unnamed_segment(self):
    # Check that output and gradients are correct when there is
    # only one unnamed segment in the model.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = TestDepSegmentModule()
    m = m.to(device)
    x = torch.randn(4, 4, requires_grad=True, device=device)
    y = torch.randn(4, 4, requires_grad=True, device=device)
    lazy_scheduler = LazyScheduler(schedule=[])

    def segment_prefix_assignment_fn(gm):
      for node in gm.graph.nodes:
        node.meta["segment_prefix"] = "unnamed_seg1"

    torch._dynamo.config.lazy_scheduler_compile_fn = functools.partial(
      lazy_scheduler._compile_fn,
      segment_prefix_assignment_fn=segment_prefix_assignment_fn
    )

    self._validate(
      m,
      lazy_scheduler._compile_fx,
      x,
      y,
    )

  def test_split_module_dep_segments(self):
    # Check that GraphModule produced by Dynamo is correctly split
    # (each submodule only contains one segment)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = TestDepSegmentModule()
    m = m.to(device)
    x = torch.randn(4, 4, requires_grad=True, device=device)
    y = torch.randn(4, 4, requires_grad=True, device=device)
    lazy_scheduler = LazyScheduler(schedule=[])

    def segment_prefix_assignment_fn(gm):
      for node in gm.graph.nodes:
        assert "nn_module_method" in node.meta
        # One NN module method maps to one named segment
        node.meta["segment_prefix"] = str(node.meta["nn_module_method"])

    torch._dynamo.config.lazy_scheduler_compile_fn = functools.partial(
      lazy_scheduler._compile_fn,
      segment_prefix_assignment_fn=segment_prefix_assignment_fn
    )

    self._validate(
      m,
      lazy_scheduler._compile_fx,
      x,
      y,
    )
    self.assertEqual(
      lazy_scheduler._recorded_execution_order,
      [
        '<bound method TestDepSegmentModule.func1 of TestDepSegmentModule()>_fwd',
        '<bound method TestDepSegmentModule.func2 of TestDepSegmentModule()>_fwd',
        '<bound method TestDepSegmentModule.forward of TestDepSegmentModule()>_fwd',
        '<bound method TestDepSegmentModule.forward of TestDepSegmentModule()>_bwd',
        '<bound method TestDepSegmentModule.func2 of TestDepSegmentModule()>_bwd',
        '<bound method TestDepSegmentModule.func1 of TestDepSegmentModule()>_bwd',
      ],
    )

  def test_split_module_non_dep_segments(self):
    # Check that GraphModule produced by Dynamo is correctly split
    # (each submodule only contains one segment)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = TestNonDepSegmentModule()
    m = m.to(device)
    x = torch.randn(4, 4, requires_grad=True, device=device)
    y = torch.randn(4, 4, requires_grad=True, device=device)
    lazy_scheduler = LazyScheduler(schedule=[])

    def segment_prefix_assignment_fn(gm):
      for node in gm.graph.nodes:
        assert "nn_module_method" in node.meta
        # One NN module method maps to one named segment
        node.meta["segment_prefix"] = str(node.meta["nn_module_method"])

    torch._dynamo.config.lazy_scheduler_compile_fn = functools.partial(
      lazy_scheduler._compile_fn,
      segment_prefix_assignment_fn=segment_prefix_assignment_fn
    )

    self._validate(
      m,
      lazy_scheduler._compile_fx,
      x,
      y,
    )
    self.assertEqual(
      lazy_scheduler._recorded_execution_order,
      [
        '<bound method TestNonDepSegmentModule.func1 of TestNonDepSegmentModule()>_fwd',
        '<bound method TestNonDepSegmentModule.func2 of TestNonDepSegmentModule()>_fwd',
        '<bound method TestNonDepSegmentModule.forward of TestNonDepSegmentModule()>_fwd',
        '<bound method TestNonDepSegmentModule.forward of TestNonDepSegmentModule()>_bwd',
        '<bound method TestNonDepSegmentModule.func2 of TestNonDepSegmentModule()>_bwd',
        '<bound method TestNonDepSegmentModule.func1 of TestNonDepSegmentModule()>_bwd',
      ],
    )

  def test_segment_tagging(self):
    def _run_test(m, x, y, lazy_scheduler, segment_dict):
      torch._dynamo.config.lazy_scheduler_compile_fn = functools.partial(
        lazy_scheduler._compile_fn,
        segment_prefix_assignment_fn=functools.partial(segment_prefix_assignment_fn, segment_dict=segment_dict),
      )

      self._validate(
        m,
        lazy_scheduler._compile_fx,
        x,
        y,
      )
      return lazy_scheduler._recorded_execution_order

    # This is roughly how the official register_segment function will look like
    def register_segment(segment_dict, method, name):
      segment_dict[method] = name

    device = "cuda" if torch.cuda.is_available() else "cpu"

    m = TestNonDepSegmentModule()
    m = m.to(device)
    x = torch.randn(4, 4, requires_grad=True, device=device)
    y = torch.randn(4, 4, requires_grad=True, device=device)
    lazy_scheduler = LazyScheduler([])
    segment_dict = {}
    register_segment(segment_dict, m.func2, "func2")
    execution_order = _run_test(m, x, y, lazy_scheduler, segment_dict)
    self.assertEqual(execution_order, ['unnamed_0_fwd', 'func2_fwd', 'unnamed_1_fwd', 'unnamed_1_bwd', 'func2_bwd', 'unnamed_0_bwd'])

    m = TestNonDepSegmentModule()
    m = m.to(device)
    x = torch.randn(4, 4, requires_grad=True, device=device)
    y = torch.randn(4, 4, requires_grad=True, device=device)
    lazy_scheduler = LazyScheduler([])
    segment_dict = {}
    register_segment(segment_dict, m.func1, "func1")
    register_segment(segment_dict, m.func2, "func2")
    execution_order = _run_test(m, x, y, lazy_scheduler, segment_dict)
    self.assertEqual(execution_order, ['func1_fwd', 'func2_fwd', 'unnamed_0_fwd', 'unnamed_0_bwd', 'func2_bwd', 'func1_bwd'])

  def test_explicit_schedule(self):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = TestNonDepSegmentModule()
    m = m.to(device)
    x = torch.randn(4, 4, requires_grad=True, device=device)
    y = torch.randn(4, 4, requires_grad=True, device=device)

    # This is the explicit schedule (i.e. execution order)
    schedule = ["func1_fwd", "func2_fwd", "func2_bwd", "func1_bwd"]
    lazy_scheduler = LazyScheduler(schedule)

    # This is roughly how the official register_segment function will look like
    def register_segment(segment_dict, method, name):
      segment_dict[method] = name

    segment_dict = {}
    register_segment(segment_dict, m.func1, "func1")
    register_segment(segment_dict, m.func2, "func2")

    torch._dynamo.config.lazy_scheduler_compile_fn = functools.partial(
      lazy_scheduler._compile_fn,
      segment_prefix_assignment_fn=functools.partial(segment_prefix_assignment_fn, segment_dict=segment_dict),
    )

    self._validate(
      m,
      lazy_scheduler._compile_fx,
      x,
      y,
    )

    # Assert that execution order is as expected
    self.assertEqual(
      lazy_scheduler._recorded_execution_order,
      ['func1_fwd', 'func2_fwd', 'unnamed_0_fwd', 'unnamed_0_bwd', 'func2_bwd', 'func1_bwd'],
    )

  def test_explicit_schedule_reordering(self):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = TestNonDepSegmentModule()
    m = m.to(device)
    x = torch.randn(4, 4, requires_grad=True, device=device)
    y = torch.randn(4, 4, requires_grad=True, device=device)

    # This is the explicit schedule (i.e. execution order)
    schedule = ["func2_fwd", "func1_fwd", "func2_bwd", "func1_bwd"]
    lazy_scheduler = LazyScheduler(schedule)

    # This is roughly how the official register_segment function will look like
    def register_segment(segment_dict, method, name):
      segment_dict[method] = name

    segment_dict = {}
    register_segment(segment_dict, m.func1, "func1")
    register_segment(segment_dict, m.func2, "func2")

    torch._dynamo.config.lazy_scheduler_compile_fn = functools.partial(
      lazy_scheduler._compile_fn,
      segment_prefix_assignment_fn=functools.partial(segment_prefix_assignment_fn, segment_dict=segment_dict),
    )

    self._validate(
      m,
      lazy_scheduler._compile_fx,
      x,
      y,
    )

    # Assert that execution order is as expected
    self.assertEqual(
      lazy_scheduler._recorded_execution_order,
      ['func2_fwd', 'func1_fwd', 'unnamed_0_fwd', 'unnamed_0_bwd', 'func2_bwd', 'func1_bwd'],
    )

  def test_register_segment_hook(self):
    # Use segment hook instead of explicit schedule to specify the execution order
    """
    class SDDModule(nn.Module):
      def forward(self, x):
        return x

    sdd_m = SDDModule()
    register_segment(sdd_m.forward, is_backward=False, nth_call=0, name="sdd_fwd")


    class OverArchModule(nn.Module):
      def func1(self, x):
        return x

      def func2(self, x):
        return x

      def forward(self, x):
        x = self.func1(x)
        x = self.func2(x)
        return x

    overarch_m = OverArchModule()
    register_segment(overarch_m.func1, is_backward=True, nth_call=0, name="overarch_func2_bwd")
    # Run "sdd_fwd" right before "overarch_func2_bwd"
    register_segment_backward_pre_hook("overarch_func2_bwd", "sdd_fwd")

    # We also have `register_segment_forward_pre_hook` that can run another segment before a specific fwd segment.
    """
    pass

"""
TODO:
- Eager mode prototype (use https://docs.google.com/document/d/1jJyGiWNntkHefI2MX4dHOP8MISmsiYBRn3oWMXQyHCk/edit as design source of truth)
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
