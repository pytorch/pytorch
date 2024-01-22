import types
import traceback
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
from torch._dynamo.backends.debugging import boxed_nop
from torch._inductor.compile_fx import compile_fx as inductor_compile_fx
from torch._inductor.compile_fx import compile_fx_inner as inductor_compile_fx_inner
from torch._dynamo.output_graph import GraphCompileReason
from torch._dynamo.eval_frame import get_compiler_fn
from functorch.compile import min_cut_rematerialization_partition

next_unnamed_segment_id = 0
unnamed_segment_prefix = "__unnamed_"
unregistered_named_segment_prefix = "__unregistered_"


def extract_segment_prefix_from_gm(gm):
  segment_prefix = None
  for node in gm.graph.nodes:
    if node.op != "placeholder" and node.op != "output":
      assert "segment_prefix" in node.meta, f"node.meta: {node.meta}"
      if not segment_prefix:
        segment_prefix = node.meta["segment_prefix"]
      else:
        assert segment_prefix == node.meta["segment_prefix"], f"{segment_prefix} vs. {node.meta['segment_prefix']}"
  return segment_prefix


def compute_segment_name(segment_prefix, is_backward):
  if is_backward:
    return f"{segment_prefix}_bwd"
  else:
    return f"{segment_prefix}_fwd"


from dataclasses import dataclass

@dataclass
class Segment:
  name: str
  nn_method: callable
  is_backward: bool = False
  nth_call: int = 0


def segment_prefix_assignment_fn(gm, method_to_segment_prefix_map):
  global next_unnamed_segment_id
  in_unnamed_segment = False
  for _, node in enumerate(gm.graph.nodes):
    assert "nn_module_method" in node.meta
    print(f'node.meta["nn_module_method"]: {node.meta["nn_module_method"]}')
    print(f'id(node.meta["nn_module_method"]): {id(node.meta["nn_module_method"])}')
    print(f"method_to_segment_prefix_map: {method_to_segment_prefix_map}")
    for k, v in method_to_segment_prefix_map.items():
      print(f"k: {k}, id(k): {id(k)}, v: {v}, id(v): {id(v)}")
    if node.meta["nn_module_method"] in method_to_segment_prefix_map:
      if in_unnamed_segment:
        in_unnamed_segment = False
        next_unnamed_segment_id += 1
      node.meta["segment_prefix"] = method_to_segment_prefix_map[node.meta["nn_module_method"]]
    else:
      if not in_unnamed_segment:
        in_unnamed_segment = True
      node.meta["segment_prefix"] = f"{unnamed_segment_prefix}{str(next_unnamed_segment_id)}"


def apply_segment_prefix(gm, segment_prefix):
  for _, node in enumerate(gm.graph.nodes):
    node.meta["segment_prefix"] = segment_prefix


class AsyncFuncHandle:
  """
  We use this class to represent the function that needs to be scheduled.
  It also has methods for checking whether the function has been scheduled or completed.
  """
  def __init__(self, fn, segment, args, outs_async, scheduler):
    self.cuda_event = torch.cuda.Event()
    self.fn: Callable = fn
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
    self._scheduler().add_to_recorded_exec_order(self.segment)
    self.outs = self.fn(*args_materialized)
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
        if not node.meta["segment_prefix"].startswith(unnamed_segment_prefix):
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


def compiled_method(mod, *args, **kwargs):
  segment_nn_method = kwargs["segment_nn_method"]
  del kwargs["segment_nn_method"]
  compile_fx_fn = kwargs["compile_fx_fn"]
  del kwargs["compile_fx_fn"]
  return torch.compile(
    segment_nn_method,
    fullgraph=False,
    backend=functools.partial(compile_fx_fn, backend="aot_eager"),
  )(mod, *args, **kwargs)


class LazyScheduler:
  """
  LazyScheduler is used to decide when to schedule the execution of a graph module (based on the schedule).
  """
  def __init__(self, module, *, segments=[], schedule=[], compile_options=None):
    self._module = module
    self._segments = segments
    self._user_specified_segment_names = set([s.name for s in segments])
    # If `schedule` is empty list, it means we don't enforce the execution order.
    self._schedule = schedule
    self._compile_options = compile_options

    # Runtime states
    self._registered_segment_prefixes = set()
    self._method_to_segment_prefix_map = {}
    self._gm_to_handle_map = OrderedDict()
    self._handle_to_gm_map = OrderedDict()
    self._segment_to_gms_map = defaultdict(list)
    self._recorded_exec_order = []

  def _populate_method_to_segment_prefix_map(self):
    for segment in self._segments:
      assert not segment.name.startswith(unnamed_segment_prefix), f"User-defined segment name {segment.name} should not start with '{unnamed_segment_prefix}' (it's a reserved prefix), please rename it."
      assert segment.name.endswith("_fwd") or segment.name.endswith("_bwd")
      segment_prefix = segment.name[:-4]
      if segment.nn_method in self._method_to_segment_prefix_map:
        assert self._method_to_segment_prefix_map[segment.nn_method] == segment_prefix, f"""
Attempted to register {segment.nn_method} function with segment prefix `{segment_prefix}`,
but the function is already registered with another segment prefix `{self._method_to_segment_prefix_map[segment.nn_method]}`.
Please do not register the same function with different segment prefixes.
"""
      else:
        self._method_to_segment_prefix_map[segment.nn_method] = segment_prefix

  def _register_segment(self, segment):
    # NOTE: this code path is only invoked in eager mode

    # For each method, swap the original function with a patched function that maybe schedule the execution.
    # Solutions how to hook into the backward pass of this segment:
    # Approach 1. [Preferred] Use torch.compile (e.g. with "eager" backend) for this segment.
    #   Pros:
    #   (1) Better for handling control flow.
    #   (2) More shared code with compile mode code path.
    #   (3) Still works with graph-break FSDP since we are only tracing a single (comm) op in FSDP case.
    #   (4) Also we have to Dynamo trace to detect (and ban) global side-effects anyway.
    #   How to make nested segment work:
    #   (1) For nested compile, outer compile doesn't respect the inner compile result (it overwrite it, see https://gist.github.com/yf225/16d97499e2ecebf7e8867e9fae05e891).
    #   One way to support nested in eager is to "re-compile and split graph" when compiling the outer method. This works regardless of segment registration order.
    #   To make this possible, compiled segment needs to keep eager path so that outer compile can use it (it seems to already be the default behavior, see https://gist.github.com/yf225/16d97499e2ecebf7e8867e9fae05e891).
    # Approach 2. Annotate segment for a method using tensor subclass context manager (see test_subclass.py example).

    def _compiled_method_wrapper(_, mod, *args, **kwargs):
      kwargs["segment_nn_method"] = segment.nn_method
      kwargs["compile_fx_fn"] = self._compile_fx_for_segment
      return compiled_method(mod, *args, **kwargs)

    assert segment.name.endswith("_fwd") or segment.name.endswith("_bwd")
    segment_prefix = segment.name[:-4]
    if segment_prefix not in self._registered_segment_prefixes:
      nn_module = segment.nn_method.__self__
      method_name = segment.nn_method.__name__
      # TODO: where do we enforce each segment contains no graph break (except for nested segment)?
      # Maybe compile twice, first-time enforce `fullgraph=True`, and then do the actual split+compile?
      setattr(nn_module, method_name, types.MethodType(_compiled_method_wrapper, nn_module))
      # Put the newly bound (compiled) method into the segment prefix map.
      self._method_to_segment_prefix_map[getattr(nn_module, method_name)] = segment_prefix
      self._registered_segment_prefixes.add(segment_prefix)

  def __call__(self, *args, **kwargs):
    self._populate_method_to_segment_prefix_map()
    with torch._dynamo.config.patch(
      "lazy_scheduler_compile_fn",
      functools.partial(
        self._split_segments_and_compile,
        segment_prefix_assignment_fn=functools.partial(segment_prefix_assignment_fn, method_to_segment_prefix_map=self._method_to_segment_prefix_map),
      )
    ):
      if self._compile_options is not None:  # compile mode
        # TODO: add unit test for `torch.compile(..., backend=functools.partial(inductor_compile_fx, inner_compile=...))` for inner_compile customization.
        self._compile_options["backend"] = functools.partial(self._compile_fx_for_segment, backend=self._compile_options["backend"])
        return torch.compile(self._module, **self._compile_options)(*args, **kwargs)
      else:  # eager mode
        for segment in self._segments:
          self._register_segment(segment)
        return self._module(*args, **kwargs)

  def add_to_recorded_exec_order(self, segment):
    if len(self._recorded_exec_order) > 0 and self._recorded_exec_order[-1] == segment:
      return
    self._recorded_exec_order.append(segment)

  def reset_recorded_exec_order(self):
    self._recorded_exec_order = []

  def _compile_fx_inner_nop(self, gm, fake_tensor_inputs, **kwargs):
    def inner(*args):
      return torch.fx.Interpreter(gm).run(*args)

    return inner

  def _compile_fx_inner_for_segment(
    self,
    gm: torch.fx.GraphModule,
    *args,
    **kwargs,
  ):
    global next_unnamed_segment_id
    """
    Compiles a graph module using Inductor compile_fx_inner,
    and wraps the output compiled_fn in a LazySchedulerGraphModule to be called later.
    """
    assert isinstance(gm, torch.fx.GraphModule)

    assert "segment_prefix" in kwargs
    segment_prefix = kwargs["segment_prefix"]
    del kwargs["segment_prefix"]

    is_backward = kwargs.get("is_backward", False)

    segment_name = compute_segment_name(segment_prefix, is_backward)
    if (not segment_prefix.startswith(unnamed_segment_prefix)) and (segment_name not in self._user_specified_segment_names):
      # NOTE: if the segment prefix is not unnamed_segment_prefix, and the segment name is also not specified by user,
      # it means this segment is the named forward segment's corresponding (unnamed) backward segment, or
      # the named backward segment's corresponding (unnamed) forward segment.
      # In this case, since user doesn't care about this (fwd or bwd) segment's execution order,
      # we give it a special "unregistered but named" segment name.
      segment_prefix = f"{unregistered_named_segment_prefix}{segment_prefix}"
      apply_segment_prefix(gm, segment_prefix=segment_prefix)
      next_unnamed_segment_id += 1
      segment_name = compute_segment_name(segment_prefix, is_backward)

    # NOTE: `gm` in this function is the post-AOTAutograd fwd or bwd GraphModule,
    # each node in `gm` originally does not have segment info, and we need to re-populate it here.
    for node in gm.graph.nodes:
      node.meta["segment"] = segment_name

    assert "inner_compile_orig" in kwargs
    inner_compile_orig = kwargs["inner_compile_orig"]
    del kwargs["inner_compile_orig"]
    # Call the user-specified original compiler
    compiled_fn = inner_compile_orig(gm, *args, **kwargs)

    lazy_gm = LazySchedulerGraphModule(
      self,
      segment_name,
      gm,
      compiled_fn,
    )

    # Build segment -> GMs mapping
    self._segment_to_gms_map[segment_name].append(gm)

    return lazy_gm

  def _compile_fx_for_segment(
    self,
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    backend="inductor",
    **kwargs,
  ):
    segment_prefix = extract_segment_prefix_from_gm(gm)
    compiler_fn = None
    assert backend == "aot_eager"
    if backend == "aot_eager":
      inner_compile_orig = self._compile_fx_inner_nop
      compiler_fn = aot_autograd(
        fw_compiler=functools.partial(self._compile_fx_inner_for_segment, segment_prefix=segment_prefix, inner_compile_orig=inner_compile_orig),
        bw_compiler=functools.partial(self._compile_fx_inner_for_segment, segment_prefix=segment_prefix, inner_compile_orig=inner_compile_orig, is_backward=True),
        partition_fn=min_cut_rematerialization_partition,
      )
    else:
      # `inner_compile` can be passed in via `torch.compile(m, functools.partial(inductor_compile_fx, inner_compile=...))`.
      # It's the custom compiler for each fwd or bwd GraphModule. Default is inductor_compile_fx_inner.
      inner_compile_orig = kwargs.get("inner_compile", inductor_compile_fx_inner)
      kwargs.update({
        "inner_compile": functools.partial(self._compile_fx_inner_for_segment, segment_prefix=segment_prefix, inner_compile_orig=inner_compile_orig)
      })
      compiler_fn = get_compiler_fn(backend)

    return compiler_fn(gm, example_inputs, **kwargs)

  def _split_segments_and_compile(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor], backend_compile_fn, segment_prefix_assignment_fn):
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
