import copy
import types
import traceback
import torch
import torch.utils._pytree as pytree
from torch.testing._internal.common_utils import TestCase as TorchTestCase
from torch._dynamo import disable
import functools
import itertools
from typing import Any, Optional, Dict, Callable, List
from torch._subclasses.async_tensor import AsyncTensor, get_fake_mode, TensorContainer
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
from torch._inductor.codecache import CompiledFxGraph
import pprint

next_unnamed_segment_id = 0
unnamed_segment_prefix = "__unnamed_"
unregistered_named_segment_prefix = "__unregistered_"


def is_call_func_node(node):
  return node.op in ("call_function", "call_method", "call_module")


def extract_segment_prefix_from_gm(gm):
  segment_prefix = None
  for node in gm.graph.nodes:
    if is_call_func_node(node):
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
  nth_call: int = 0
  backend: Any = None


def remove_generated_wait_if_explicit_wait_exists(gm):
  node_list = list(gm.graph.nodes)
  for i in range(len(node_list)):
    n = node_list[i]
    if (
      is_call_func_node(n) \
      and n.target is torch.ops._c10d_functional.wait_tensor \
      and i < len(node_list) - 1 \
      and is_call_func_node(node_list[i+1]) \
      and node_list[i+1].target is torch.ops._c10d_functional.wait_tensor \
      and node_list[i+1].args[0] == n  # if input of the 2nd wait_tensor (explicit) is the output of 1st wait_tensor (generated)
    ):
      assert "nn_module_method" in node_list[i+1].meta  # explicit wait_tensor op should always have nn_module_method metadata
      generated_wait_out = n
      comm_out = generated_wait_out.args[0]
      explicit_wait_out = node_list[i+1]
      comm_out.replace_all_uses_with(explicit_wait_out)
      explicit_wait_out.args = (comm_out,)
      generated_wait_out.replace_all_uses_with(explicit_wait_out)
      gm.graph.erase_node(generated_wait_out)
      gm.graph.lint()
      gm.recompile()


def segment_prefix_assignment_fn(gm):
  global next_unnamed_segment_id
  method_to_segment_prefix_map = LazyScheduler._current_scheduler._method_to_segment_prefix_map
  in_unnamed_segment = False
  remove_generated_wait_if_explicit_wait_exists(gm)
  for _, node in enumerate(gm.graph.nodes):
    if is_call_func_node(node):
      assert "nn_module_method" in node.meta
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
  def __init__(self, fn, segment_name, args, outs_fake, scheduler):
    self.cuda_event = None
    self.fn: Callable = fn
    self.args = args
    self.outs_fake = outs_fake
    self.scheduler = weakref.ref(scheduler)
    self.outs_async = tuple(
      AsyncTensor(
        # TODO: the handling for `out_fake is None` case seems dicey here.
        fake_tensor=out_fake if out_fake is not None else get_fake_mode().from_tensor(torch.zeros([])),
        handle=None,
        materialized_tensor_container=TensorContainer()
      ) for out_fake in self.outs_fake
    )
    self.outs = None
    self.segment_name = segment_name
    self.args_materialized = None

    assert not any(arg is None for arg in args)
    for out_async in self.outs_async:
      out_async.set_handle(self)

  def schedule(self):
    # make sure to schedule only once
    scheduler = self.scheduler()
    if self.cuda_event is not None:
      print(f"handle is already scheduled! segment name: {self.segment_name}")
      return
    else:
      print(f"scheduling handle! segment name: {self.segment_name}")
    self.cuda_event = torch.cuda.Event()
    assert not any(arg is None for arg in self.args)
    AsyncTensor.wait_until_materialized(self.args)
    assert not any(arg is None for arg in self.args)
    # Since we are doing real computations here, we should disable fake mode if any.
    # with torch.fx.experimental.proxy_tensor.maybe_disable_fake_tensor_mode():
    args_materialized = pytree.tree_map_only(AsyncTensor, lambda x: x.get_materialized_tensor(), self.args)
    args_materialized = pytree.tree_map(lambda x: x.detach(), args_materialized)
    self.args_materialized = args_materialized
    scheduler.record_execution(self.segment_name)
    with torch.profiler.record_function(f"{self.segment_name} (LazyScheduler)"):
      if isinstance(self.fn, CompiledFxGraph):
        outs = self.fn(list(self.args_materialized))
      else:
        # TODO: when do we hit this case?
        outs = self.fn(self.args_materialized)
    self.outs = [out.get_materialized_tensor() if isinstance(out, AsyncTensor) else out for out in outs]
    self.cuda_event.record()

  def wait_for_completion(self):
    if self.cuda_event is None:
      raise RuntimeError("Cannot wait for completion for a handle that's not scheduled yet!")
    self.cuda_event.synchronize()
    for out, out_async in zip(self.outs, self.outs_async):
      assert (isinstance(out, torch.Tensor) and not isinstance(out, AsyncTensor)) or out is None, f"out: {out}, type(out): {type(out)}"
      out_async.materialize_with_value(out if out is not None else torch.zeros([]))

  def is_completed(self):
    cuda_event = self.cuda_event
    return cuda_event.query() if cuda_event is not None else False

  def scheduler(self):
    scheduler = self.scheduler()
    assert scheduler is not None
    return scheduler


def split_module_based_on_segment_info(gm: torch.fx.GraphModule):
  known_segments = []
  for node in gm.graph.nodes:
    if is_call_func_node(node) and (len(known_segments) == 0 or node.meta["segment_prefix"] != known_segments[-1]):
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
      if is_call_func_node(node):
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
  def __init__(self, segment_name, gm, compiled_fn):
    super().__init__()
    self.segment_name = segment_name
    self.gm = gm
    self.compiled_fn = compiled_fn

  def __call__(self, *args):
    assert self.compiled_fn is not None
    return LazyScheduler._current_scheduler.maybe_run(self.gm, self.compiled_fn, self.segment_name, *args)


def _compile_fx_inner_for_graph_in_segment(
  gm: torch.fx.GraphModule,
  *args,
  **kwargs,
):
  global next_unnamed_segment_id
  """
  Compiles a graph module for this graph (which is from a segment),
  and wraps the output compiled_fn in a LazySchedulerGraphModule to be called later.

  NOTE: this function might be cached and is not guaranteed to re-run in subsequent calls.
  """
  assert isinstance(gm, torch.fx.GraphModule)

  assert "segment_prefix" in kwargs
  segment_prefix = kwargs["segment_prefix"]
  del kwargs["segment_prefix"]

  lazy_scheduler = LazyScheduler._current_scheduler

  is_backward = kwargs.get("is_backward", False)

  segment_name = compute_segment_name(segment_prefix, is_backward)
  if (not segment_prefix.startswith(unnamed_segment_prefix)) and (segment_name not in lazy_scheduler._user_specified_segment_names):
    # NOTE: if the segment prefix is not equal to `unnamed_segment_prefix`, and the segment name is also not specified by user,
    # it means this segment is the named forward segment's corresponding backward segment, or
    # the named backward segment's corresponding forward segment.
    # In this case, since user doesn't care about this corresponding (fwd or bwd) segment's execution order,
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
    segment_name,
    gm,
    compiled_fn,
  )

  # Build segment -> GMs mapping
  if segment_name not in lazy_scheduler._segment_to_gms_map:
    lazy_scheduler._segment_to_gms_map[segment_name] = []
  lazy_scheduler._segment_to_gms_map[segment_name].append(gm)

  return lazy_gm


def _compile_fx_inner_boxed_nop(gm, example_inputs, **kwargs):
  def run(args):
    out = torch.fx.Interpreter(gm).boxed_run(list(args))
    return out

  run._boxed_call = True
  return run


def extract_method_name(formatted_name):
  return formatted_name.split("###")[-1]



class LazyScheduler:
  """
  LazyScheduler is used to decide when to schedule the execution of a graph module (based on the schedule).
  """

  # Assumption: we only run one instance of LazyScheduler per process.
  # NOTE: this is not a hard requirement, but to implement multi-LazyScheduler,
  # we need to be careful of _compile_fx_inner_for_graph_in_segment caching by torch.compile
  # and make sure we don't reuse old LazyScheduler instance.
  # Also, is multi-LazyScheduler ever useful in practice?
  _current_scheduler = None

  def __init__(self, module, *, segments=[], schedule=[], compile_options=None):
    self._module = module
    self._segments = segments
    self._schedule = schedule
    self._compile_options = compile_options
    # defaults to aot_eager for eager mode LazyScheduler segment backend
    self._user_specified_segment_names = set([s.name for s in segments])
    self._default_backend = compile_options["backend"] if compile_options is not None else "aot_eager"
    self._segment_prefix_to_backend = {}
    self._registered_segment_prefixes = set()
    self._method_to_segment_prefix_map = {}
    self._gm_to_handle_map = OrderedDict()
    self._segment_to_gms_map = OrderedDict()
    self._recorded_execution_order = []
    self._recorded_delayed = set()
    self._has_inplace_op_segment_prefixes = set()
    self._access_mod_attr_or_glb_var_segment_prefixes = set()
    self._compile_starting_point_nn_method = None
    LazyScheduler._current_scheduler = self
    # TODO: reset `next_unnamed_segment_id` to 0 after every iteration

  # TODO: why do we need `_method_to_segment_prefix_map` to be an instance attribute instead of a local var?
  def _prepare_segments(self, compile_options):
    segment_name_to_segment = {}
    for segment in self._segments:
      if segment.backend is None:
        segment.backend = self._default_backend
      segment_name_to_segment[segment.name] = segment

    for segment in self._segments:
      assert not segment.name.startswith(unnamed_segment_prefix), \
        f"User-defined segment name {segment.name} should not start with '{unnamed_segment_prefix}' (it's a reserved prefix), please rename it."
      assert not segment.name.startswith(unregistered_named_segment_prefix), \
        f"User-defined segment name {segment.name} should not start with '{unregistered_named_segment_prefix}' (it's a reserved prefix), please rename it."
      assert segment.name.endswith("_fwd") or segment.name.endswith("_bwd"), \
        f"Segment name must end with '_fwd' or '_bwd'. Violating segment name: {segment.name}."
      segment_prefix = segment.name[:-4]

      # Check backend compatibility
      if segment.backend == "eager":
        assert segment.name.endswith("_fwd"), f"`{segment.name}` tries to use 'eager' backend, but only forward segment can use 'eager' backend."
        assert f"{segment_prefix}_bwd" not in segment_name_to_segment, f"""
If forward segment `{segment.name}` uses 'eager' backend, this segment must be forward-only and cannot have a corresponding backward segment.
But we found a backward segment `{segment_prefix}_bwd` for this forward segment.
"""
      else:
        fwd_segment_name = f"{segment_prefix}_fwd"
        bwd_segment_name = f"{segment_prefix}_bwd"
        if fwd_segment_name in segment_name_to_segment and bwd_segment_name in segment_name_to_segment:
          assert segment_name_to_segment[fwd_segment_name].backend == segment_name_to_segment[bwd_segment_name].backend, f"""
If both forward and backward segments are registered for an NN method, both segments must use the same backend.
Violating segments:
- `{fwd_segment_name}` (uses `{segment_name_to_segment[fwd_segment_name].backend}` backend)
- `{bwd_segment_name}` (uses `{segment_name_to_segment[bwd_segment_name].backend}` backend)
"""

      if segment.nn_method in self._method_to_segment_prefix_map:
        assert self._method_to_segment_prefix_map[segment.nn_method] == segment_prefix, f"""
Attempted to register {segment.nn_method} function with segment prefix `{segment_prefix}`,
but the function is already registered with another segment prefix `{self._method_to_segment_prefix_map[segment.nn_method]}`.
Please do not register the same function with different segment prefixes.
"""
      else:
        self._method_to_segment_prefix_map[segment.nn_method] = segment_prefix
      self._segment_prefix_to_backend[segment_prefix] = segment.backend

  def _register_segment(self, segment):
    # NOTE: this code path is only invoked in eager mode

    # For each method, swap the original function with a patched function that maybe schedule the execution.
    # Solutions how to hook into the backward pass of this segment:
    # Approach 1 [Chosen]: Use torch.compile (e.g. with "eager" or "aot_eager" backend) for this segment.
    #   Pros:
    #   (1) Better for handling control flow.
    #   (2) More shared code with compile mode code path.
    #   (3) Lower runtime overhead than tensor subclass approach.
    #   Cons:
    #   (1) The segment has to be compile-able, which might not always be true.
    # Approach 2: Annotate segment for a method using tensor subclass context manager (see test_subclass.py example).
    # Approach 3: Wrap each segment with a custom autograd function, and customize its backward pass to let LazyScheduler control the execution.
    #   (similar to RegisterPostBackwardFunction in ppFSDP, just hook it to a module method that requires grad) Then test test_explicit_schedule_reordering_fwd_and_bwd_segments again.

    method_name = extract_method_name(segment.nn_method.__name__)
    stashed_eager_method_name = f"{method_name}_eager"
    compiled_method_name = f"{method_name}_compiled"

    def _private_patched_nn_method(mod, *args, **kwargs):
      eager_nn_method = getattr(mod, stashed_eager_method_name)
      # TODO: can we unify `_compile_starting_point_nn_method` with `nn_method_stack`?
      if self._compile_starting_point_nn_method is None:
        # If we are calling this method directly in eager mode (and not from an outer method being compiled),
        # then we compile this method.
        self._compile_starting_point_nn_method = eager_nn_method
        compiled_nn_method = getattr(nn_module, compiled_method_name)
        out = compiled_nn_method(*args, **kwargs)
        self._compile_starting_point_nn_method = None
        return out
      else:
        # Otherwise, if we are calling this method from an outer method being compiled,
        # then we take the eager mode path for this method so that outer method compile can continue successfully.
        return eager_nn_method(*args, **kwargs)

    # NOTE: How to do nested segments
    # Fact: If we put a compiled region within a compiled region, outer compile doesn't respect the inner compile result (it overwrite it, see https://gist.github.com/yf225/16d97499e2ecebf7e8867e9fae05e891).
    # One way to support nested in eager is to "re-compile and split graph" when compiling the outer method. This works regardless of segment registration order.
    # To make this possible, compiled segment needs to keep eager path so that outer compile can use it (it seems to already be the default behavior, see https://gist.github.com/yf225/16d97499e2ecebf7e8867e9fae05e891).

    assert segment.name.endswith("_fwd") or segment.name.endswith("_bwd")
    segment_prefix = segment.name[:-4]
    # NOTE: Only register (including doing method swapping) if the segment has not been registered yet.
    if segment_prefix not in self._registered_segment_prefixes:
      print(f"registering segment: {segment}")
      nn_module = segment.nn_method.__self__
      print(f"nn_module: {nn_module}")
      print(f"type(nn_module): {type(nn_module)}")
      print(f"id(nn_module): {id(nn_module)}")

      # TODO: always redo compilation and do not cache previous compiled method from previous iteration? are we already doing this?

      eager_method = getattr(nn_module, method_name)
      if not hasattr(nn_module, stashed_eager_method_name):
        setattr(nn_module, stashed_eager_method_name, eager_method)
      setattr(nn_module, compiled_method_name, torch.compile(
        # TODO: change this to `getattr(mod, stashed_eager_method_name)`?
        segment.nn_method,
        fullgraph=False,
        backend=self._compile_fx_for_graph_in_segment,
      ))
      _private_patched_nn_method.__name__ = f"_compiled_wrapper###{nn_module.__class__.__name__}###{method_name}"
      setattr(nn_module, method_name, types.MethodType(_private_patched_nn_method, nn_module))
      # Put the newly bound (compiled) method into the segment prefix map.
      bound_nn_method = getattr(nn_module, method_name)
      print(f"bound_nn_method.__name__: {bound_nn_method.__name__}")
      print(f"id(bound_nn_method): {id(bound_nn_method)}")
      self._method_to_segment_prefix_map[getattr(nn_module, method_name)] = segment_prefix
      self._registered_segment_prefixes.add(segment_prefix)
      print(f"self._registered_segment_prefixes: {self._registered_segment_prefixes}")
    else:
      print(f"already registered segment: {segment}")

  def __call__(self, *args, **kwargs):
    self._prepare_segments(self._compile_options)
    with torch._dynamo.config.patch(
      "lazy_scheduler_compile_fn",
      functools.partial(
        self._split_segments_and_compile,
        # TODO: avoid caching `method_to_segment_prefix_map` here too! use LazyScheduler singleton.
        segment_prefix_assignment_fn=segment_prefix_assignment_fn,
      )
    ):
      if self._compile_options is not None:  # compile mode
        # TODO: add unit test for `torch.compile(..., backend=functools.partial(inductor_compile_fx, inner_compile=...))` for inner_compile customization.
        # TODO: allow calling "eager" (fwd only) or "aot_eager" (fwd and bwd) region within the "inductor" region, under compile mode. (How to do this during graph splitting?)
        compile_options_without_backend = {k: v for k, v in self._compile_options.items() if k != "backend"}
        outs = torch.compile(self._module, backend=self._compile_fx_for_graph_in_segment, **compile_options_without_backend)(*args, **kwargs)
      else:  # eager mode
        for segment in self._segments:
          self._register_segment(segment)
        outs = self._module(*args, **kwargs)
    return outs

  def record_execution(self, segment_name):
    print(f"Adding segment {segment_name} to recorded execution order.")
    if len(self._recorded_execution_order) > 0 and self._recorded_execution_order[-1] == segment_name:
      print("record_execution early return")
      return
    print("record_execution add to _recorded_execution_order list")
    self._recorded_execution_order.append(segment_name)

  def get_recorded_execution_order(self):
    return [
      s for s in self._recorded_execution_order
      if (not s.startswith("__unnamed_")) and not (s.startswith("__unregistered_"))
    ]

  def _compile_fx_for_graph_in_segment(
    self,
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    **kwargs,
  ):
    segment_prefix = extract_segment_prefix_from_gm(gm)

    for node in gm.graph.nodes:
      if is_call_func_node(node) and str(node.target).endswith("_"):  # likely in-place op
        self._has_inplace_op_segment_prefixes.add(segment_prefix)

    backend = self._segment_prefix_to_backend.get(segment_prefix, self._default_backend)
    compiler_fn = None
    if backend == "eager":
      inner_compile_orig = _compile_fx_inner_boxed_nop
      compiler_fn = functools.partial(_compile_fx_inner_for_graph_in_segment, segment_prefix=segment_prefix, inner_compile_orig=inner_compile_orig)
    elif backend == "aot_eager":
      inner_compile_orig = _compile_fx_inner_boxed_nop
      compiler_fn = aot_autograd(
        fw_compiler=functools.partial(_compile_fx_inner_for_graph_in_segment, segment_prefix=segment_prefix, inner_compile_orig=inner_compile_orig),
        bw_compiler=functools.partial(_compile_fx_inner_for_graph_in_segment, segment_prefix=segment_prefix, inner_compile_orig=inner_compile_orig, is_backward=True),
        partition_fn=min_cut_rematerialization_partition,
      )
    elif backend == "inductor":
      # `inner_compile` can be passed in via `torch.compile(m, functools.partial(inductor_compile_fx, inner_compile=...))`.
      # It's the custom compiler for each fwd or bwd GraphModule. Default is inductor_compile_fx_inner.
      inner_compile_orig = kwargs.get("inner_compile", inductor_compile_fx_inner)
      kwargs.update({
        "inner_compile": functools.partial(_compile_fx_inner_for_graph_in_segment, segment_prefix=segment_prefix, inner_compile_orig=inner_compile_orig)
      })
      compiler_fn = get_compiler_fn("inductor")
    else:
      raise RuntimeError(f"Unsupported backend: {backend}")

    return compiler_fn(gm, example_inputs, **kwargs)

  def _split_segments_and_compile(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor], backend_compile_fn, segment_prefix_assignment_fn):
    segment_prefix_assignment_fn(gm)
    split_gm = split_module_based_on_segment_info(gm)

    submod_compiler = SubmoduleReplacer(split_gm, backend_compile_fn)
    submod_compiler.run(*example_inputs)
    split_gm.recompile()

    return split_gm

  def maybe_run(self, gm, compiled_fn, cur_segment_name, *args):
    """
    Decides whether to run the graph module based on the schedule.

    Always immediately returns AsyncTensor as output, and the AsyncTensor will be populated
    when the graph module is eventually executed.
    """
    # Create the handle and the async tensors
    assert not any(arg is None for arg in args)
    args_fake = []
    for arg in args:
      if isinstance(arg, AsyncTensor):
        args_fake.append(arg._fake_tensor)
      elif isinstance(arg, torch.Tensor):
        args_fake.append(get_fake_mode().from_tensor(arg))
    with get_fake_mode():
      outs_fake = gm(*args_fake)

    if gm in self._gm_to_handle_map:
      cur_handle = self._gm_to_handle_map[gm]
    else:
      cur_handle = AsyncFuncHandle(
        compiled_fn, cur_segment_name, args=args,
        outs_fake=outs_fake,
        scheduler=self,
      )
      self._gm_to_handle_map[gm] = cur_handle

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
      _segment_name = self._schedule[_next_segment_index]
      if _segment_name not in self._segment_to_gms_map:
        all_preceding_graph_handles_are_created = False
        break
      else:
        for g in self._segment_to_gms_map[_segment_name]:
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
      # If not all preceding graph handles are created, it means current graph is delayed
      # and we don't schedule the current graph yet.
      self._recorded_delayed.add(cur_segment_name)
      return cur_handle.outs_async
    else:
      # If all preceding graph handles are created, then we schedule all of them,
      # and then schedule the current graph.
      for handle in all_preceding_graph_handles:
        handle.schedule()
      cur_handle.schedule()
      return cur_handle.outs_async

  def debug(self):
    if len(self._schedule) != len(self.get_recorded_execution_order()) or set(self._schedule) != set(self.get_recorded_execution_order()):
      raise RuntimeError(f"""
The LazyScheduler's actual execution order has different number of segments or different segments compared to the schedule.

- Schedule: {self._schedule}. Length: {len(self._schedule)}.
- Recorded execution order: {self.get_recorded_execution_order()}. Length: {len(self.get_recorded_execution_order())}.

Please update the schedule to have the same segments as the recorded execution order.
""")

    debug_mode_issue_msgs = []
    debug_mode_msg = """
=======================================================================================================================
                                        LazyScheduler Debug Mode Suggestions
=======================================================================================================================

"""

    if self._schedule != self.get_recorded_execution_order():
      debug_mode_issue_msgs.append(f"""
Issue: The LazyScheduler's actual execution order is not the same as the schedule.

Suggestion: This usually happens when a segment A's output (or a tensor it in-place mutates) is immediately used by
downstream code, causing segment A to not be able to be delayed after it.

For example, if the schedule is [B, C, A], but the recorded execution order is [B, A, C], then we know that
some code between B (exclusive) and C (inclusive) is using A's output which causes A to not be able to be delayed after C.

For this model, we have:
- Schedule: {self._schedule}. Length: {len(self._schedule)}.
- Recorded execution order: {self.get_recorded_execution_order()}. Length: {len(self.get_recorded_execution_order())}.

Best guess on the potential dependencies causing the issue:
{find_delayed_seg_dependencies_best_effort(self._schedule, self.get_recorded_execution_order())}

With the above information, please audit the code of all of delayed segments and all other segments that have
data dependency with these delayed segments, to find the root cause.
-----------------------------------------------------------------------------------------------------------------------
""")

    if len(debug_mode_issue_msgs) > 0:
      debug_mode_msg += """
Identified Issues:

"""
      debug_mode_msg += "\n\n".join(debug_mode_issue_msgs)

    debug_mode_msg += f"""
Common Questions:

Q: I found numerical mismatch between LazyScheduler prod mode and LazyScheduler-disabled mode. How to debug?

A: Usually LazyScheduler numerical mismatch is due to:
- A shared tensor (via module attribute, global variable, or input tensor) is mutated by another function and read by one of the delayed segments.
- A shared tensor (via module attribute, global variable, or input tensor) is mutated by one of the delayed segments and read by another function.

For this model, we have:
- Delayed segments: {sorted(list(self._recorded_delayed))}
- Segments that contain in-place mutation ops: {sorted(list(self._has_inplace_op_segment_prefixes))}
- Segments that access module attribute or global variable: {sorted(list(self._access_mod_attr_or_glb_var_segment_prefixes))}

With the above information, please audit the code of all of delayed segments and all other functions that have data dependency with these delayed segments.
Pay special attention to data dependency via shared tensor (module attribute, global variable, or input tensor).

If you are still unable to find the root cause, the best way to debug it is to remove segments one by one from your LazyScheduler schedule and re-run your model.
If you find that removing a specific segment fixes the issue, then audit that segment and other functions that have data dependency with it.
-----------------------------------------------------------------------------------------------------------------------
"""
    raise RuntimeError(debug_mode_msg)

def find_delayed_seg_dependencies_best_effort(schedule_orig, recorded_execution_order_orig):
  dependencies = {}
  schedule = copy.deepcopy(schedule_orig)
  recorded_execution_order = copy.deepcopy(recorded_execution_order_orig)
  assert len(schedule) == len(recorded_execution_order)
  while schedule != recorded_execution_order:
    # Algorithm:
    # - From end to beginning, find the first segment that is executed earlier than expected.
    # - The next segment executed after it is the segment that uses its output.
    # - Remove both segments from schedule and recorded execution order.
    # - Repeat this process, until schedule and recorded execution order are exactly the same.
    for index, seg in reversed(list(enumerate(schedule))):
      index_of_seg_in_execution_order = recorded_execution_order.index(seg)
      if index_of_seg_in_execution_order < index:  # segment is early materialized
        dep_seg = recorded_execution_order[index_of_seg_in_execution_order+1]
        dep_seg_prev = recorded_execution_order[index_of_seg_in_execution_order-1] if index_of_seg_in_execution_order-1 >= 0 else None
        dependencies[seg] = (dep_seg_prev, dep_seg)
        schedule.remove(seg)
        schedule.remove(dep_seg)
        recorded_execution_order.remove(seg)
        recorded_execution_order.remove(dep_seg)
        break
  dep_msgs = []
  for delayed_seg, (dep_seg_prev, dep_seg) in dependencies.items():
    if dep_seg_prev is None:
      dep_msgs.append(f"- The code starting from end of `{delayed_seg}` (exclusive) to end of `{dep_seg}` (inclusive) depends on output of `{delayed_seg}`")
    else:
      dep_msgs.append(f"- The code starting from end of `{dep_seg_prev}` (exclusive) to end of `{dep_seg}` (inclusive) depends on output of `{delayed_seg}`")
  return "\n".join(dep_msgs)

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
    # TODO: this doesn't seem to show in log, need to figure out why.
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
