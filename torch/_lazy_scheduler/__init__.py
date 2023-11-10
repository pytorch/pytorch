# Older version that tries to reorder FWD/BWD ops: https://gist.github.com/yf225/10a6f559daadc7f1b51cc197d82720cb

import torch
import itertools
from typing import Optional, Dict, Callable
from torch._subclasses.fake_tensor import FakeTensorMode
from collections import defaultdict, OrderedDict
import weakref
import threading
from torch._inductor.compile_fx import compile_fx_inner

fake_mode = FakeTensorMode()

class Segment:
  _cur_segment: Optional[str] = None
  _func_to_segment_mapping: Dict[Callable, str] = {}
  _unnamed_counter = itertools.count(start=0)

  @classmethod
  def get_next_unnamed_segment(cls):
    return f"unnamed_{next(cls._unnamed_counter)}"


class NoneWithUsageDetector:
  def __init__(self):
    pass

  def __getattribute__(self, attr):
    # breakpoint()
    return super().__getattribute__(attr)


class AsyncTensor(torch.Tensor):
  def __new__(cls, fake_tensor):
    r = torch.Tensor._make_wrapper_subclass(
      cls,
      fake_tensor.size(),
      dtype=fake_tensor.dtype,
      device=fake_tensor.device,
      layout=fake_tensor.layout,
      requires_grad=fake_tensor.requires_grad,
    )
    r._materialized_tensor = NoneWithUsageDetector()
    r._handle = None
    r._fake = fake_tensor
    return r

  # NOTE: Any non-PyTorch reads or mutations in eager region will need to access one of these APIs: `.data_ptr` / `.storage` / `.data`.
  # We materialize the tensor before executing those calls, so that non-PyTorch reads or mutations in eager region still work normally.

  def async_repr(self):
    return f"AsyncTensor({self._handle}, {self._fake})"

  def __repr__(self):
    # NOTE: `print(tensor)` goes through this
    if self._handle is not None:
      AsyncTensor.wait_until_materialized([self])
      return self._materialized_tensor.__repr__()
    else:
      return self.async_repr()

  # TODO: likely need to define __div__ etc. between Tensor and AsyncTensor
  # TODO: implement torch.allclose
  # TODO: define all these Tensor methods: https://www.internalfb.com/code/fbsource/[0edfc6f1cc1db161177cc596f8a6dea83ad3df52]/fbcode/caffe2/torch/csrc/autograd/python_variable.cpp?lines=1515

  # def __add__(self, other):
  #   TODO: between Tensor and AsyncTensor

  def __format__(self, format_spec):
    # NOTE: `print(f"{tensor}")` goes through this
    AsyncTensor.wait_until_materialized([self])
    return self._materialized_tensor.__format__(format_spec)

  def __getattribute__(self, attr):
    print(f"getattr: {attr}")
    if attr in dir(torch.Tensor):
      if self._handle is not None:
        AsyncTensor.wait_until_materialized([self])
        print(f"_materialized_tensor: {self._materialized_tensor}")
        return getattr(self._materialized_tensor, attr)
      else:
        raise Exception(f"Getting {attr} on an AsyncTensor that doesn't have handle is not allowed")
    else:
      return super().__getattribute__(attr)

  def __setattr__(self, attr, value):
    print(f"setattr: {attr} -> {value}")
    if attr in dir(torch.Tensor):
      if self._handle is not None:
        AsyncTensor.wait_until_materialized([self])
        return setattr(self._materialized_tensor, attr, value)
      else:
        raise Exception(f"Setting {attr} on an AsyncTensor that doesn't have handle is not allowed")
    else:
      return super().__setattr__(attr, value)

  def handle(self):
    assert self._handle is not None
    handle = self._handle()
    assert handle is not None
    return handle

  def set_handle(self, handle):
    self._handle = weakref.ref(handle)

  def set_materialized_tensor(self, materialized_tensor):
    self._materialized_tensor = materialized_tensor

  # NOTE: Any PyTorch reads or mutations in eager region will go through __torch_dispatch__, so we materialize the tensor here.
  @classmethod
  def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
    # TODO: handle tuple / list / etc.
    # TODO: do the same for kwargs
    AsyncTensor.wait_until_materialized(args)
    return func(*args, **kwargs)

  @staticmethod
  def check_materialized(async_tensors):
    # breakpoint()
    all_materialized = True
    for t in async_tensors:
      if isinstance(t, AsyncTensor) and isinstance(t._materialized_tensor, NoneWithUsageDetector):
          all_materialized = False
          break
    return all_materialized

  @staticmethod
  def wait_until_materialized(async_tensors):
    # breakpoint()
    for async_tensor in async_tensors:
      if not AsyncTensor.check_materialized([async_tensor]):
        # NOTE: recursively schedule the deps first
        print(f"waiting on deps for {async_tensor.async_repr()}")
        AsyncTensor.wait_until_materialized([async_tensor.handle().args])
        print(f"going to schedule {async_tensor.async_repr()}")
        async_tensor.handle().schedule()
        print(f"waiting for completion {async_tensor.async_repr()}")
        async_tensor.handle().wait_for_completion()
        print(f"done waiting for completion {async_tensor.async_repr()}")


# TODO: dedup info and simplify all classes below
class AsyncFuncHandle:
  _gm_to_handle_mapping: Dict[torch.fx.GraphModule, "AsyncFuncHandle"] = {}

  def __init__(self, compiled_fn, segment, args, outs_async, scheduler):
    self.cuda_event = torch.cuda.Event()
    self.compiled_fn: Callable = compiled_fn
    self.segment: str = segment  # for bookkeeping
    # dependency graph is built implicitly as we run the program
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
    print(f"self.args: {self.args}")
    print(f"handle id: {id(self)}, scheduling {gm} ... with id {id(gm)} ... from segment: {self.segment}")
    self._scheduler().add_to_recorded_execution_order(self.segment)
    self.outs = self.compiled_fn(list(self.args))
    self.cuda_event.record()

  def wait_for_completion(self):
    print("wait_for_completion is called")
    self.cuda_event.synchronize()
    for out, out_async in zip(self.outs, self.outs_async):
      print(f"out: {out}")
      print("set materialized tensor")
      out_async.set_materialized_tensor(out)

  def is_completed(self):
    return self.cuda_event.query()

  def scheduler(self):
    scheduler = self._scheduler()
    assert scheduler is not None
    return scheduler


class LazyGraphModule(torch.nn.Module):  # TODO: better name?
  def __init__(self, scheduler, segment, gm, compiled_fn):
    super().__init__()
    self.scheduler = scheduler
    self.segment = segment  # bookkeeping
    self.gm = gm
    self.compiled_fn = compiled_fn

  def __call__(self, *args):
    assert self.compiled_fn is not None
    return self.scheduler.maybe_run(self.gm, self.compiled_fn, self.segment, *args)


class LazyScheduler:
  def __init__(self, schedule):
    mapped_segments = set(Segment._func_to_segment_mapping.values())
    segments_in_schedule = set(schedule)
    assert mapped_segments == segments_in_schedule
    self._schedule = schedule
    self._gm_to_handle_map = OrderedDict()
    self._handle_to_gm_map = OrderedDict()
    self._segment_to_gms_map = defaultdict(list)
    # self._next_segment_index = 0
    self._recorded_execution_order = []

  def add_to_recorded_execution_order(self, segment):
    if len(self._recorded_execution_order) > 0 and self._recorded_execution_order[-1] == segment:
      return
    self._recorded_execution_order.append(segment)

  def is_expected_execution_order_for_named_segments(self, expected_execution_order_for_named_segments):
    recorded_execution_order_for_named_segments = [s for s in self._recorded_execution_order if s in self._schedule]
    return recorded_execution_order_for_named_segments == expected_execution_order_for_named_segments, \
      f"{recorded_execution_order_for_named_segments} vs. {expected_execution_order_for_named_segments}"

  # matches compile_fx_inner signature
  def compile(
    self,
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
    # breakpoint()
    known_segments = []
    print(f"gm.graph: {gm.graph}")
    segment = None
    for node in gm.graph.nodes:
      # Look up the NN module method in the segment map
      method = node.meta.get('nn_module_method', None)
      if method is not None and method in Segment._func_to_segment_mapping:
        segment = Segment._func_to_segment_mapping[method]
      else:
        # The logic here means that we don't have a single unnamed segment
        # for all graphs in between the two named segments.
        # This should not matter, because users should not care about unnamed segments in general.
        segment = Segment.get_next_unnamed_segment()
      node.meta['segment'] = segment

      if len(known_segments) == 0 or node.meta["segment"] != known_segments[-1]:
        known_segments.append(node.meta["segment"])

    print(f"known_segments: {known_segments}")

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

    print(f"gm.graph: {gm.graph}")
    print(f"gm_after_split: {gm_after_split.graph}")
    for name, sub_gm in gm_after_split.named_children():
      print(f"subgraph {name}: {str(sub_gm.graph)}")

    print(f"gm.graph: {gm.code}")
    print(f"gm_after_split: {gm_after_split.code}")
    for name, sub_gm in gm_after_split.named_children():
      print(f"subgraph {name}: {str(sub_gm.code)}")

    for name, sub_gm in gm_after_split.named_children():
      print(f"subgraph {name}: {str(sub_gm.graph)}")
      for node in sub_gm.graph.nodes:
        print(f"node: {node}")
        if "segment" not in node.meta:
          print(f"offending node: {node}")
        else:
          print(f'segment: {node.meta["segment"]}')
          print(f'partition: {known_segments.index(node.meta["segment"])}')
        print("-")
      print("------")
    for name, sub_gm in gm_after_split.named_children():
      assert segment is not None
      # Replace subgraph with the lazy version
      print(f"is_inference: {is_inference}")
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
      lazy_sub_gm = LazyGraphModule(
        self,
        segment,
        sub_gm,
        compiled_fn,
      )
      setattr(gm_after_split, name, lazy_sub_gm)
      # Build segment -> GMs mapping
      self._segment_to_gms_map[segment].append(sub_gm)

    # breakpoint()
    return gm_after_split

  def maybe_run(self, gm, compiled_fn, segment, *args):
    # Create the handle and the async tensors
    args_fake = []
    for arg in args:
      if isinstance(arg, AsyncTensor):
        args_fake.append(arg._fake)
      elif isinstance(arg, torch.Tensor):
        args_fake.append(fake_mode.from_tensor(arg))
    with fake_mode:
      outs_fake = gm(*args_fake)
    # NOTE: important to make sure the same async tensor is used in downstream user code
    # as well as materialized when handle is finally run.
    outs_async = tuple(AsyncTensor(out_fake) for out_fake in outs_fake)
    if gm in self._gm_to_handle_map:
      cur_handle = self._gm_to_handle_map[gm]
    else:
      cur_handle = AsyncFuncHandle(compiled_fn, segment, args=args, outs_async=outs_async, scheduler=self)
      self._gm_to_handle_map[gm] = cur_handle
      self._handle_to_gm_map[cur_handle] = gm
    for out_async in outs_async:
      out_async.set_handle(cur_handle)

    # # # DEBUG ONLY
    # # return compiled_fn(list(args))

    # # DEBUG ONLY
    # cur_handle.schedule()
    # cur_handle.wait_for_completion()
    # # breakpoint()
    # # TODO: Problem: when returned value is AsyncTensor, the .grad_fn of it is not populated.
    # # But maybe we don't need this to implement SDD reordering?
    # # Basically, to implement SDD reordering, we just need to schedule some SDD op before some FWD/BWD op,
    # # following a schedule. The order of FWD/BWD ops is actually not changed.
    # return cur_handle.outs  # works

    # First, schedule all graphs from all segments that are before the incoming graph in the schedule.
    all_preceding_graph_handles = []
    reached_current_graph = False
    # TODO: for now, we always check the schedule from the beginning.
    # We can optimize this by keeping track of which segments have been scheduled already.
    _next_segment_index = 0
    while _next_segment_index < len(self._schedule):
      segment = self._schedule[_next_segment_index]
      for g in self._segment_to_gms_map[segment]:
        if str(g.graph) == str(gm.graph):  # TODO: is there a better way to check graph equivalence?
          reached_current_graph = True
          break
        all_preceding_graph_handles.append(
          self._gm_to_handle_map.get(g, None)
        )
      if reached_current_graph:
        break
      else:
        _next_segment_index += 1

    all_preceding_graph_handles_are_scheduled = True
    for handle in all_preceding_graph_handles:
      if handle is not None:
        print(f"will run preceding graph: {str(self._handle_to_gm_map[handle].code)}")
        handle.schedule()
      else:
        # Some preceding graph is not scheduled yet
        print(f"some preceding graphs are not scheduled yet, skipping current graph")
        all_preceding_graph_handles_are_scheduled = False
        break

    # Then, if all preceding graph handles are scheduled, then we schedule the incoming graph;
    # otherwise, we don’t schedule the incoming graph.
    #
    # NOTE: We can only do lazy scheduling for FWD-only op (e.g. SDD) for now.
    # Reason is that AsyncTensor output from FWD graph currently doesn't have .grad_fn,
    # and we need to fix it before we can implement general FWD/BWD op reordering.
    if is_fwd_only_op(cur_handle):
      if all_preceding_graph_handles_are_scheduled:
        print(f"will run current graph: {str(self._handle_to_gm_map[cur_handle].code)}")
        cur_handle.schedule()
      assert isinstance(outs_async, tuple)
      if len(outs_async) == 1:
        return outs_async[0]
      else:
        return outs_async
    else:
      assert all_preceding_graph_handles_are_scheduled
      print(f"will run current graph: {str(self._handle_to_gm_map[cur_handle].code)}")
      cur_handle.schedule()
      cur_handle.wait_for_completion()
      return cur_handle.outs

"""
TODO: we can potentially implement segment tagging and propagation via:
```
mod.register_forward_pre_hook(
  mod.func1 = RunWithSomeTorchDispatchMode(mod.func1, segment_name)
)

and then in RunWithSomeTorchDispatchMode, we call mod.func1 with a torch dispatch mode
that tags the FX node with segment_name via `fx_traceback.current_meta`

This way we don't need to do any change to Dynamo internals.

But, does this work for tagging BWD nodes?
```
"""


"""
TODO: graph with only in-place op doesn't have its output node, why?
"""


"""
FAQ

Q1: What happens if we have a user-defined segment deep down in a submodule?
Answer: everything before the defined segment will be in their own segment. Everything after is in another segment.
You can call this a "segment break".

Q2: What if there are multiple calls to the same module instance's same function?
Answer: we don't support it for now (we turn off the schedule in this case). In the future we could support it.
Note that we do support calling the same module class' (but different instances') same function.
"""
