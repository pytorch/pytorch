import torch
import itertools
from typing import Optional, Dict, Callable
from torch._subclasses.fake_tensor import FakeTensorMode
from collections import defaultdict
import weakref

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
    print(f"NoneWithUsageDetector attr: {attr}")
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

  def data_ptr(self):
    if self._handle is not None:
      AsyncTensor.wait_until_materialized([self])
      return self._materialized_tensor.data_ptr()
    else:
      raise Exception("Calling data_ptr on an unmaterialized AsyncTensor!")

  def storage(self):
    if self._handle is not None:
      AsyncTensor.wait_until_materialized([self])
      return self._materialized_tensor.storage()
    else:
      raise Exception("Calling storage on an unmaterialized AsyncTensor!")

  # TODO: implement `.data = X`
  @property
  def data(self):
    if self._handle is not None:
      AsyncTensor.wait_until_materialized([self])
      return self._materialized_tensor.data
    else:
      raise Exception("Calling data on an unmaterialized AsyncTensor!")

  def __repr__(self):
    # NOTE: `print(tensor)` goes through this
    if self._handle is not None:
      AsyncTensor.wait_until_materialized([self])
      return self._materialized_tensor.__repr__()
    else:
      return self.async_repr()

  def async_repr(self):
    return f"AsyncTensor({self._handle}, {self._fake})"

  def __getattribute__(self, attr):
    print(f"attr: {attr}")
    # if attr == "_materialized_tensor":
    #   breakpoint()
    return super().__getattribute__(attr)

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
      if isinstance(async_tensor, AsyncTensor):
        print(f"looking at: {async_tensor.async_repr()}")
      if not AsyncTensor.check_materialized([async_tensor]):
        # NOTE: recursively schedule the deps first
        print(f"waiting on deps for {async_tensor.async_repr()}")
        AsyncTensor.wait_until_materialized([async_tensor.handle().args])
        print(f"going to schedule {async_tensor.async_repr()}")
        async_tensor.handle().schedule()
        print(f"waiting for completion {async_tensor.async_repr()}")
        async_tensor.handle().wait_for_completion()
        print(f"done waiting for completion {async_tensor.async_repr()}")

class AsyncFuncHandle:
  def __init__(self, compiled_fn, gm, segment, args, outs_async, scheduler):
    self.cuda_event = torch.cuda.Event()
    self.compiled_fn: Callable = compiled_fn
    self.gm: torch.fx.GraphModule = gm
    self.segment: str = segment  # for bookkeeping
    # dependency graph is built implicitly as we run the program
    self.args = args
    self.outs_async = outs_async
    self.outs = None
    self.is_scheduled = False
    self._scheduler = weakref.ref(scheduler)

  def schedule(self):
    print(f"scheduling {self.gm} ... from segment: {self.segment}")
    # make sure to schedule only once
    if self.is_scheduled:
      return
    AsyncTensor.wait_until_materialized(self.args)
    self.outs = self.compiled_fn(*self.args)
    self.cuda_event.record()
    self.is_scheduled = True

  def wait_for_completion(self):
    print("wait_for_completion is called")
    self.cuda_event.synchronize()
    for out, out_async in zip(self.outs, self.outs_async):
      print("set materialized tensor")
      out_async.set_materialized_tensor(out)

  def is_completed(self):
    return self.cuda_event.query()

  def scheduler(self):
    scheduler = self._scheduler()
    assert scheduler is not None
    return scheduler


class LazyGraphModule(torch.nn.Module):  # TODO: better name?
  def __init__(self, gm, scheduler, segment):
    super().__init__()
    self.gm = gm
    self.scheduler = scheduler
    self.compiled_fn = None
    self.segment = segment  # bookkeeping

  def __call__(self, *args):
    if self.compiled_fn is None:
      # fake_mode = FakeTensorMode()
      # breakpoint()
      # for x in args:
      #   if isinstance(x, list):
      #     print(f"x: {x}")
      # args_fake = [fake_mode.from_tensor(x) for x in args]
      self.compiled_fn = torch._inductor.compile(self.gm, args)
      self.scheduler.record_compiled_fn(self.gm, self.compiled_fn, self.segment)
      # During compile time, return real tensor, because downstream Inductor compilation needs real tensor
      return self.compiled_fn(*args)
    else:
      # breakpoint()
      print(f"lgm call args will be called")
      print(f"lgm call args: {args}")
      return self.scheduler.maybe_run(self.gm, self.compiled_fn, *args)

    # return self.gm(*args)


class LazyScheduler:
  def __init__(self, schedule):
    self._schedule = schedule
    self._gm_to_handle_map = {}
    self._segment_to_gms_map = defaultdict(list)
    self._next_segment_index = 0

  def compile(self, gm, example_inputs):
    # breakpoint()
    known_segments = []
    print(f"gm.graph: {gm.graph}")
    for node in gm.graph.nodes:
      # print(f"node: {node}")
      # print(f"node.meta['segment']: {node.meta['segment']}")
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
    gm_node_list = list(gm.graph.nodes)
    gm_after_split_node_list = list(gm_after_split.graph.nodes)
    gm_after_split_children_list = list(gm_after_split.children())
    # breakpoint()
    for name, sub_gm in gm_after_split.named_children():
      print(f"subgraph: {str(sub_gm.graph)}")
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
      # Build segment -> GMs mapping
      node_list = list(sub_gm.graph.nodes)
      assert node_list[-1].op == "output"
      # the 2nd last node is guaranteed to not be a placeholder (i.e. input) node,
      # so its segment tag will actually match what we expect.
      assert node_list[-2].op != "placeholder"
      segment = node_list[-2].meta["segment"]
      print(f"segment: {segment}")
      # Replace subgraph with the lazy version
      lazy_sub_gm = LazyGraphModule(sub_gm, self, segment)
      setattr(gm_after_split, name, lazy_sub_gm)
      self._segment_to_gms_map[segment].append(sub_gm)

    print(f"gm.graph: {gm.graph}")
    print(f"gm_after_split: {gm_after_split.graph}")
    for name, sub_gm in gm_after_split.named_children():
      print(f"subgraph {name}: {str(sub_gm.gm.graph)}")

    print(f"gm.graph: {gm.code}")
    print(f"gm_after_split: {gm_after_split.code}")
    for name, sub_gm in gm_after_split.named_children():
      print(f"subgraph {name}: {str(sub_gm.gm.code)}")

    # breakpoint()
    return gm_after_split

  def record_compiled_fn(self, gm, compiled_fn, segment):
    cur_handle = AsyncFuncHandle(compiled_fn, gm, segment, args=[], outs_async=[], scheduler=self)
    self._gm_to_handle_map[gm] = cur_handle

  def maybe_run(self, gm, compiled_fn, *args):
    # For now, we just eagerly run it
    # TODO: implement running based on schedule and returning AsyncTensor
    # return compiled_fn(*args)

    # Create the handle and the async tensors
    args_fake = []
    for arg in args:
      if isinstance(arg, AsyncTensor):
        args_fake.append(arg._fake)
      elif isinstance(arg, torch.Tensor):
        args_fake.append(fake_mode.from_tensor(arg))
    print(f"gm: {str(gm.graph)}")
    print(f"args: {args}")
    print(f"args_fake: {args_fake}")
    with fake_mode:
      outs_fake = gm(*args_fake)
    print(f"outs_fake: {outs_fake}")
    # NOTE: important to make sure the same async tensor is used in downstream user code
    # as well as materialized when handle is finally run.
    outs_async = tuple(AsyncTensor(out_fake) for out_fake in outs_fake)
    # breakpoint()
    cur_handle = self._gm_to_handle_map[gm]
    cur_handle.args = args
    cur_handle.outs_async = outs_async
    for out_async in outs_async:
      out_async.set_handle(cur_handle)

    # First, schedule all graphs from all segments that are before the incoming graph in the schedule.
    all_preceding_graph_handles = []
    reached_current_segment = False
    while self._next_segment_index < len(self._schedule):
      segment = self._schedule[self._next_segment_index]
      for g in self._segment_to_gms_map[segment]:
        if g == gm:
          reached_current_segment = True
          break
        all_preceding_graph_handles.append(
          self._gm_to_handle_map.get(g, None)
        )
      self._next_segment_index += 1
      if reached_current_segment:
        break

    all_preceding_graph_handles_are_scheduled = True
    for handle in all_preceding_graph_handles:
      if handle is not None:
        print(f"will run: {str(handle.gm.code)}")
        handle.schedule()
      else:
        # Some preceding graph is not recorded yet
        all_preceding_graph_handles_are_scheduled = False
        break

    # Then, if all preceding graph handles are scheduled, then we schedule the incoming graph; otherwise, we don’t schedule the incoming graph.
    if all_preceding_graph_handles_are_scheduled:
      cur_handle.schedule()

    # TODO: at end of loop slice, we need to schedule all remaining unscheduled handles
    assert isinstance(outs_async, tuple)
    if len(outs_async) == 1:
      return outs_async[0]
    else:
      return outs_async


"""
TODO: graph with only in-place op doesn't have its output node, why?

(Pdb) gm_after_split_children_list
[GraphModule(), GraphModule(), GraphModule(), GraphModule()]
(Pdb) gm_after_split_children_list[0]
GraphModule()
(Pdb) gm_after_split_children_list[0].graph
<torch.fx.graph.Graph object at 0x7f14a79ddd90>
(Pdb) str(gm_after_split_children_list[0].graph)
graph():
    %x : torch.Tensor [num_users=1] = placeholder[target=x]
    %l__self___buf : [num_users=1] = placeholder[target=l__self___buf]
    %l_y_ : torch.Tensor [num_users=1] = placeholder[target=l_y_]
    %relu_ : [num_users=0] = call_method[target=relu_](args = (%x,), kwargs = {})
    %relu__1 : [num_users=0] = call_method[target=relu_](args = (%l__self___buf,), kwargs = {})
    %chunk : [num_users=2] = call_function[target=torch.chunk](args = (%l_y_, 2), kwargs = {})
    %getitem : [num_users=1] = call_function[target=operator.getitem](args = (%chunk, 0), kwargs = {})
    %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%chunk, 1), kwargs = {})
    %cat : [num_users=1] = call_function[target=torch.cat](args = ((%getitem, %getitem_1),), kwargs = {})
    return cat
(Pdb) str(gm_after_split_children_list[1].graph)
graph():
    %y : [num_users=1] = placeholder[target=y]
    %relu_ : [num_users=0] = call_method[target=relu_](args = (%y,), kwargs = {})
(Pdb) str(gm_after_split_children_list[2].graph)
graph():
    %x : torch.Tensor [num_users=1] = placeholder[target=x]
    %y : [num_users=1] = placeholder[target=y]
    %add : [num_users=1] = call_function[target=operator.add](args = (%x, %y), kwargs = {})
    return add'
(Pdb) str(gm_after_split_children_list[3].graph)
graph():
    %z : [num_users=1] = placeholder[target=z]
    %relu_ : [num_users=0] = call_method[target=relu_](args = (%z,), kwargs = {})
(Pdb)

(Pdb) str(gm_after_split.code)
def forward(self, L_x_ : torch.Tensor, L_y_ : torch.Tensor):
    l_x_ = L_x_
    l__self___buf = self.L__self___buf
    l_y_ = L_y_
    submod_1 = self.submod_1(l_x_, l__self___buf, l_y_);  l__self___buf = l_y_ = None
    submod_2 = self.submod_2(submod_1)
    submod_3 = self.submod_3(l_x_, submod_1);  l_x_ = None
    submod_4 = self.submod_4(submod_3)
    return (submod_1, submod_3)
"""


"""
FAQ

Q1: What happens if we have a user-defined segment deep down in a submodule?
Answer: everything before the defined segment will be in their own segment. Everything after is in another segment.
You can call this a "segment break".
"""
