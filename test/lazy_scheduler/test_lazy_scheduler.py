"""
pytest -vs test/lazy_scheduler/test_lazy_scheduler.py::TestLazyScheduler::test_backward_simple_no_segment
"""

import torch
import torch.utils._pytree as pytree
from torch.testing._internal.common_utils import TestCase as TorchTestCase
from torch._dynamo import disable
import functools
from torch._inductor.compile_fx import compile_fx
import itertools
from typing import Optional, Dict, Callable
from torch._subclasses.fake_tensor import FakeTensorMode
from collections import defaultdict, OrderedDict
import weakref
import threading
from torch._inductor.compile_fx import compile_fx_inner

fake_mode = FakeTensorMode()

class AsyncTensor(torch.Tensor):
  """
  This is a subclass of Tensor that represents a "lazy tensor".
  This tensor will be materialized by calling any tensor methods on it.
  """
  def __new__(cls, fake_tensor):
    shape = fake_tensor.shape
    kwargs = {}
    kwargs["strides"] = fake_tensor.stride()
    kwargs["storage_offset"] = fake_tensor.storage_offset()
    kwargs["device"] = fake_tensor.device
    kwargs["layout"] = fake_tensor.layout
    kwargs["requires_grad"] = fake_tensor.requires_grad
    kwargs["dtype"] = fake_tensor.dtype
    out = torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)
    return out

  def __init__(self, fake_tensor):
    super().__init__()
    self._materialized_tensor = None
    self._handle = None
    self._fake = fake_tensor

  def async_repr(self):
    return f"AsyncTensor({self._handle}, {self._fake})"

  def __repr__(self):
    # NOTE: `print(tensor)` goes through this
    if self._handle is not None:
      AsyncTensor.wait_until_materialized([self])
      return self._materialized_tensor.__repr__()
    else:
      return self.async_repr()

  def __format__(self, format_spec):
    # NOTE: `print(f"{tensor}")` goes through this
    AsyncTensor.wait_until_materialized([self])
    return self._materialized_tensor.__format__(format_spec)

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
    # TODO: implement randn_like etc. method that doesn't require a materialized tensor as input
    if func in [torch.ops.aten.ones_like.default]:
      shape = args[0].shape
      dtype = args[0].dtype
      device = args[0].device
      requires_grad = args[0].requires_grad
      return torch.ones(shape, dtype=dtype, device=device, requires_grad=requires_grad)
    else:
      AsyncTensor.wait_until_materialized(args)
      # TODO: handle tuple / list / etc in args
      # TODO: handle kwargs
      assert not kwargs
      args_materialized = pytree.tree_map_only(AsyncTensor, lambda x: x._materialized_tensor, args)
      return func(*args_materialized)

  @staticmethod
  def check_materialized(async_tensors):
    all_materialized = True
    for t in async_tensors:
      if isinstance(t, AsyncTensor) and t._materialized_tensor is None:
          all_materialized = False
          break
    return all_materialized

  @staticmethod
  def wait_until_materialized(async_tensors):
    for async_tensor in async_tensors:
      if not AsyncTensor.check_materialized([async_tensor]):
        # NOTE: recursively schedule the deps first
        AsyncTensor.wait_until_materialized([async_tensor.handle().args])
        async_tensor.handle().schedule()
        async_tensor.handle().wait_for_completion()


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
    args_materialized = pytree.tree_map_only(AsyncTensor, lambda x: x._materialized_tensor, self.args)
    self.outs = self.compiled_fn(list(args_materialized))
    self.cuda_event.record()

  def wait_for_completion(self):
    self.cuda_event.synchronize()
    for out, out_async in zip(self.outs, self.outs_async):
      # Set the output AsyncTensor's underlying materialized tensor
      # to be the actual output tensor.
      out_async.set_materialized_tensor(out)

  def is_completed(self):
    return self.cuda_event.query()

  def scheduler(self):
    scheduler = self._scheduler()
    assert scheduler is not None
    return scheduler


class LazyGraphModule(torch.nn.Module):
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

  # NOTE: this matches compile_fx_inner signature
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
    """
    Compiles a graph module using Inductor compile_fx_inner,
    and wraps the output compiled_fn in a LazyGraphModule to be called later.
    """
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

    outs_async = tuple(AsyncTensor(out_fake) for out_fake in outs_fake)
    if gm in self._gm_to_handle_map:
      cur_handle = self._gm_to_handle_map[gm]
    else:
      cur_handle = AsyncFuncHandle(compiled_fn, args=args, outs_async=outs_async, scheduler=self)
      self._gm_to_handle_map[gm] = cur_handle
      self._handle_to_gm_map[cur_handle] = gm
    for out_async in outs_async:
      out_async.set_handle(cur_handle)

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
    if all_preceding_graph_handles_are_scheduled:
      print(f"will run current graph: {str(self._handle_to_gm_map[cur_handle].code)}")
      cur_handle.schedule()
    assert isinstance(outs_async, tuple)
    if len(outs_async) == 1:
      return outs_async[0]
    else:
      return outs_async


class TestCase(TorchTestCase):
  def setUp(self):
    torch._dynamo.reset()
    super().setUp()

  def tearDown(self):
    super().tearDown()
    torch._dynamo.reset()


class TestLazyScheduler(TestCase):
  def test_backward_simple_no_segment(self):
    class TestModule(torch.nn.Module):
      def __init__(self):
        super().__init__()

      def func1(self, x, y):
        z = torch.matmul(x, y)
        return z

      def forward(self, x, y):
        z = self.func1(x, y)
        z = z * z
        return z

    device = "cuda" if torch.cuda.is_available() else "cpu"

    m = TestModule()
    m = m.to(device)
    x = torch.randn(4, 4, requires_grad=True, device=device)
    y = torch.randn(4, 4, requires_grad=True, device=device)

    actual_e = m(x, y)
    actual_e.sum().backward()
    print(f"eager: first iter done")
    actual_e = m(x, y)
    actual_e.sum().backward()
    print(f"eager: second iter done")

    lazy_scheduler = LazyScheduler()
    compiled_m_ls = torch.compile(
      m,
      backend=functools.partial(compile_fx, inner_compile=lazy_scheduler.compile),
      fullgraph=False
    )

    actual_ls = compiled_m_ls(x, y)
    print(f"actual_ls: {actual_ls}")
    actual_ls.sum().backward()
    print(f"compiled_ls: first iter done")
    actual_ls = compiled_m_ls(x, y)
    print(f"actual_ls: {actual_ls}")
    actual_ls.sum().backward()
    print(f"compiled_ls: second iter done")

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests
    run_tests()
