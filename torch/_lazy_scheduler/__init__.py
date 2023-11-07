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
    r._materialized_tensor = None
    r._handle = None
    r._fake = fake_tensor
    return r

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

  def __getattribute__(self, attr):
    if attr in dir(torch.Tensor):
      if self._handle is not None:
        AsyncTensor.wait_until_materialized([self])
        return getattr(self._materialized_tensor, attr)
      else:
        raise Exception(f"Getting {attr} on an AsyncTensor that doesn't have handle is not allowed")
    else:
      return super().__getattribute__(attr)

  def __setattr__(self, attr, value):
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
      if isinstance(t, AsyncTensor) and t._materialized_tensor is None:
          all_materialized = False
          break
    return all_materialized

  @staticmethod
  def wait_until_materialized(async_tensors):
    # breakpoint()
    for async_tensor in async_tensors:
      if not AsyncTensor.check_materialized([async_tensor]):
        # NOTE: recursively schedule the deps first
        AsyncTensor.wait_until_materialized([async_tensor.handle().args])
        async_tensor.handle().schedule()
        async_tensor.handle().wait_for_completion()


class AsyncFuncHandle:
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
    self.outs = self.compiled_fn(list(self.args))
    self.cuda_event.record()

  def wait_for_completion(self):
    self.cuda_event.synchronize()
    for out, out_async in zip(self.outs, self.outs_async):
      out_async.set_materialized_tensor(out)

  def is_completed(self):
    return self.cuda_event.query()

  def scheduler(self):
    scheduler = self._scheduler()
    assert scheduler is not None
    return scheduler


class LazyGraphModule(torch.nn.Module):
  def __init__(self, scheduler, gm, compiled_fn):
    super().__init__()
    self.scheduler = scheduler
    self.gm = gm
    self.compiled_fn = compiled_fn

  def __call__(self, *args):
    assert self.compiled_fn is not None
    return self.scheduler.maybe_run(self.gm, self.compiled_fn, *args)


class LazyScheduler:
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

    cur_handle.schedule()
    cur_handle.wait_for_completion()
    # return cur_handle.outs
    # The expectation is that when user tries to use the output,
    # it should either be materialized already, or its materialization will be scheduled immediately.
    # Problem: when returned value is AsyncTensor, the .grad_fn of it is not populated.
    return cur_handle.outs_async  # fails
