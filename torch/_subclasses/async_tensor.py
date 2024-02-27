from typing import Dict

import torch
import weakref
import torch.utils._pytree as pytree
from torch.overrides import get_default_nowrap_functions
from torch._tensor import _convert
from torch._functorch._aot_autograd.functional_utils import is_fun

import traceback

# TODO: what's the point of using fake_tensor in AsyncTensor class?
fake_mode = None
def get_fake_mode():
  global fake_mode
  if fake_mode is None:
    fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()
  return fake_mode


class TensorContainer:
  _tensor: torch.Tensor

  def __init__(self, tensor=None):
    self._tensor = tensor

  def set_tensor(self, tensor):
    self._tensor = tensor

  def get_tensor(self):
    return self._tensor


class AsyncTensor(torch.Tensor):
  # _unused_real_tensor: torch.Tensor
  _fake_tensor: "FakeTensor"
  _handle: "AsyncFuncHandle"
  _materialized_tensor_container: TensorContainer
  __slots__ = ["_fake_tensor", "_handle", "_materialized_tensor_container"]

  """
  This is a subclass of Tensor that represents a "lazy tensor".
  This tensor will be materialized by calling any tensor methods on it.
  """
  def __new__(cls, fake_tensor, handle, materialized_tensor_container):
    shape = fake_tensor.shape
    tensor_ctor_kwargs = {}
    tensor_ctor_kwargs["strides"] = fake_tensor.stride()
    tensor_ctor_kwargs["storage_offset"] = fake_tensor.storage_offset()
    tensor_ctor_kwargs["device"] = fake_tensor.device
    tensor_ctor_kwargs["layout"] = fake_tensor.layout
    tensor_ctor_kwargs["requires_grad"] = fake_tensor.requires_grad
    tensor_ctor_kwargs["dtype"] = fake_tensor.dtype
    out = torch.Tensor._make_wrapper_subclass(cls, shape, **tensor_ctor_kwargs)
    out._fake_tensor = fake_tensor
    out._handle = handle
    out._materialized_tensor_container = materialized_tensor_container
    return out

  # TODO: follow DTensor impl in https://github.com/pytorch/pytorch/blob/main/torch/distributed/_tensor/api.py#L207
  # and Float8Tensor impl in https://github.com/pytorch-labs/float8_experimental/blob/289c1225b93a60d9e40bb0750a28f1e7a2885207/float8_experimental/float8_tensor.py#L60

  @property
  def materialized_tensor(self):
    return self.get_materialized_tensor()

  # NOTE: we explicitly don't want this tensor subclass to propagate through the Dynamo tracing system.
  # Instead, we want it to be treated as real tensor starting from beginning of Dynamo so that model is traced normally.
  # def __tensor_flatten__(self):
  #   """
  #   protocol to inform how to flatten a AsyncTensor to local tensor
  #   for PT2 tracing
  #   """
  #   ctx = {
  #     "_fake_tensor": self._fake_tensor,
  #     "_handle": self._handle,
  #     "_materialized_tensor_container": self._materialized_tensor_container,
  #   }
  #   return ["materialized_tensor"], ctx

  # @staticmethod
  # def __tensor_unflatten__(inner_tensors: Dict, metadata, outer_size, outer_stride):
  #   assert len(inner_tensors) == 1
  #   return AsyncTensor(
  #     # inner_tensors["_unused_real_tensor"],
  #     metadata["_fake_tensor"],
  #     metadata["_handle"],
  #     metadata["_materialized_tensor_container"],
  #   )

  __torch_function__ = torch._C._disabled_torch_function_impl

  def async_repr(self):
    return f"AsyncTensor({self._handle}, {self._fake_tensor})"

  def __repr__(self):
    # NOTE: `print(tensor)` goes through this
    if self._handle is not None:
      AsyncTensor.wait_until_materialized([self])
      return self._materialized_tensor_container.get_tensor().__repr__()
    else:
      return self.async_repr()

  def __format__(self, format_spec):
    # NOTE: `print(f"{tensor}")` goes through this
    if self._handle is not None:
      AsyncTensor.wait_until_materialized([self])
      return self._materialized_tensor_container.get_tensor().__format__(format_spec)
    else:
      return self.async_repr()

  def handle(self):
    assert self._handle is not None
    handle = self._handle()
    assert handle is not None
    return handle

  def set_handle(self, handle):
    self._handle = weakref.ref(handle)

  # NOTE: Any PyTorch reads or mutations in eager region will go through __torch_dispatch__,
  # so we materialize the underlying tensor here and returns it.
  @classmethod
  def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
    # TODO: implement randn_like etc. method that doesn't require a materialized tensor as input
    # TODO: implement other new_X etc. similar to new_empty_strided
    if kwargs is None:
      kwargs = {}
    if func in [torch.ops.aten.ones_like.default]:
      shape = args[0].shape
      dtype = args[0].dtype
      device = args[0].device
      requires_grad = args[0].requires_grad
      return torch.ones(shape, dtype=dtype, device=device, requires_grad=requires_grad)
    elif func in [torch.ops.aten.new_empty_strided.default]:
      args_except_self_tensor = list(args)[1:]
      return torch.empty_strided(*args_except_self_tensor, **kwargs)
    else:
      # TODO: handle tuple / list / etc in args
      # TODO: handle tensor kwargs
      AsyncTensor.wait_until_materialized(args)
      args_from_functional = pytree.tree_map_only(torch._subclasses.functional_tensor.FunctionalTensor, lambda x: x.from_functional(), args)
      args_materialized = pytree.tree_map_only(AsyncTensor, lambda x: x._materialized_tensor_container.get_tensor(), args_from_functional)
      # kwargs_materialized = {k: pytree.tree_map_only(AsyncTensor, lambda x: x._materialized_tensor_container.get_tensor(), v) for k, v in kwargs.items()}
      # out = func(*args_materialized, **kwargs_materialized)
      assert not any(isinstance(x, AsyncTensor) for x in args_materialized)
      out = func(*args_materialized, **kwargs)
      # NOTE: if we don't re-wrap the output with AsyncTensor, sometimes the output will still be re-wrapped as AsyncTensor
      # (by another unknown mechanism outside of this code, maybe in `def __torch_function__(...)` in torch/_tensor.py?)
      # but it will lose all its AsyncTensor attributes like `_materialized_tensor_container`
      if isinstance(out, torch.Tensor) and not isinstance(out, AsyncTensor):
        out = AsyncTensor(fake_tensor=get_fake_mode().from_tensor(out), handle=None, materialized_tensor_container=TensorContainer(out))
      return out

  def materialize_with_value(self, materialized_tensor):
    assert (not isinstance(materialized_tensor, AsyncTensor)) and isinstance(materialized_tensor, torch.Tensor), f"Received type: {type(materialized_tensor)}"
    self._materialized_tensor_container.set_tensor(materialized_tensor)
    # if self._unused_real_tensor is None:
    #   self._unused_real_tensor = materialized_tensor

  def get_materialized_tensor(self):
    AsyncTensor.wait_until_materialized([self])
    tensor = self._materialized_tensor_container.get_tensor()
    assert tensor is not None
    return tensor

  @staticmethod
  def check_materialized(async_tensors):
    all_materialized = True
    for t in async_tensors:
      if isinstance(t, AsyncTensor) and t._materialized_tensor_container.get_tensor() is None:
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
