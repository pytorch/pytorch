from typing import Dict

import torch
import weakref
import torch.utils._pytree as pytree
from torch.overrides import get_default_nowrap_functions
from torch._tensor import _convert
from torch._functorch._aot_autograd.functional_utils import is_fun

from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
import traceback

# TODO: how to use Dynamo's fake mode instead of creating our own? is there a global singleton we can access?
# see _dynamo/variables/builder.py line 1406
# Broader question: what's the point of using fake_tensor in AsyncTensor class?
fake_mode = FakeTensorMode()


class TensorContainer:
  _tensor: torch.Tensor

  def __init__(self, tensor=None):
    self._tensor = tensor

  def set_tensor(self, tensor):
    self._tensor = tensor

  def get_tensor(self):
    return self._tensor


class AsyncTensor(torch.Tensor):
  _unused_real_tensor: torch.Tensor
  _fake_tensor: FakeTensor
  _handle: "AsyncFuncHandle"
  _materialized_tensor_container: TensorContainer
  __slots__ = ["_fake_tensor", "_handle", "_materialized_tensor_container"]

  """
  This is a subclass of Tensor that represents a "lazy tensor".
  This tensor will be materialized by calling any tensor methods on it.
  """
  def __new__(cls, unused_real_tensor, fake_tensor, handle, materialized_tensor_container):
    shape = fake_tensor.shape
    tensor_ctor_kwargs = {}
    tensor_ctor_kwargs["strides"] = fake_tensor.stride()
    tensor_ctor_kwargs["storage_offset"] = fake_tensor.storage_offset()
    tensor_ctor_kwargs["device"] = fake_tensor.device
    tensor_ctor_kwargs["layout"] = fake_tensor.layout
    tensor_ctor_kwargs["requires_grad"] = fake_tensor.requires_grad
    tensor_ctor_kwargs["dtype"] = fake_tensor.dtype
    out = torch.Tensor._make_wrapper_subclass(cls, shape, **tensor_ctor_kwargs)
    if unused_real_tensor is None:
      # TODO figure out this:
      # this maybe uses extra memory.
      # without this, it likely throws: https://gist.github.com/yf225/e6330c85d1b92fa68a5013acd9427466
      # I believe meta tensor creation needs a concrete inner tensor to track shared storage.
      out._unused_real_tensor = torch.empty(shape, dtype=fake_tensor.dtype, layout=fake_tensor.layout, device=fake_tensor.device, requires_grad=fake_tensor.requires_grad)
    else:
      out._unused_real_tensor = unused_real_tensor
    out._fake_tensor = fake_tensor
    out._handle = handle
    out._materialized_tensor_container = materialized_tensor_container
    traceback.print_stack()
    print(f"here7: id(out): {id(out)}")
    return out

  # TODO: follow DTensor impl in https://github.com/pytorch/pytorch/blob/main/torch/distributed/_tensor/api.py#L207
  # and Float8Tensor impl in https://github.com/pytorch-labs/float8_experimental/blob/289c1225b93a60d9e40bb0750a28f1e7a2885207/float8_experimental/float8_tensor.py#L60

  def __tensor_flatten__(self):
    """
    protocol to inform how to flatten a DTensor to local tensor
    for PT2 tracing
    """
    ctx = {
      "_fake_tensor": self._fake_tensor,
      "_handle": self._handle,
      "_materialized_tensor_container": self._materialized_tensor_container,
    }
    return ["_unused_real_tensor"], ctx

  @staticmethod
  def __tensor_unflatten__(inner_tensors: Dict, metadata, outer_size, outer_stride):
    assert len(inner_tensors) == 1
    return AsyncTensor(
      inner_tensors["_unused_real_tensor"],
      metadata["_fake_tensor"],
      metadata["_handle"],
      metadata["_materialized_tensor_container"],
    )

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
    # print(f"func: {func}")
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
      args_materialized = pytree.tree_map_only(AsyncTensor, lambda x: x._materialized_tensor_container.get_tensor(), args)
      # kwargs_materialized = {k: pytree.tree_map_only(AsyncTensor, lambda x: x._materialized_tensor_container.get_tensor(), v) for k, v in kwargs.items()}
      # out = func(*args_materialized, **kwargs_materialized)
      assert not any(isinstance(x, AsyncTensor) for x in args_materialized)
      out = func(*args_materialized, **kwargs)
      # NOTE: if we don't re-wrap the output with AsyncTensor, sometimes the output will still be re-wrapped as AsyncTensor
      # (by another unknown mechanism outside of this code, maybe in `def __torch_function__(...)` in torch/_tensor.py?)
      # but lose all its AsyncTensor attributes like `_materialized_tensor_container`
      if isinstance(out, torch.Tensor) and not isinstance(out, AsyncTensor):  # and not func in [torch.ops.aten.clone.default]:
        print(f"here5: type(out): {type(out)}")
        print(f"torch._utils.is_compiling(): {torch._utils.is_compiling()}")
        # TODO maybe look at input args to know if we are compiling? see how we do it for SAC dispatch mode. We should not re-wrap with AsyncTensor if we are compiling with AsyncTensor input
        """
        def _is_compiling(func, args, kwargs):
          # Check if we are under AOTAutograd tracing
          # There should probably be a better way to do this...
          # TODO: unify _is_compiling across all compile stacks
          for arg in args:
              if isinstance(arg, torch.Tensor) and is_fun(arg):
                  return True
          return False
        """
        # breakpoint()
        out = AsyncTensor(unused_real_tensor=out, fake_tensor=fake_mode.from_tensor(out), handle=None, materialized_tensor_container=TensorContainer(out))
      print(f"in dispatch: out._is_view(): {out._is_view()}")
      return out
      # return return_and_correct_aliasing(func, args, kwargs, out)

  @classmethod
  def __torch_function__(cls, func, types, args=(), kwargs=None):
    if kwargs is None:
      kwargs = {}

    with torch._C.DisableTorchFunctionSubclass():
      ret = func(*args, **kwargs)
      if func in get_default_nowrap_functions():
        return ret
      else:
        return _convert(ret, cls)

  def materialize_with_value(self, materialized_tensor):
    assert (not isinstance(materialized_tensor, AsyncTensor)) and isinstance(materialized_tensor, torch.Tensor), f"Received type: {type(materialized_tensor)}"
    self._materialized_tensor_container.set_tensor(materialized_tensor)
    if self._unused_real_tensor is None:
      self._unused_real_tensor = materialized_tensor

  def get_materialized_tensor(self):
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
