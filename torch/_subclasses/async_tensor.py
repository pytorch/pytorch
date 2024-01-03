import torch
import weakref
import torch.utils._pytree as pytree

from torch._subclasses.fake_tensor import FakeTensorMode

fake_mode = FakeTensorMode()

class AsyncTensor(torch.Tensor):
  """
  This is a subclass of Tensor that represents a "lazy tensor".
  This tensor will be materialized by calling any tensor methods on it.
  """
  def __new__(cls, fake_tensor, *args, **kwargs):
    shape = fake_tensor.shape
    tensor_ctor_kwargs = {}
    tensor_ctor_kwargs["strides"] = fake_tensor.stride()
    tensor_ctor_kwargs["storage_offset"] = fake_tensor.storage_offset()
    tensor_ctor_kwargs["device"] = fake_tensor.device
    tensor_ctor_kwargs["layout"] = fake_tensor.layout
    tensor_ctor_kwargs["requires_grad"] = fake_tensor.requires_grad
    tensor_ctor_kwargs["dtype"] = fake_tensor.dtype
    out = torch.Tensor._make_wrapper_subclass(cls, shape, **tensor_ctor_kwargs)
    return out

  def __init__(self, fake_tensor, materialized_tensor=None):
    super().__init__()
    self._materialized_tensor = materialized_tensor
    self._fake = fake_tensor
    self._handle = None

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

  # NOTE: Any PyTorch reads or mutations in eager region will go through __torch_dispatch__,
  # so we materialize the underlying tensor here and returns it.
  @classmethod
  def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
    # TODO: implement randn_like etc. method that doesn't require a materialized tensor as input
    # TODO: implement other new_X etc. similar to new_empty_strided
    # print(f"func: {func}")
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
      # TODO: handle kwargs
      assert kwargs is None or len(kwargs) == 0
      AsyncTensor.wait_until_materialized(args)
      args_materialized = pytree.tree_map_only(AsyncTensor, lambda x: x._materialized_tensor, args)
      # kwargs_materialized = {k: pytree.tree_map_only(AsyncTensor, lambda x: x._materialized_tensor, v) for k, v in kwargs.items()}
      # out = func(*args_materialized, **kwargs_materialized)
      out = func(*args_materialized)
      # NOTE: if we don't re-wrap the output with AsyncTensor, sometimes the output will still be re-wrapped as AsyncTensor
      # (by another unknown mechanism outside of this code) but lose all its AsyncTensor attributes like `_materialized_tensor`
      if isinstance(out, torch.Tensor) and not isinstance(out, AsyncTensor):
        out = AsyncTensor(fake_tensor=fake_mode.from_tensor(out), materialized_tensor=out)
      return out
      # return return_and_correct_aliasing(func, args, kwargs, out)

  def materialize_with_value(self, tensor):
    self._materialized_tensor = tensor

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
