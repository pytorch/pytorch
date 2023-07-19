from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.experimental.proxy_tensor import ShapeEnv
from torch.fx.experimental.symbolic_shapes import DimDynamic, free_symbols
import torch.utils._pytree as pytree
from typing import Union, Tuple
import torch

class DynamicTensorMode(FakeTensorMode):
    def __init__(self):
        super().__init__(allow_fallback_kernels=False, allow_non_fake_inputs=True, shape_env=ShapeEnv(), cls=DynamicTensor)

    def dispatch(self, func, types, args=(), kwargs=None):
        # Fakeify all of the non-dynamic args and run the fake func
        fake_args, fake_kwargs = pytree.tree_map_only(
            torch.Tensor, lambda t: self.fake_tensor_converter(self, t), (args, kwargs)
        )
        fake_ret = super().dispatch(func, types, fake_args, fake_kwargs)
        # Realify all of the args and run regular func
        real_args, real_kwargs = pytree.tree_map_only(
            DynamicTensor, lambda t: t._backing_tensor, (args, kwargs)
        )
        real_ret = func(*real_args, **real_kwargs)
        flat_fake_ret, fake_spec = pytree.tree_flatten(fake_ret)
        flat_real_ret, real_spec = pytree.tree_flatten(real_ret)
        assert fake_spec == real_spec, f"{fake_spec} != {real_spec}"
        assert len(flat_fake_ret) == len(flat_real_ret)
        for fake_r, real_r in zip(flat_fake_ret, flat_real_ret):
            fake_r._backing_tensor = real_r
        return pytree.tree_unflatten(flat_fake_ret, fake_spec)

class DynamicTensor(FakeTensor):
    @staticmethod
    def wrap(backing_tensor, dim: Union[Tuple[int,...], int, None]):
        if isinstance(dim, int):
            dim = (dim,)
        elif dim is None:
            dim = tuple(range(backing_tensor.dim()))
        elem = GLOBAL_MODE.fake_tensor_converter.meta_converter(
            backing_tensor,
            shape_env=GLOBAL_MODE.shape_env,
            dynamic_dims=[DimDynamic.UNBACKED if i in dim else DimDynamic.STATIC for i in range(backing_tensor.dim())]
        )
        self = DynamicTensor(GLOBAL_MODE, elem, backing_tensor.device)
        self._backing_tensor = backing_tensor
        return self

    @property
    def dynamic_dims(self):
        return tuple(i for i, s in enumerate(self.size()) if free_symbols(s))

    def __repr__(self):
        return f"DynamicTensor.wrap({repr(self._backing_tensor)}, dim={self.dynamic_dims})"

GLOBAL_MODE = DynamicTensorMode()
