from typing import Callable, Union

import torch
import torch.utils._pytree as pytree
from torch._ops import OpOverload
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

aten = torch.ops.aten


class CrossRefFakeMode(TorchDispatchMode):
    def __init__(
        self,
        ignore_op_fn: Union[Callable[[OpOverload], bool], None] = None,
        *,
        check_strides=True,
    ):
        self.ignore_op_fn = (
            ignore_op_fn if ignore_op_fn is not None else lambda fn: False
        )
        self.check_strides = check_strides

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        from torch._subclasses.fake_tensor import (
            FakeTensorMode,
            UnsupportedFakeTensorException,
        )

        fake_r = None

        # empty_like excluded for now due to sparse complex
        # aten._to_dense.default this one is getting called with csc
        if (
            func
            not in (
                aten.lift_fresh.default,
                aten.lift_fresh_copy.default,
                aten.set_.source_Storage_storage_offset,
            )
            and not self.ignore_op_fn(func)
            and torch.Tag.dynamic_output_shape not in func.tags  # type: ignore[attr-defined]
            and torch.Tag.inplace_view not in func.tags  # type: ignore[attr-defined]
            and torch.Tag.data_dependent_output not in func.tags  # type: ignore[attr-defined]
        ):
            try:
                with FakeTensorMode() as fake_mode:
                    fake_args, fake_kwargs = pytree.tree_map_only(
                        torch.Tensor, fake_mode.from_tensor, (args, kwargs)
                    )
                    fake_r = func(*fake_args, **fake_kwargs)
            except UnsupportedFakeTensorException:
                pass

        r = func(*args, **kwargs)
        if fake_r is not None:
            for r_out, fake_out in zip(tree_flatten(r)[0], tree_flatten(fake_r)[0]):
                r_ten = isinstance(r_out, torch.Tensor)
                assert r_ten == isinstance(fake_out, torch.Tensor)
                if r_ten:
                    try:
                        torch._prims.utils.compare_tensor_meta(
                            r_out, fake_out, check_strides=self.check_strides
                        )
                    except Exception as e:
                        raise RuntimeError(f"Mismatch on {func}: {e}")
        return r
