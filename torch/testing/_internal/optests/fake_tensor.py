# mypy: ignore-errors

import functools

import torch
import torch._subclasses
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import FakeTensorMode, UnsupportedFakeTensorException


def is_builtin(op):
    return op.namespace in ('aten', 'prims', 'prim')


def fake_check(op, args, kwargs):
    with torch._subclasses.CrossRefFakeMode(ignore_op_fn=is_builtin):
        op(*args, **kwargs)


def symbolic_fake_check(op, args, kwargs):
    if (
        is_builtin(op)
        or not torch._C._dispatch_has_kernel_for_dispatch_key(op.name(), "Meta")
        or torch.Tag.dynamic_output_shape in op.tags
        or torch.Tag.inplace_view in op.tags
        or torch.Tag.data_dependent_output in op.tags
    ):
        return

    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    shape_env = ShapeEnv()
    with FakeTensorMode(shape_env=shape_env) as fake_mode:
        try:
            fake_args, fake_kwargs = pytree.tree_map_only(
                torch.Tensor,
                functools.partial(fake_mode.from_tensor, static_shapes=False),
                (args, kwargs),
            )
        except UnsupportedFakeTensorException:
            return
        try:
            with shape_env.error_on_new_guards():
                op(*fake_args, **fake_kwargs)
        except UnsupportedFakeTensorException as e:
            raise RuntimeError(str(e)) from e
