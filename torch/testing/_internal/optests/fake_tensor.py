# mypy: ignore-errors

import torch
import torch._subclasses
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import FakeTensorMode, UnsupportedFakeTensorException
from torch.utils._python_dispatch import is_traceable_wrapper_subclass


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

    from torch._dynamo.source import LocalSource
    from torch.fx.experimental.symbolic_shapes import (
        DimDynamic,
        ShapeEnv,
        StatelessSymbolicContext,
    )

    flat_args, spec = pytree.tree_flatten((args, kwargs))
    with FakeTensorMode(shape_env=ShapeEnv()) as fake_mode:
        try:
            for i, arg in enumerate(flat_args):
                if isinstance(arg, torch.Tensor):
                    if is_traceable_wrapper_subclass(arg):
                        return
                    ctx = StatelessSymbolicContext(
                        dynamic_sizes=[DimDynamic.UNBACKED] * arg.dim()
                    )
                    flat_args[i] = fake_mode.from_tensor(
                        arg,
                        source=LocalSource(f"arg{i}"),
                        symbolic_context=ctx,
                    )
            fake_args, fake_kwargs = pytree.tree_unflatten(flat_args, spec)
            op(*fake_args, **fake_kwargs)
        except UnsupportedFakeTensorException:
            return
