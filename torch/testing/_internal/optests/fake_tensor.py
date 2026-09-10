# mypy: ignore-errors

import torch._subclasses


def is_builtin(op):
    return op.namespace in ('aten', 'prims', 'prim')


def fake_check(op, args, kwargs):
    with torch._subclasses.CrossRefFakeMode(ignore_op_fn=is_builtin):
        op(*args, **kwargs)
