# mypy: ignore-errors

import torch._subclasses


def is_builtin(op):
    return op.namespace in ('aten', 'prims', 'prim')


def fake_check(op, args, kwargs, *, check_symbolic_guards=False):
    with torch._subclasses.CrossRefFakeMode(
        ignore_op_fn=is_builtin,
        check_symbolic_guards=check_symbolic_guards,
    ):
        op(*args, **kwargs)
