import torch._subclasses
from torch._subclasses.fake_tensor import DynamicOutputShapeException


def is_builtin(op):
    return op.namespace in ('aten', 'prims', 'prim')


def fake_check(op, args, kwargs, dynamic_only):
    with torch._subclasses.CrossRefFakeMode(ignore_op_fn=is_builtin):
        try:
            op(*args, **kwargs)
        except DynamicOutputShapeException:
            if not dynamic_only:
                raise
            return
        if dynamic_only:
            raise AssertionError(
                f"fake_check({op}, ..., dynamic_only={dynamic_only}): "
                f"dynamic_only means that the operator is expected to have "
                f"data-dependent output shape. We have not detected that this is "
                f"the case. Please check that your operator's FakeTensor "
                f"implementation is actually data dependent")
