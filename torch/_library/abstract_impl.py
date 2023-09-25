import contextlib
from typing import Callable

import torch


def get_none():
    return None


global_ctx_getter: Callable = get_none


# NOTE [ctx inside the fake implementation]
# If a user has an operator with data-dependent output shape, then when writing
# a fake implementation they must query the current ctx and use methods on the
# ctx to construct a new unbacked symint.
#
# This is done via us setting the global_ctx_getter function every time a fake
# implementation is invoked.
def get_ctx() -> "AbstractImplCtx":
    """get_ctx() returns the current AbstractImplCtx object.

    Calling ``get_ctx()`` is only valid inside of an abstract implementation.
    """
    return global_ctx_getter()


@contextlib.contextmanager
def set_ctx_getter(ctx_getter):
    global global_ctx_getter
    prev = global_ctx_getter
    try:
        global_ctx_getter = ctx_getter
        yield
    finally:
        global_ctx_getter = prev


class AbstractImplCtx:
    """
    Context object for writing abstract implementations for custom operators.
    """

    def __init__(self, _shape_env, _op):
        self._shape_env = _shape_env
        self._op = _op

    def create_unbacked_symint(self, *, min=2, max=None) -> torch.SymInt:
        """Constructs a new symint (symbolic int) representing a data-dependent value.

        This is useful for writing the abstract implementation (which is necessary
        for torch.compile) for a CustomOp where an output Tensor has a size
        that depends on the data of the input Tensors.

        Args:
            min (int): A statically known inclusive lower bound for this symint.
                min must be at least 2 due to implementation details of
                torch.compile. Default: 2.
            max (Optional[int]): A statically known inclusive upper bound for this
                symint. Default: None

        .. warning:

            It is important that the ``min`` and ``max`` (if not None) values are set
            correctly, otherwise, there will be undefined behavior under
            torch.compile. The default value of ``min`` is 2 due to torch.compile
            specializing on 0/1 sizes.

            You must also verify that your implementation on concrete Tensors
            (e.g. CPU/CUDA) only returns Tensors where the size that corresponds
            to the symint also has respects these constraint.
            The easiest way to do this is to add an assertion in the CPU/CUDA/etc
            implementation that the size follows these bounds.

        Example::

            >>> # an operator with data-dependent output shape
            >>> @custom_op("mylibrary::custom_nonzero")
            >>> def custom_nonzero(x: Tensor) -> Tensor:
            >>>     ...
            >>>
            >>> @custom_nonzero.impl_abstract():
            >>> def custom_nonzero_abstract(x):
            >>>     # Number of nonzero-elements is data-dependent
            >>>     ctx = torch._custom_op.get_ctx()
            >>>     nnz = ctx.create_unbacked_symint()
            >>>     shape = [x.dim(), nnz]
            >>>     result = x.new_empty(shape, dtype=torch.long)
            >>>     return result
            >>>
            >>> @numpy_nonzero.impl(['cpu', 'cuda'])
            >>> def custom_nonzero_impl(x):
            >>>     x_np = to_numpy(x)
            >>>     res = np.stack(np.nonzero(x_np), axis=1)
            >>>     # the size associated with ctx.create_unbacked_symint()
            >>>     # must be constrained in the same way, so we add an assertion here.
            >>>     if res.shape[0] < 2 or res.shape[0] > x.numel():
            >>>         raise RuntimeError("not supported")
            >>>     return torch.tensor(res, device=x.device)

        """
        if (
            self._shape_env is None
            or not self._shape_env.allow_dynamic_output_shape_ops
        ):
            raise torch._subclasses.fake_tensor.DynamicOutputShapeException(self._op)

        if isinstance(min, torch.SymInt) or isinstance(max, torch.SymInt):
            raise ValueError(
                f"ctx.create_unbacked_symint(min={min}, max={max}): expected "
                f"min and max to be statically known ints but got SymInt. "
                f"This is not supported."
            )

        if min < 2:
            raise ValueError(
                f"ctx.create_unbacked_symint(min={min}, ...): expected min to be "
                f"greater than or equal to 2. PyTorch only supports new "
                f"data-dependent sizes of >= 2"
            )

        result = self._shape_env.create_unbacked_symint()
        torch.fx.experimental.symbolic_shapes.constrain_range(result, min=2, max=max)
        return result
