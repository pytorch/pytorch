# mypy: allow-untyped-defs
import contextlib
import functools
from typing import Callable, Optional
from typing_extensions import deprecated

import torch
from torch._library.utils import Kernel, RegistrationHandle


class AbstractImplHolder:
    """A holder where one can register an fake impl to."""

    def __init__(self, qualname: str):
        self.qualname: str = qualname
        self.kernel: Optional[Kernel] = None
        self.lib: Optional[torch.library.Library] = None

    def register(self, func: Callable, source: str) -> RegistrationHandle:
        """Register an fake impl.

        Returns a RegistrationHandle that one can use to de-register this
        fake impl.
        """
        if self.kernel is not None:
            raise RuntimeError(
                f"register_fake(...): the operator {self.qualname} "
                f"already has an fake impl registered at "
                f"{self.kernel.source}."
            )
        if torch._C._dispatch_has_kernel_for_dispatch_key(self.qualname, "Meta"):
            raise RuntimeError(
                f"register_fake(...): the operator {self.qualname} "
                f"already has an DispatchKey::Meta implementation via a "
                f"pre-existing torch.library or TORCH_LIBRARY registration. "
                f"Please either remove that registration or don't call "
                f"register_fake."
            )

        if torch._C._dispatch_has_kernel_for_dispatch_key(
            self.qualname, "CompositeImplicitAutograd"
        ):
            raise RuntimeError(
                f"register_fake(...): the operator {self.qualname} "
                f"already has an implementation for this device type via a "
                f"pre-existing registration to "
                f"DispatchKey::CompositeImplicitAutograd."
                f"CompositeImplicitAutograd operators do not need an fake "
                f"impl; "
                f"instead, the operator will decompose into its constituents "
                f"and those "
                f"can have fake impls defined on them."
            )

        # Store the kernel in this holder
        self.kernel = Kernel(func, source)

        # Also register the fake impl to Meta key
        if self.lib is None:
            ns = self.qualname.split("::")[0]
            self.lib = torch.library.Library(ns, "FRAGMENT")
        meta_kernel = construct_meta_kernel(self.qualname, self)
        self.lib.impl(self.qualname, meta_kernel, "Meta")

        def deregister_fake_class():
            if self.lib:
                self.lib._destroy()
                self.lib = None
            self.kernel = None

        return RegistrationHandle(deregister_fake_class)


def construct_meta_kernel(
    qualname: str, abstract_impl_holder: AbstractImplHolder
) -> Callable:
    assert abstract_impl_holder.kernel is not None

    @functools.wraps(abstract_impl_holder.kernel.func)
    def meta_kernel(*args, **kwargs):
        assert abstract_impl_holder.kernel is not None
        source = abstract_impl_holder.kernel.source

        def error_on_ctx():
            raise RuntimeError(
                f"Attempted to call get_ctx() for the meta implementation "
                f"for {qualname} (implemented at {source})"
                f"You have presumably called get_ctx() because the operator "
                f"has a data-dependent output shape; if so, there is no "
                f"such meta implementation and this error is the correct "
                f"behavior."
            )

        with set_ctx_getter(error_on_ctx):
            return abstract_impl_holder.kernel(*args, **kwargs)

    return meta_kernel


def get_none():
    return None


global_ctx_getter: Callable = get_none


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
    Context object for writing fake implementations for custom operators.
    """

    def __init__(self, _fake_mode, _op):
        self._fake_mode = _fake_mode
        self._shape_env = _fake_mode.shape_env
        self._op = _op

    @deprecated(
        "`create_unbacked_symint` is deprecated, please use `new_dynamic_size` instead",
        category=FutureWarning,
    )
    def create_unbacked_symint(self, *, min=2, max=None) -> torch.SymInt:
        return self.new_dynamic_size(min=min, max=max)

    def new_dynamic_size(self, *, min=0, max=None) -> torch.SymInt:
        """Constructs a new symint (symbolic int) representing a data-dependent value.

        This is useful for writing the fake implementation (which is necessary
        for torch.compile) for a CustomOp where an output Tensor has a size
        that depends on the data of the input Tensors.

        Args:
            min (int): A statically known inclusive lower bound for this symint. Default: 0
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

            >>> # An operator with data-dependent output shape
            >>> lib = torch.library.Library("mymodule", "FRAGMENT")
            >>> lib.define("mymodule::custom_nonzero(Tensor x) -> Tensor")
            >>>
            >>> @torch.library.register_fake("mymodule::custom_nonzero")
            >>> def _(x):
            >>>     # Number of nonzero-elements is data-dependent.
            >>>     # Since we cannot peek at the data in an fake impl,
            >>>     # we use the ctx object to construct a new symint that
            >>>     # represents the data-dependent size.
            >>>     ctx = torch.library.get_ctx()
            >>>     nnz = ctx.new_dynamic_size()
            >>>     shape = [nnz, x.dim()]
            >>>     result = x.new_empty(shape, dtype=torch.int64)
            >>>     return result
            >>>
            >>> @torch.library.impl(lib, "custom_nonzero", "CPU")
            >>> def _(x):
            >>>     x_np = x.numpy()
            >>>     res = np.stack(np.nonzero(x_np), axis=1)
            >>>     return torch.tensor(res, device=x.device)

        """
        if (
            self._shape_env is None
            or not self._shape_env.allow_dynamic_output_shape_ops
        ):
            raise torch._subclasses.fake_tensor.DynamicOutputShapeException(self._op)

        if isinstance(min, torch.SymInt) or isinstance(max, torch.SymInt):
            raise ValueError(
                f"ctx.new_dynamic_size(min={min}, max={max}): expected "
                f"min and max to be statically known ints but got SymInt. "
                f"This is not supported."
            )

        if min < 0:
            raise ValueError(
                f"ctx.new_dynamic_size(min={min}, ...): expected min to be "
                f"greater than or equal to 0: this API can only create "
                f"non-negative sizes."
            )

        result = self._shape_env.create_unbacked_symint()
        torch.fx.experimental.symbolic_shapes._constrain_range_for_size(
            result, min=min, max=max
        )
        return result
