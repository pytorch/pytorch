"""Utilities shared by optimizer implementations."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar


# None distinguishes the default policy from an explicit choice of False.
_preserve_state_dtype_on_load: ContextVar[bool | None] = ContextVar(
    "preserve_state_dtype_on_load", default=None
)


@contextmanager
def preserve_state_dtype_on_load(preserve_dtype: bool = True) -> Iterator[None]:
    r"""preserve_state_dtype_on_load(preserve_dtype=True)

    Control the dtype of state tensors loaded by
    :meth:`~torch.optim.Optimizer.load_state_dict` within this context.

    Args:
        preserve_dtype (bool, optional): If ``True``, preserve the dtype of saved
            state tensors versus casting to the parameter's dtype. If ``False``,
            cast these tensors to the parameter dtype for floating-point parameters.
            Note that the state will be moved to match the parameter's device in
            both cases. Default: ``True``.

    The setting applies only to state processed by
    :meth:`~torch.optim.Optimizer.load_state_dict` in the current thread or
    asynchronous task, including nested calls. Exiting the context restores the
    previous setting, including when an exception is raised. Using this context
    manager will suppress the warning about the planned change to the default
    loading behavior.

    .. note::
        Preserved state dtypes must be compatible with the selected optimizer
        implementation, for example fused bfloat16 AdamW. Otherwise,
        :meth:`~torch.optim.Optimizer.step` may raise a dtype mismatch error
        or use unintended precision. Pass ``preserve_dtype=False`` to retain
        automatic casting to match parameter dtype, or explicitly convert state
        tensors before stepping, for example with
        :meth:`~torch.optim.Optimizer.register_load_state_dict_post_hook`.

    .. note::
        Subclasses that override ``load_state_dict`` control their own loading
        behavior. State loaded independently by an override is unaffected.
        Calls to :meth:`~torch.optim.Optimizer.load_state_dict`, through
        ``super()`` or a wrapped optimizer, use the context's dtype policy.

    .. note::
        The ``step`` tensor keeps its saved dtype and device unless the parameter
        group is fused or capturable, which moves it to the parameter's device as
        ``torch.float32`` for performance reasons. This behavior is independent of
        ``preserve_dtype``.

    Examples::

        >>> # xdoctest: +SKIP
        >>> with torch.optim.preserve_state_dtype_on_load():
        ...     optimizer.load_state_dict(checkpoint)
        >>> with torch.optim.preserve_state_dtype_on_load(False):
        ...     optimizer.load_state_dict(checkpoint)
    """
    token = _preserve_state_dtype_on_load.set(preserve_dtype)
    try:
        yield
    finally:
        _preserve_state_dtype_on_load.reset(token)
