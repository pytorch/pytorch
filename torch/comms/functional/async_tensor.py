# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Async tensor utilities for torch.comms.

This module provides async tensor wrappers and utilities for handling
asynchronous collective operations in torch.comms.
"""

from typing import Any, Union

import torch
from torch._C._comms import TorchWork


__all__ = ["FakeWork", "TorchCommsAsyncTensor"]


class _OnceWaitWork:
    """Wrapper that ensures wait() is only called once for shared work handles.

    Use this when multiple tensors share the same underlying work handle
    (e.g., in coalesced collectives) to avoid generating duplicate wait ops
    in the traced graph.
    """

    __slots__ = ["_work", "_waited", "_cached_result"]

    def __init__(self, work: Any) -> None:
        self._work = work
        self._waited = False
        self._cached_result = None

    def wait(self):
        if not self._waited:
            self._waited = True
            if self._work is not None:
                self._cached_result = self._work.wait()
        return self._cached_result


class FakeWork:
    """Fake work object for async ops during tracing.

    When .wait() is called, generates the torchcomm_wait_tensors op.
    Used in tracing contexts where we need to track async operations
    without an actual work handle.

    Since we trace functional ops, the tensors are the RESULT of the collective.
    wait() creates proper data dependencies by waiting on these result tensors.
    """

    __slots__ = ["tensors", "_waited", "_original_result"]

    def __init__(self, result: list | torch.Tensor) -> None:
        """Initialize with the original result structure.

        Args:
            result: The original op result - can be a single tensor, list of tensors,
                   or nested structure. The structure is preserved for wait() return.
        """
        self._original_result = result
        self.tensors = self._flatten(result)
        self._waited = False

    @staticmethod
    def _flatten(x) -> list[torch.Tensor]:
        """Flatten any tensor structure into a flat list for wait_tensors."""
        if x is None:
            return []
        if isinstance(x, torch.Tensor):
            return [x]
        if isinstance(x, (list, tuple)):
            tensors = []
            for item in x:
                tensors.extend(FakeWork._flatten(item))
            return tensors
        return []

    def _unflatten(self, waited_tensors: list[torch.Tensor], template):
        """Reconstruct the original structure with waited tensors.

        The waited_tensors list elements are already properly associated with
        traced getitem nodes by functionalization/ProxyTensorMode. We just
        need to reconstruct the original structure by indexing.
        """
        idx = 0

        def reconstruct(t):
            nonlocal idx
            if t is None:
                return None
            if isinstance(t, torch.Tensor):
                result = waited_tensors[idx]
                idx += 1
                return result
            if isinstance(t, (list, tuple)):
                return type(t)(reconstruct(item) for item in t)
            return t

        return reconstruct(template)

    def wait(self):
        if not self._waited:
            self._waited = True
            if not self.tensors:
                return self._original_result
            waited = torch.ops.torchcomms.torchcomm_wait_tensors_(self.tensors)
            self._original_result = self._unflatten(waited, self._original_result)
        return self._original_result


class TorchCommsAsyncTensor(torch.Tensor):
    """
    Async tensor wrapper for torchcomms collectives.

    Similar to AsyncCollectiveTensor but uses TorchWork for waiting.
    Automatically waits on the work handle when the tensor is first accessed.

    If multiple tensors are associated with the same work handle,
    the wait() function will be a no-op on subsequent calls.

    The wrapper is transparent to autograd - grad_fn is delegated to the
    wrapped elem tensor so gradients flow through correctly.
    """

    elem: torch.Tensor
    work: Union[TorchWork, "_OnceWaitWork", None]
    completed: bool

    __slots__ = ["elem", "work", "completed"]

    @staticmethod
    def __new__(
        cls, elem: torch.Tensor, work: Union[TorchWork, "_OnceWaitWork"]
    ) -> "TorchCommsAsyncTensor":
        r = torch.Tensor._make_wrapper_subclass(
            cls,
            elem.size(),
            strides=elem.stride(),
            storage_offset=elem.storage_offset(),
            dtype=elem.dtype,
            layout=elem.layout,
            device=elem.device,
            requires_grad=elem.requires_grad,
        )
        r.elem = elem
        r.work = work
        r.completed = False
        return r

    @property
    def grad_fn(self):  # pyrefly: ignore[bad-override]
        """Delegate grad_fn to the wrapped elem tensor for autograd transparency."""
        return self.elem.grad_fn if self.elem is not None else None

    @property
    def is_leaf(self):  # pyrefly: ignore[bad-override]
        """Delegate is_leaf to the wrapped elem tensor."""
        return self.elem.is_leaf if self.elem is not None else True

    @property
    def grad(self):  # pyrefly: ignore[bad-override]
        """Delegate grad to the wrapped elem tensor."""
        return self.elem.grad if self.elem is not None else None

    @grad.setter
    def grad(self, value):
        """Set grad on the wrapped elem tensor."""
        if self.elem is not None:
            self.elem.grad = value

    def __repr__(self) -> str:  # pyre-ignore[14]: override
        if self.work is None:
            return f"TorchCommsAsyncTensor({self.elem})"
        return f"TorchCommsAsyncTensor({self.trigger_wait()})"

    def trigger_wait(self) -> torch.Tensor:
        if not self.completed and self.work is not None:
            self.work.wait()
            self.completed = True
        return self.elem

    def wait(self) -> torch.Tensor:
        if self.work is not None:
            self.work.wait()
        self.completed = True
        return self.elem

    def backward(
        self, gradient=None, retain_graph=None, create_graph=False, inputs=None
    ):
        """Delegate backward to the wrapped elem tensor."""
        self.trigger_wait()
        return self.elem.backward(gradient, retain_graph, create_graph, inputs)

    def __tensor_flatten__(
        self,
    ) -> tuple[list[str], tuple[Union[TorchWork, "_OnceWaitWork"], bool]]:
        """Flatten for dynamo tracing."""
        return ["elem"], (self.work, self.completed)  # pyrefly: ignore[bad-return]

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: dict[str, torch.Tensor],
        metadata: tuple[Union[TorchWork, "_OnceWaitWork"], bool],
        outer_size: torch.Size,
        outer_stride: tuple[int, ...],
    ) -> "TorchCommsAsyncTensor":
        """Unflatten from dynamo tracing."""
        elem = inner_tensors["elem"]
        work, completed = metadata
        result = TorchCommsAsyncTensor(elem, work)
        result.completed = completed
        return result

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # type: ignore[override]
        """Unwrap and wait on tensor access."""

        def unwrap(e: "TorchCommsAsyncTensor") -> torch.Tensor:
            return e.trigger_wait()

        unwrapped_args = torch.utils._pytree.tree_map_only(
            TorchCommsAsyncTensor, unwrap, args
        )
        unwrapped_kwargs = torch.utils._pytree.tree_map_only(
            TorchCommsAsyncTensor, unwrap, kwargs or {}
        )

        return func(*unwrapped_args, **unwrapped_kwargs)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Handle torch functions by unwrapping to elem."""

        def unwrap(e):
            if isinstance(e, TorchCommsAsyncTensor):
                return e.trigger_wait()
            return e

        unwrapped_args = torch.utils._pytree.tree_map(unwrap, args)
        unwrapped_kwargs = torch.utils._pytree.tree_map(unwrap, kwargs or {})

        return func(*unwrapped_args, **unwrapped_kwargs)


def _are_we_tracing() -> bool:
    """Check if we're in a tracing/compiling context."""
    from torch.compiler import is_compiling as is_torchdynamo_compiling

    if is_torchdynamo_compiling():
        return True
    # If fake mode is turned on, we are almost definitely compiling/tracing.
    if torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.FAKE) is not None:
        return True
    return False


def _maybe_wrap_tensor(
    tensor: torch.Tensor,
    work: Union[TorchWork, "_OnceWaitWork", None, torch.Tensor],
) -> torch.Tensor:
    """Wrap tensor for async wait semantics.

    Behavior depends on context:
    - If work is a tensor (from patched autograd path): return it directly
    - If tracing (torch.compile): wait sync and return result tensor from work
    - Otherwise: return TorchCommsAsyncTensor for lazy wait

    For FakeWork from autograd path: wraps the result tensor (with grad_fn) from
    FakeWork.tensors instead of the passed tensor, so grad_fn is properly delegated.
    """
    if work is None:
        return tensor

    # Autograd wrapper returned the tensor with grad_fn - use it
    # note that it's already been wrapped internally using
    # _wrap_result_with_registered_work
    if isinstance(work, (torch.Tensor, tuple, list)):
        return work

    if _are_we_tracing():
        if isinstance(work, (FakeWork, _OnceWaitWork)):
            # If we're tracing, wait on the result tensors from FakeWork
            # and return the original result structure
            return work.wait()
        work.wait()
        return tensor

    # Async path - return async tensor wrapper
    return TorchCommsAsyncTensor(tensor, work)


def _wrap_result_with_registered_work(
    result: Any,
) -> Any:
    from torch.comms.functional.collectives import _get_tensor_work

    """Wrap a result (tensor or list/tuple of tensors) with its registered work handle.

    This helper retrieves the work handle from the tensor registry and wraps
    the result for async wait semantics.

    Args:
        result: A single tensor or list/tuple of tensors to wrap
        get_tensor_work: Function to retrieve work handle for a tensor

    Returns:
        The wrapped result (same type as input)
    """
    if isinstance(result, torch.Tensor):
        work = _get_tensor_work(result)
        if work is not None:
            return _maybe_wrap_tensor(result, work)  # type: ignore[parameter-type]
        return result

    if isinstance(result, (list, tuple)):
        first_tensor = result[0] if result else None
        if first_tensor is not None:
            work = _get_tensor_work(first_tensor)
            if work is not None:
                # Wrap with _OnceWaitWork so wait() is only called once
                once_work = _OnceWaitWork(work)
                return type(result)(_maybe_wrap_tensor(t, once_work) for t in result)
        return result

    return result
