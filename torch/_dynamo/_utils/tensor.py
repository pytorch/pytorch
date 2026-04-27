from __future__ import annotations

import dataclasses
import functools
import gc
import inspect
import itertools
import logging
import re
import time
from contextlib import contextmanager
from typing import Any, Generic, overload, TYPE_CHECKING, TypeVar
from typing_extensions import TypeIs

import torch
from torch._subclasses.meta_utils import is_sparse_compressed
from torch.utils._python_dispatch import is_traceable_wrapper_subclass


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Sequence


try:
    from torch._subclasses.fake_tensor import is_fake
except ImportError:
    pass


T = TypeVar("T")
log = logging.getLogger("torch._dynamo.utils")


def nothing(*args: Any, **kwargs: Any) -> None:
    return None


def clone_tensor(x: torch.Tensor) -> torch.Tensor:
    """Clone the tensor and its gradient"""
    y = x.clone().requires_grad_(x.requires_grad)
    if x.is_leaf and x.grad is not None:
        y.grad = x.grad.clone()
    return y


def _copy_dynamo_attr(src: torch.Tensor, dst: torch.Tensor, attr: str) -> None:
    """Copy a single dynamo attribute from src to dst, or remove it from dst if src doesn't have it."""
    if hasattr(src, attr):
        setattr(dst, attr, getattr(src, attr).copy())
    elif hasattr(dst, attr):
        delattr(dst, attr)


def copy_dynamo_tensor_attributes(src: torch.Tensor, dst: torch.Tensor) -> None:
    """
    Copy dynamo-specific tensor attributes from src to dst.
    These attributes are used for dynamic shape marking and must be preserved
    when cloning or casting tensors. If src doesn't have an attribute but dst does,
    the attribute is removed from dst.
    """
    _copy_dynamo_attr(src, dst, "_dynamo_dynamic_indices")
    _copy_dynamo_attr(src, dst, "_dynamo_unbacked_indices")
    _copy_dynamo_attr(src, dst, "_dynamo_hint_overrides")
    _copy_dynamo_attr(src, dst, "_dynamo_shape_ids")
    _copy_dynamo_attr(src, dst, "_dynamo_strict_unbacked_indices")
    _copy_dynamo_attr(src, dst, "_dynamo_weak_dynamic_indices")


def clone_input(x: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    """copy while preserving strides"""
    # TODO: this is questionable
    if is_fake(x):
        # this func fails on fake tensors in __torch_dispatch__
        return x

    def torch_clone(x: torch.Tensor) -> torch.Tensor:
        y = torch.clone(x)
        if x.is_leaf:
            y.requires_grad_(x.requires_grad)
        if x.is_leaf and x.grad is not None:
            y.grad = clone_input(x.grad, dtype=dtype)
        copy_dynamo_tensor_attributes(x, y)
        return y

    with torch.no_grad():
        if x.device.type == "xla":
            # Access data_ptr() for a xla tensor will cause crash
            return torch_clone(x)

        # Handle sparse storage (no stride).
        if x.layout is torch.sparse_coo:
            return torch.sparse_coo_tensor(
                torch_clone(x._indices()),
                torch_clone(x._values()),
                x.shape,
                is_coalesced=x.is_coalesced(),
            )
        elif is_sparse_compressed(x):
            if x.layout in {torch.sparse_csr, torch.sparse_bsr}:
                compressed_indices = x.crow_indices()
                plain_indices = x.col_indices()
            else:
                compressed_indices = x.ccol_indices()
                plain_indices = x.row_indices()
            return torch.sparse_compressed_tensor(
                torch_clone(compressed_indices),
                torch_clone(plain_indices),
                torch_clone(x.values()),
                x.shape,
                layout=x.layout,
            )
        elif is_traceable_wrapper_subclass(x):
            # Questionable - but this is required to not fail executorch related
            # torchao tests.
            return torch_clone(x)

        needed_size = sum(
            (shape - 1) * stride for shape, stride in zip(x.size(), x.stride())
        )
        if x.is_quantized:
            result = torch.empty_quantized((needed_size + 32,), x)
        else:
            result = torch.empty(
                needed_size + 32, dtype=dtype or x.dtype, device=x.device
            )
        cache_line_offset = (
            (x.data_ptr() - result.data_ptr()) % 32
        ) // x.element_size()
        result.as_strided_(x.size(), x.stride(), cache_line_offset)
        try:
            result.copy_(x.clone())
            if x.is_leaf:
                result.requires_grad_(x.requires_grad)
            if x.is_leaf and x.grad is not None:
                result.grad = clone_input(x.grad, dtype=dtype)
        except RuntimeError:
            # RuntimeError: unsupported operation: more than one element of the written-to
            # tensor refers to a single memory location. Please clone() the tensor before
            # performing the operation.
            return torch_clone(x)
        copy_dynamo_tensor_attributes(x, result)
        return result


@overload
def clone_inputs(
    example_inputs: dict[str, T | tuple[T, ...]],
) -> dict[str, list[T]]: ...


@overload
def clone_inputs(example_inputs: Sequence[T]) -> list[T]: ...


def clone_inputs(example_inputs: Any) -> Any:
    res: dict[str, Any] | list[Any]
    if type(example_inputs) is dict:
        res = dict(example_inputs)
        for key, value in res.items():
            if isinstance(value, tuple):
                res[key] = clone_inputs(value)
            else:
                assert isinstance(value, torch.Tensor), type(value)
                res[key] = clone_input(value)
        return res

    res = list(example_inputs)
    for i in range(len(res)):
        if isinstance(res[i], torch.Tensor):
            res[i] = clone_input(res[i])
    return res


def skip_frame_if_in_functorch_mode(val: torch.Tensor) -> None:
    try:
        val.data_ptr()  # will throw for functorch tensors
    except RuntimeError as e:
        from ..exc import unimplemented

        # This will be GradTrackingTensor/BatchedTensor/etc
        functorch_subclass_name = re.sub(r"\(.*", "", repr(val))

        unimplemented(
            gb_type="skip frame due to being in functorh mode",
            context="",
            explanation=f"torch.compile cannot be run in context: {functorch_subclass_name}. Skipping frame.",
            hints=[],
            from_exc=e,
            skip_frame=True,
        )


@contextmanager
def preserve_rng_state() -> Generator[None, None, None]:
    disable_functorch = torch._C._DisableFuncTorch
    disable_current_modes = torch.utils._python_dispatch._disable_current_modes
    with disable_current_modes(), disable_functorch():
        rng_state = torch.clone(torch.random.get_rng_state())
        skip_frame_if_in_functorch_mode(rng_state)
        if torch.cuda.is_available():
            cuda_rng_state = torch.clone(torch.cuda.get_rng_state())
        if torch.xpu.is_available():
            xpu_rng_state = torch.clone(torch.xpu.get_rng_state())
    try:
        yield
    finally:
        with torch.utils._python_dispatch._disable_current_modes():
            torch.random.set_rng_state(rng_state)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(cuda_rng_state)  # type: ignore[possibly-undefined]
            if torch.xpu.is_available():
                torch.xpu.set_rng_state(xpu_rng_state)  # type: ignore[possibly-undefined]


def is_jit_model(
    model0: Any,
) -> TypeIs[
    torch.jit._trace.TopLevelTracedModule
    | torch.jit._script.RecursiveScriptModule
    | torch.jit.ScriptFunction[Any, Any]
    | torch.jit.ScriptModule
]:
    return isinstance(
        model0,
        (
            torch.jit._trace.TopLevelTracedModule,
            torch.jit._script.RecursiveScriptModule,
            torch.jit.ScriptFunction,
            torch.jit.ScriptModule,
        ),
    )


def torchscript(model: Any, example_inputs: Any, verbose: bool = False) -> Any:
    if is_jit_model(model):
        # already done?
        return model

    try:
        return torch.jit.trace(model, example_inputs)
    except Exception:
        try:
            return torch.jit.script(model)
        except Exception:
            if verbose:
                log.exception("jit error")
            else:
                log.error("Both torch.jit.trace and torch.jit.script failed")
    return None


def getfile(obj: Any) -> str | None:
    try:
        return inspect.getfile(obj)
    except (TypeError, OSError):
        return None


def is_namedtuple(obj: Any) -> bool:
    """Test if an object is a namedtuple or a torch.return_types.* quasi-namedtuple"""
    return is_namedtuple_cls(type(obj))


def is_namedtuple_cls(cls: Any) -> bool:
    """Test if an object is a namedtuple or a (torch.return_types|torch.autograd.forward_ad).* quasi-namedtuple"""
    try:
        if issubclass(cls, tuple):
            module = getattr(cls, "__module__", None)
            if module in ("torch.return_types", "torch.autograd.forward_ad"):
                return True
            if isinstance(getattr(cls, "_fields", None), tuple) and callable(
                getattr(cls, "_make", None)
            ):
                # The subclassing style namedtuple can have an extra base `typing.Generic`
                bases = tuple(t for t in cls.__bases__ if t is not Generic)
                if bases == (tuple,):
                    # This is a namedtuple type directly created by `collections.namedtuple(...)`
                    return True
                if bases and any(
                    (
                        # Subclass of namedtuple
                        is_namedtuple_cls(t)
                        # For subclasses of namedtuple, the __new__ method should not be customized
                        and cls.__new__ is t.__new__
                    )
                    for t in bases
                ):
                    return True
    except TypeError:
        pass
    return False


@functools.lru_cache(1)
def namedtuple_fields(cls: type) -> tuple[str, ...]:
    """Get the fields of a namedtuple or a torch.return_types.* quasi-namedtuple"""
    if cls is slice:
        return ("start", "stop", "step")

    assert issubclass(cls, tuple)
    if hasattr(cls, "_fields"):
        # normal namedtuples
        return cls._fields

    @dataclasses.dataclass
    class Marker:
        index: int

    # frustrating ones e.g. torch.return_types.max
    assert cls.__module__ == "torch.return_types"
    obj = cls(map(Marker, range(cls.n_fields)))  # type: ignore[attr-defined]
    fields: dict[str, int] = {}
    for name in dir(obj):
        if name[0] != "_" and isinstance(getattr(obj, name), Marker):
            fields[name] = getattr(obj, name).index
    assert len(fields) == cls.n_fields  # type: ignore[attr-defined]
    return tuple(sorted(fields, key=fields.get))  # type: ignore[arg-type]


def checkpoint_params(gm: torch.fx.GraphModule) -> Callable[[], None]:
    with torch.no_grad():
        rng_state = torch.clone(torch.random.get_rng_state())
        if torch.cuda.is_available():
            cuda_rng_state = torch.clone(torch.cuda.get_rng_state())
        saved_state = [
            (param, param._version, torch.clone(param))
            # pyrefly: ignore [bad-argument-type]
            for param in itertools.chain(gm.parameters(), gm.buffers())
        ]

    def restore() -> None:
        with torch.no_grad():
            torch.random.set_rng_state(rng_state)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(cuda_rng_state)
            for param, version, original_value in saved_state:
                if param._version != version:
                    param.copy_(original_value)

    return restore


def timed(
    model: Any, example_inputs: Iterable[Any], times: int = 1
) -> tuple[Any, float]:
    if torch.cuda.is_available():
        synchronize = torch.cuda.synchronize
    else:
        synchronize = nothing

    synchronize()
    gc.collect()
    torch.manual_seed(1337)
    t0 = time.perf_counter()
    for _ in range(times):
        result = model(*example_inputs)
        synchronize()
    t1 = time.perf_counter()
    return result, t1 - t0  # type: ignore[possibly-undefined]


def check_is_cuda(gm: torch.fx.GraphModule, example_inputs: Iterable[Any]) -> bool:
    return all(x.is_cuda for x in itertools.chain(example_inputs, gm.parameters(True)))
