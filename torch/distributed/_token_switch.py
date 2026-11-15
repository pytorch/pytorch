# mypy: allow-untyped-defs
from __future__ import annotations

import abc
import ctypes
import importlib.util
import os
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch
from torch._library.opaque_object import CustomClassBase, register_custom_class


if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup


# Keeps preloaded shared libraries resident for the process lifetime.
_NCCL_EP_KEEPALIVE: list[ctypes.CDLL] = []


def _find_pkg_dir(name: str) -> str | None:
    # Locate an installed package's directory without importing it. find_spec
    # raises (rather than returning None) when a dotted name's parent namespace
    # is absent, so guard that.
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return spec.submodule_search_locations[0]


def _prepare_nccl_extensions() -> None:
    # Dynamic (USE_SYSTEM_NCCL=ON / wheel) build only: the extension NEEDED-links
    # libnccl_ep.so, which the nccl-extensions wheel provides at runtime. Point
    # nccl-ep's JIT at its EP headers (NCCL_EP_HOME) and the nvidia.nccl wheel's
    # NCCL headers (NCCL_HOME), and make libnccl_ep resolvable before importing
    # the extension (its libnccl dependency is already loaded by torch). The
    # static build bakes all of this in and never reaches here.
    # Locate nccl without importing its initializer or EP's Python bindings.
    nccl_pkg = _find_pkg_dir("nccl")
    if nccl_pkg is None or not os.path.isdir(os.path.join(nccl_pkg, "ep")):
        raise ImportError(
            "This TokenSwitchNCCL build requires the nccl-extensions package. "
            "Install it with `pip install nccl-extensions`."
        )
    ep_dir = os.path.join(nccl_pkg, "ep")
    cuda_version = torch.version.cuda
    if cuda_version is None:
        raise ImportError("TokenSwitchNCCL requires a CUDA build of PyTorch.")
    cuda_major = cuda_version.split(".")[0]
    lib = os.path.join(ep_dir, "lib", f"cu{cuda_major}", "libnccl_ep.so.0")
    if not os.path.isfile(lib):
        raise ImportError(
            f"nccl-extensions is missing the CUDA {cuda_major} library at {lib}. "
            "Install it with `pip install nccl-extensions`."
        )
    if "NCCL_EP_HOME" not in os.environ:
        ep_headers = os.path.join(ep_dir, "include", "nccl_ep")
        if not os.path.isdir(ep_headers):
            raise ImportError(
                f"nccl-extensions at {nccl_pkg} is missing the EP JIT headers "
                f"expected at {ep_headers}; reinstall nccl-extensions or set NCCL_EP_HOME."
            )
        os.environ["NCCL_EP_HOME"] = ep_dir

    # Point the JIT at NCCL headers. libnccl.so itself needs no preload here: a
    # USE_SYSTEM_NCCL=ON torch (the only build that reaches this path) already
    # dynamically loaded the system libnccl, so libnccl_ep.so's NEEDED
    # libnccl.so.2 resolves to that already-loaded copy.
    if "NCCL_HOME" not in os.environ:
        nvidia_nccl = _find_pkg_dir("nvidia.nccl")
        if nvidia_nccl is not None and os.path.isdir(
            os.path.join(nvidia_nccl, "include")
        ):
            os.environ.setdefault("NCCL_HOME", nvidia_nccl)

    # The extension's RPATH does not cover nccl-extensions' per-CUDA lib/cu<N>
    # directory; load libnccl_ep by path to register its SONAME
    # (libnccl_ep.so.0, the extension's NEEDED entry) for the import below.
    _NCCL_EP_KEEPALIVE.append(ctypes.CDLL(lib, mode=ctypes.RTLD_LOCAL))


def _import_nccl_ep() -> Any:
    # The EP bindings live in the optional torch._nccl_ep extension (USE_NCCL_EP).
    # A USE_SYSTEM_NCCL=OFF build links libnccl_ep statically and bakes its JIT
    # header paths, so the extension imports directly -- self-contained, no
    # nccl-extensions. A USE_SYSTEM_NCCL=ON build NEEDED-links libnccl_ep from the
    # nccl-extensions wheel, so the first import fails until it is set up; fall
    # back to that and retry.
    try:
        # pyrefly: ignore [missing-import]  # built only with USE_NCCL_EP
        import torch._nccl_ep as _ep

        return _ep
    except ImportError:
        pass

    _prepare_nccl_extensions()
    try:
        # pyrefly: ignore [missing-import]  # built only with USE_NCCL_EP
        import torch._nccl_ep as _ep
    except ImportError as e:
        raise ImportError(
            "torch._nccl_ep is unavailable; this PyTorch was not built with "
            "USE_NCCL_EP (or, for a USE_SYSTEM_NCCL=ON build, the nccl-extensions wheel "
            "is missing)."
        ) from e

    return _ep


class _NCCLEPObject(CustomClassBase):
    """Keep a native EP object opaque to compiler tracing.

    Routing updates mutate the native handle in place. Wrappers of one handle
    share ``versions``; ``version`` is the update this wrapper observed, so
    communication through a stale routing (e.g. a backward whose forward
    predates an update) raises instead of using the new routes.
    """

    def __init__(self, value: Any = None, versions: list[int] | None = None):
        self.value = value
        self.versions = [0] if versions is None else versions
        self.version = self.versions[0]

    def native(self) -> Any:
        if self.version != self.versions[0]:
            raise RuntimeError(
                "TokenSwitch routing was updated after this routing was created; "
                "use the routing returned by update_routing, and do not update a "
                "routing whose forward still needs its backward"
            )
        return self.value


register_custom_class(_NCCLEPObject, typ="symbolic")


@dataclass(frozen=True, slots=True, init=False)
class Routing:
    _opaque_handle: _NCCLEPObject
    topk_idx: torch.Tensor
    layout: str = "flat"

    def __init__(self, handle: object, topk_idx: torch.Tensor, layout: str = "flat"):
        opaque_handle = (
            handle if isinstance(handle, _NCCLEPObject) else _NCCLEPObject(handle)
        )
        object.__setattr__(self, "_opaque_handle", opaque_handle)
        object.__setattr__(self, "topk_idx", topk_idx)
        object.__setattr__(self, "layout", layout)

    @property
    def handle(self) -> object:
        return self._opaque_handle.value


@torch.library.custom_op(
    "_nccl_ep::create_routing",
    mutates_args=(),
    schema=(
        "(torch.distributed._token_switch._NCCLEPObject group, Tensor topk_idx, "
        "Tensor? per_expert_token_counts, str layout, SymInt alignment) -> "
        "(torch.distributed._token_switch._NCCLEPObject, Tensor?)"
    ),
)
def _create_routing_op(
    group: _NCCLEPObject,
    topk_idx: torch.Tensor,
    per_expert_token_counts: torch.Tensor | None,
    layout: str,
    alignment: int,
) -> tuple[_NCCLEPObject, torch.Tensor | None]:
    ep = _import_nccl_ep()
    native_layouts = {"flat": ep.Layout.FLAT, "expert_major": ep.Layout.EXPERT_MAJOR}
    if layout not in native_layouts:
        raise ValueError(f"Unsupported NCCL EP routing layout: {layout}")
    # Native code writes the counts; clone so this functional op leaves its input intact.
    counts = (
        per_expert_token_counts.clone() if per_expert_token_counts is not None else None
    )
    handle = ep._NcclEpHandle.create(
        group.native(), topk_idx, counts, native_layouts[layout], alignment
    )
    return _NCCLEPObject(handle), counts


@_create_routing_op.register_fake
def _create_routing_fake(group, topk_idx, per_expert_token_counts, layout, alignment):
    counts = (
        torch.empty_like(per_expert_token_counts)
        if per_expert_token_counts is not None
        else None
    )
    return _NCCLEPObject(), counts


@torch.library.custom_op(
    "_nccl_ep::update_routing",
    mutates_args=(),
    schema=(
        "(torch.distributed._token_switch._NCCLEPObject handle, Tensor topk_idx, "
        "Tensor? per_expert_token_counts) -> "
        "(torch.distributed._token_switch._NCCLEPObject, Tensor?)"
    ),
)
def _update_routing_op(handle, topk_idx, per_expert_token_counts):
    counts = (
        per_expert_token_counts.clone() if per_expert_token_counts is not None else None
    )
    handle.value.update(topk_idx, counts)
    handle.versions[0] += 1
    return _NCCLEPObject(handle.value, handle.versions), counts


@_update_routing_op.register_fake
def _update_routing_fake(handle, topk_idx, per_expert_token_counts):
    counts = (
        torch.empty_like(per_expert_token_counts)
        if per_expert_token_counts is not None
        else None
    )
    return _NCCLEPObject(), counts


@torch.library.custom_op(
    "_nccl_ep::dispatch_out",
    mutates_args=("out_tokens", "out_topk_weights", "out_topk_idx"),
)
def _dispatch_op(
    handle: _NCCLEPObject,
    tokens: torch.Tensor,
    topk_weights: torch.Tensor,
    out_tokens: torch.Tensor,
    out_topk_weights: torch.Tensor | None,
    out_topk_idx: torch.Tensor | None,
) -> None:
    _import_nccl_ep()._nccl_ep_dispatch(
        handle.native(),
        tokens,
        topk_weights,
        out_tokens,
        out_topk_weights,
        out_topk_idx,
    )


@torch.library.custom_op("_nccl_ep::combine_out", mutates_args=("out_tokens",))
def _combine_op(
    handle: _NCCLEPObject,
    expert_tokens: torch.Tensor,
    out_tokens: torch.Tensor,
) -> None:
    _import_nccl_ep()._nccl_ep_combine(handle.native(), expert_tokens, out_tokens)


@torch.library.custom_op(
    "_nccl_ep::dispatch_backward_out", mutates_args=("out_tokens",)
)
def _dispatch_backward_op(
    handle: _NCCLEPObject,
    grad_tokens: torch.Tensor,
    out_tokens: torch.Tensor,
) -> None:
    _import_nccl_ep()._nccl_ep_dispatch_backward(
        handle.native(), grad_tokens, out_tokens
    )


@torch.library.custom_op(
    "_nccl_ep::combine_backward_out",
    mutates_args=("out_tokens", "out_topk_weights"),
)
def _combine_backward_op(
    handle: _NCCLEPObject,
    grad_tokens: torch.Tensor,
    grad_topk_weights: torch.Tensor,
    out_tokens: torch.Tensor,
    out_topk_weights: torch.Tensor,
) -> None:
    _import_nccl_ep()._nccl_ep_combine_backward(
        handle.native(), grad_tokens, grad_topk_weights, out_tokens, out_topk_weights
    )


@_dispatch_op.register_fake
@_combine_op.register_fake
@_dispatch_backward_op.register_fake
@_combine_backward_op.register_fake
def _out_op_fake(*args, **kwargs) -> None:
    # Output metadata is already supplied by the caller's buffers.
    return None


@torch.library.custom_op(
    "_nccl_ep::dispatch_async",
    mutates_args=(),
    schema=(
        "(torch.distributed._token_switch._NCCLEPObject handle, Tensor tokens, "
        "Tensor topk_weights, SymInt capacity, str layout) -> (Tensor, Tensor, Tensor?)"
    ),
)
def _dispatch_async_op(
    handle: _NCCLEPObject,
    tokens: torch.Tensor,
    topk_weights: torch.Tensor,
    capacity: int,
    layout: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    outputs = _dispatch_async_fake(handle, tokens, topk_weights, capacity, layout)
    _import_nccl_ep()._nccl_ep_dispatch(
        handle.native(),
        tokens.contiguous(),
        topk_weights.contiguous(),
        *outputs,
        async_op=True,
    )
    return outputs


@_dispatch_async_op.register_fake
def _dispatch_async_fake(
    handle, tokens, topk_weights, capacity, layout, *, initialize_tokens=True
):
    if layout not in ("flat", "expert_major"):
        raise ValueError(f"Unsupported NCCL EP dispatch layout: {layout}")
    weights_shape = (
        (capacity,) if layout == "expert_major" else (capacity, topk_weights.shape[1])
    )
    # Callers opting out must mask unused receive rows.
    allocate_tokens = tokens.new_zeros if initialize_tokens else tokens.new_empty
    return (
        allocate_tokens(capacity, tokens.shape[1]),
        topk_weights.new_zeros(weights_shape),
        tokens.new_full((capacity, topk_weights.shape[1]), -1, dtype=torch.int64)
        if layout == "flat"
        else None,
    )


@torch.library.custom_op("_nccl_ep::combine_async", mutates_args=())
def _combine_async_op(
    handle: _NCCLEPObject,
    expert_tokens: torch.Tensor,
    num_tokens: int,
    capacity: int,
) -> torch.Tensor:
    output = _combine_async_fake(handle, expert_tokens, num_tokens, capacity)
    _import_nccl_ep()._nccl_ep_combine(
        handle.native(), expert_tokens.contiguous(), output, async_op=True
    )
    return output


@_combine_async_op.register_fake
def _combine_async_fake(handle, expert_tokens, num_tokens, capacity):
    # Native combine writes every source row, including zero-contribution rows.
    return expert_tokens.new_empty(num_tokens, expert_tokens.shape[1])


@torch.library.custom_op("_nccl_ep::dispatch_backward_async", mutates_args=())
def _dispatch_backward_async_op(
    handle: _NCCLEPObject,
    grad_tokens: torch.Tensor,
    capacity: int,
    *,
    initialize_tokens: bool = True,
) -> torch.Tensor:
    output = _dispatch_backward_async_fake(
        handle, grad_tokens, capacity, initialize_tokens=initialize_tokens
    )
    _import_nccl_ep()._nccl_ep_dispatch_backward(
        handle.native(), grad_tokens.contiguous(), output, async_op=True
    )
    return output


@_dispatch_backward_async_op.register_fake
def _dispatch_backward_async_fake(
    handle, grad_tokens, capacity, *, initialize_tokens=True
):
    # Native dispatch writes routed rows; callers own unused receive rows.
    allocate_tokens = (
        grad_tokens.new_zeros if initialize_tokens else grad_tokens.new_empty
    )
    return allocate_tokens(capacity, grad_tokens.shape[1])


@torch.library.custom_op("_nccl_ep::combine_backward_async", mutates_args=())
def _combine_backward_async_op(
    handle: _NCCLEPObject,
    grad_tokens: torch.Tensor,
    grad_topk_weights: torch.Tensor,
    num_tokens: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = _combine_backward_async_fake(
        handle, grad_tokens, grad_topk_weights, num_tokens, top_k
    )
    _import_nccl_ep()._nccl_ep_combine_backward(
        handle.native(),
        grad_tokens.contiguous(),
        grad_topk_weights.contiguous(),
        *outputs,
        async_op=True,
    )
    return outputs


@_combine_backward_async_op.register_fake
def _combine_backward_async_fake(
    handle, grad_tokens, grad_topk_weights, num_tokens, top_k
):
    return (
        grad_tokens.new_empty(num_tokens, grad_tokens.shape[1]),
        grad_topk_weights.new_zeros(num_tokens, top_k),
    )


def _dispatch_async_setup_context(ctx, inputs, output):
    handle, tokens, topk_weights, _, _ = inputs
    ctx.handle = handle
    ctx.num_tokens = tokens.shape[0]
    ctx.top_k = topk_weights.shape[1]
    ctx.capacity = output[0].shape[0]
    ctx.needs_weight_grad = ctx.needs_input_grad[2]


def _dispatch_async_backward(ctx, grad_tokens, grad_weights, grad_ids):
    from torch.distributed._functional_collectives import wait_tensor

    if not ctx.needs_weight_grad:
        token_grads = _combine_async_op(
            ctx.handle, grad_tokens.contiguous(), ctx.num_tokens, ctx.capacity
        )
        return None, wait_tensor(token_grads), None, None, None
    if grad_weights is None:
        raise NotImplementedError(
            "TokenSwitch dispatch backward requires received topk_weights"
        )
    token_grads, weight_grads = _combine_backward_async_op(
        ctx.handle,
        grad_tokens.contiguous(),
        grad_weights.contiguous(),
        ctx.num_tokens,
        ctx.top_k,
    )
    return None, wait_tensor(token_grads), wait_tensor(weight_grads), None, None


def _combine_async_setup_context(ctx, inputs, output):
    handle, expert_tokens, _, capacity = inputs
    ctx.handle = handle
    ctx.expert_rows = expert_tokens.shape[0]
    ctx.expert_dtype = expert_tokens.dtype
    ctx.capacity = max(capacity, ctx.expert_rows)


def _combine_async_backward(ctx, grad_tokens):
    from torch.distributed._functional_collectives import wait_tensor

    expert_grads = _dispatch_backward_async_op(
        ctx.handle, grad_tokens.to(ctx.expert_dtype).contiguous(), ctx.capacity
    )
    return None, wait_tensor(expert_grads)[: ctx.expert_rows], None, None


_dispatch_async_op.register_autograd(
    _dispatch_async_backward, setup_context=_dispatch_async_setup_context
)
_combine_async_op.register_autograd(
    _combine_async_backward, setup_context=_combine_async_setup_context
)

# Communication and routing updates share native state without data edges, so
# keep them in program order. Ordering also keeps them out of rematerialization.
# Effects do not apply to the mutating *_out ops; a stale handle there raises.
for _op in (
    _create_routing_op,
    _update_routing_op,
    _dispatch_async_op,
    _combine_async_op,
    _dispatch_backward_async_op,
    _combine_backward_async_op,
):
    _op.register_effect(torch.library.EffectType.ORDERED)


def _validate_quantized_scales(tokens, scales):
    if scales.ndim != 2 or tokens.ndim != 2 or scales.shape[0] != tokens.shape[0]:
        raise ValueError("scales must be 2D with one row per token row")
    if scales.dtype not in (
        torch.uint8,
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        raise ValueError("unsupported scale dtype")
    if (
        not scales.is_contiguous()
        or scales.shape[1] == 0
        or (scales.shape[1] * scales.element_size()) % 16
    ):
        raise ValueError("contiguous scale rows must be nonempty and 16-byte aligned")


@torch.library.custom_op(
    "_nccl_ep::dispatch_quantized_async",
    mutates_args=(),
    schema=(
        "(torch.distributed._token_switch._NCCLEPObject handle, Tensor tokens, "
        "Tensor scales, Tensor topk_weights, SymInt capacity, str layout, "
        "*, bool initialize_tokens=True) -> "
        "(Tensor, Tensor, Tensor?, Tensor)"
    ),
)
def _dispatch_quantized_async_op(
    handle: _NCCLEPObject,
    tokens: torch.Tensor,
    scales: torch.Tensor,
    topk_weights: torch.Tensor,
    capacity: int,
    layout: str,
    *,
    initialize_tokens: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    outputs = _dispatch_quantized_async_fake(
        handle,
        tokens,
        scales,
        topk_weights,
        capacity,
        layout,
        initialize_tokens=initialize_tokens,
    )
    received, weights, ids, received_scales = outputs
    _import_nccl_ep()._nccl_ep_dispatch_quantized(
        handle.native(),
        tokens,
        scales,
        topk_weights,
        received,
        received_scales,
        weights,
        ids,
        async_op=True,
    )
    return outputs


@_dispatch_quantized_async_op.register_fake
def _dispatch_quantized_async_fake(
    handle, tokens, scales, topk_weights, capacity, layout, *, initialize_tokens=True
):
    _validate_quantized_scales(tokens, scales)
    received, weights, ids = _dispatch_async_fake(
        handle,
        tokens,
        topk_weights,
        capacity,
        layout,
        initialize_tokens=initialize_tokens,
    )
    return received, weights, ids, scales.new_zeros(capacity, scales.shape[1])


@torch.library.custom_op("_nccl_ep::combine_quantized_async", mutates_args=())
def _combine_quantized_async_op(
    handle: _NCCLEPObject,
    expert_tokens: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    output = _combine_quantized_async_fake(handle, expert_tokens, num_tokens)
    _import_nccl_ep()._nccl_ep_combine_quantized(
        handle.native(), expert_tokens, output, async_op=True
    )
    return output


@_combine_quantized_async_op.register_fake
def _combine_quantized_async_fake(handle, expert_tokens, num_tokens):
    if expert_tokens.ndim != 2 or expert_tokens.dtype != torch.bfloat16:
        raise ValueError("MXFP8 combine requires 2D BF16 tokens")
    if expert_tokens.shape[1] == 0 or expert_tokens.shape[1] % 512:
        raise ValueError("MXFP8 combine hidden size must be a positive multiple of 512")
    return expert_tokens.new_zeros(num_tokens, expert_tokens.shape[1])


@torch.library.custom_op("_nccl_ep::dispatch_backward_quantized_async", mutates_args=())
def _dispatch_backward_quantized_async_op(
    handle: _NCCLEPObject,
    grad_tokens: torch.Tensor,
    scales: torch.Tensor,
    capacity: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = _dispatch_backward_quantized_async_fake(
        handle, grad_tokens, scales, capacity
    )
    _import_nccl_ep()._nccl_ep_dispatch_backward_quantized(
        handle.native(), grad_tokens, scales, *outputs, async_op=True
    )
    return outputs


@_dispatch_backward_quantized_async_op.register_fake
def _dispatch_backward_quantized_async_fake(handle, grad_tokens, scales, capacity):
    # Reuse the scale contract without allocating or communicating score rows.
    _validate_quantized_scales(grad_tokens, scales)
    return (
        grad_tokens.new_zeros(capacity, grad_tokens.shape[1]),
        scales.new_zeros(capacity, scales.shape[1]),
    )


@torch.library.custom_op("_nccl_ep::combine_backward_quantized_async", mutates_args=())
def _combine_backward_quantized_async_op(
    handle: _NCCLEPObject,
    grad_tokens: torch.Tensor,
    grad_topk_weights: torch.Tensor,
    num_tokens: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = _combine_backward_quantized_async_fake(
        handle, grad_tokens, grad_topk_weights, num_tokens, top_k
    )
    _import_nccl_ep()._nccl_ep_combine_backward_quantized(
        handle.native(), grad_tokens, grad_topk_weights, *outputs, async_op=True
    )
    return outputs


@_combine_backward_quantized_async_op.register_fake
def _combine_backward_quantized_async_fake(
    handle, grad_tokens, grad_topk_weights, num_tokens, top_k
):
    output = _combine_quantized_async_fake(handle, grad_tokens, num_tokens)
    if grad_topk_weights.dtype != torch.float32:
        raise ValueError("routing-score gradients must use float32")
    return output, grad_topk_weights.new_zeros(num_tokens, top_k)


for _op in (
    _dispatch_quantized_async_op,
    _combine_quantized_async_op,
    _dispatch_backward_quantized_async_op,
    _combine_backward_quantized_async_op,
):
    _op.register_effect(torch.library.EffectType.ORDERED)


class _DispatchAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        ts: TokenSwitch,
        routing: Routing,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        max_recv_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        _N, H = tokens.shape
        K = topk_weights.shape[1]
        out_tokens, out_topk_weights, out_topk_idx = ts._alloc_dispatch_outputs(
            routing, tokens, topk_weights, max_recv_tokens, H, K
        )
        ts._dispatch(
            routing, tokens, topk_weights, out_tokens, out_topk_weights, out_topk_idx
        )
        ctx.ts = ts
        ctx.routing = routing
        ctx.needs_weight_grad = ctx.needs_input_grad[3]
        ctx.has_received_weights = out_topk_weights is not None
        return out_tokens, out_topk_weights, out_topk_idx

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(
        ctx: Any,
        grad_out_tokens: torch.Tensor,
        grad_out_topk_weights: torch.Tensor | None,
        grad_out_topk_idx: torch.Tensor | None,
    ) -> tuple[None, None, torch.Tensor, torch.Tensor | None, None]:
        if not ctx.has_received_weights or not ctx.needs_weight_grad:
            grad_tokens = grad_out_tokens.new_zeros(
                ctx.routing.topk_idx.shape[0], grad_out_tokens.shape[-1]
            )
            ctx.ts._combine(ctx.routing, grad_out_tokens.contiguous(), grad_tokens)
            return None, None, grad_tokens, None, None
        if grad_out_topk_weights is None:
            raise NotImplementedError(
                "TokenSwitch dispatch backward requires received topk_weights"
            )
        grad_tokens = grad_out_tokens.new_zeros(
            ctx.routing.topk_idx.shape[0], grad_out_tokens.shape[-1]
        )
        grad_weights = grad_out_topk_weights.new_zeros(ctx.routing.topk_idx.shape)
        ctx.ts._dispatch_backward(
            ctx.routing,
            grad_out_tokens.contiguous(),
            grad_out_topk_weights.contiguous(),
            grad_tokens,
            grad_weights,
        )
        return None, None, grad_tokens, grad_weights, None


class _CombineAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        ts: TokenSwitch,
        routing: Routing,
        expert_tokens: torch.Tensor,
    ) -> torch.Tensor:
        N = routing.topk_idx.shape[0]
        H = expert_tokens.shape[-1]
        out_tokens = expert_tokens.new_zeros(N, H)
        ts._combine(routing, expert_tokens, out_tokens)
        ctx.ts = ts
        ctx.routing = routing
        ctx.expert_shape = expert_tokens.shape
        ctx.expert_dtype = expert_tokens.dtype
        return out_tokens

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(
        ctx: Any, grad_out_tokens: torch.Tensor
    ) -> tuple[None, None, torch.Tensor]:
        ts = ctx.ts
        H = ctx.expert_shape[-1]
        grad_expert_full = grad_out_tokens.new_zeros(
            max(ts.max_recv_tokens_per_rank, ctx.expert_shape[0]),
            H,
            dtype=ctx.expert_dtype,
        )
        ts._combine_backward(
            ctx.routing,
            grad_out_tokens.to(ctx.expert_dtype).contiguous(),
            grad_expert_full,
        )
        return None, None, grad_expert_full[: ctx.expert_shape[0]].contiguous()


class TokenSwitch(abc.ABC):
    """Abstract token routing switch (e.g. expert-parallel dispatch / combine).

    Typical usage: :meth:`create_routing`, then :meth:`dispatch` / :meth:`combine`.

    Contract every backend implements:

    - A :class:`Routing` fixes the receive-slot layout. Every :meth:`dispatch` and
      :meth:`combine` on the same Routing uses that layout, so dispatching again
      (e.g. the backward of :meth:`combine`) writes the new tokens into the same
      slots. A backend may compute the layout in :meth:`create_routing` or in the
      first :meth:`dispatch` on the Routing.
    - Calls on different Routings may be interleaved in any order, e.g. two
      dispatches before either combine.
    - Flat layout: a token is received once per destination rank, however many of
      its top-k experts live there. Each ``out_topk_idx`` row holds the local ids of
      that rank's experts among the token's top-k, and -1 in its other entries;
      ``out_topk_weights`` holds the matching weights, and 0 where the id is -1.
      The order of the entries within a row is up to the backend.
    - Expert-major layout: one receive slot per (token, local expert).
    - :meth:`combine` sums, without weights, one row per receive slot that came
      from the token. Callers apply ``topk_weights`` and reduce over their local
      experts before combining. With an identity expert and the flat layout, a
      token comes back multiplied by its number of destination ranks.
    - Rows of the dispatch outputs past the number of received slots are
      unspecified.
    """

    @property
    @abc.abstractmethod
    def max_recv_tokens_per_rank(self) -> int:
        """Upper bound on receive slots per rank; sizes full dispatch outputs."""
        raise NotImplementedError

    @abc.abstractmethod
    def create_routing(
        self,
        topk_idx: torch.Tensor,
        per_expert_token_counts: torch.Tensor | None = None,
        *,
        layout: str,
    ) -> Routing:
        """Create expert routing for the current phase (e.g. top-k indices).

        ``per_expert_token_counts`` is optional 1D int32 with exactly
        ``num_experts // world_size`` elements: output buffer for per-expert
        receive counts (NCCL EP ``RECV_EXPERT_COUNTER``), i.e. the number of
        received (token, local expert) pairs per local expert.
        ``layout`` selects the dispatch output memory layout.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _alloc_dispatch_outputs(
        self,
        routing: Routing,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        max_recv_tokens: int,
        H: int,
        K: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Allocate dispatch output buffers with shapes matching routing.layout."""
        raise NotImplementedError

    @abc.abstractmethod
    def _dispatch(
        self,
        routing: Routing,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        out_tokens: torch.Tensor,
        out_topk_weights: torch.Tensor | None,
        out_topk_idx: torch.Tensor | None,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def _combine(
        self,
        routing: Routing,
        expert_tokens: torch.Tensor,
        out_tokens: torch.Tensor,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def _combine_backward(
        self,
        routing: Routing,
        grad_tokens: torch.Tensor,
        out_expert_grads: torch.Tensor,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def _dispatch_backward(
        self,
        routing: Routing,
        grad_recv_tokens: torch.Tensor,
        grad_recv_topk_weights: torch.Tensor,
        out_token_grads: torch.Tensor,
        out_topk_weight_grads: torch.Tensor,
    ) -> None:
        r"""Reverse dispatch into both source-shaped token and routing-weight buffers."""
        raise NotImplementedError

    def dispatch(
        self,
        routing: Routing,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        max_recv_tokens: int | None = None,
        *,
        out: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]
        | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        r"""Route tokens to experts.

        Returns ``(out_tokens, out_topk_weights, out_topk_idx)``.  For
        expert-major layouts ``out_topk_idx`` is ``None``; for LL+expert-major
        ``out_topk_weights`` is also ``None``.

        With ``out=(out_tokens, out_topk_weights, out_topk_idx)``: writes to the
        provided buffers and returns them; no autograd support.
        Without ``out``: allocates output buffers and returns the tuple with
        autograd support.  ``max_recv_tokens`` is required when ``out`` is not
        provided. Backends must implement gradients for both tokens and routing
        weights. NCCL supports first-order gradients for FP32 ``topk_weights``.
        """
        if out is not None:
            self._dispatch(routing, tokens, topk_weights, *out)
            return out
        if max_recv_tokens is None:
            raise ValueError("max_recv_tokens is required when out= is not provided")
        return _DispatchAutograd.apply(
            self, routing, tokens, topk_weights, max_recv_tokens
        )  # type: ignore[return-value]

    def combine(
        self,
        routing: Routing,
        expert_tokens: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Gather expert outputs back to token order.

        With ``out=out_tokens``: writes to the provided buffer and returns it;
        no autograd support.
        Without ``out``: allocates an output buffer and returns it with autograd support.
        """
        if out is not None:
            self._combine(routing, expert_tokens, out)
            return out
        return _CombineAutograd.apply(self, routing, expert_tokens)  # type: ignore[return-value]


class TokenSwitchNCCL(TokenSwitch):
    """Token switch backed by NCCL EP (:func:`ncclEpCreateGroup` / dispatch / combine).

    The dispatch output layout is chosen per routing via
    :meth:`create_routing`'s ``layout`` argument.
    """

    def __init__(
        self,
        process_group: ProcessGroup,
        num_experts: int,
        max_dispatch_tokens_per_rank: int,
        max_recv_tokens_per_rank: int,
        max_token_bytes: int,
        *,
        ht_flow: str = "auto",
    ) -> None:
        self._ep = _import_nccl_ep()
        ep = self._ep
        flow_map = {
            "auto": ep.HtFlow.AUTO,
            "push_pull": ep.HtFlow.PUSH_PULL,
            "pull_push": ep.HtFlow.PULL_PUSH,
        }
        if ht_flow not in flow_map:
            raise ValueError(
                f"ht_flow must be one of {list(flow_map)}; got {ht_flow!r}"
            )

        self._layout_map = {
            "flat": ep.Layout.FLAT,
            "expert_major": ep.Layout.EXPERT_MAJOR,
        }

        self._max_recv_tokens_per_rank = max_recv_tokens_per_rank
        self._group = ep._NcclEpGroup.create(
            process_group,
            num_experts,
            max_dispatch_tokens_per_rank,
            max_recv_tokens_per_rank,
            max_token_bytes,
            flow_map[ht_flow],
        )
        self._opaque_group = _NCCLEPObject(self._group)

    def close(self) -> None:
        """Release EP resources before destroying the process group.

        Tear down captured graphs first. Closing synchronizes the CUDA device,
        invalidates routing handles, and is idempotent.
        """
        self._group.close()

    @property
    def max_recv_tokens_per_rank(self) -> int:
        return self._max_recv_tokens_per_rank

    def _alloc_dispatch_outputs(
        self,
        routing: Routing,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        max_recv_tokens: int,
        H: int,
        K: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if routing.layout == "expert_major":
            # topk_weights is 1D (one scalar weight per recv slot);
            # topk_idx is not populated (nullptr in NCCL EP).
            return (
                tokens.new_zeros(max_recv_tokens, H),
                topk_weights.new_zeros(max_recv_tokens),
                None,
            )
        # flat: standard 2D outputs
        return (
            tokens.new_zeros(max_recv_tokens, H),
            topk_weights.new_zeros(max_recv_tokens, K),
            tokens.new_zeros(max_recv_tokens, K, dtype=torch.int64),
        )

    def create_routing(
        self,
        topk_idx: torch.Tensor,
        per_expert_token_counts: torch.Tensor | None = None,
        *,
        layout: str,
        alignment: int = 1,
    ) -> Routing:
        """Create expert routing for this phase; pass to :meth:`dispatch` / :meth:`combine`.

        ``layout`` (required) controls the dispatch output memory layout:
        ``"flat"`` or ``"expert_major"``.
        """
        if layout not in self._layout_map:
            raise ValueError(
                f"layout must be one of {list(self._layout_map)}; got {layout!r}"
            )
        handle, counts = _create_routing_op(
            self._opaque_group,
            topk_idx,
            per_expert_token_counts,
            layout,
            alignment,
        )
        if per_expert_token_counts is not None:
            per_expert_token_counts.copy_(counts)
        return Routing(handle=handle, topk_idx=topk_idx, layout=layout)

    def update_routing(
        self,
        routing: Routing,
        topk_idx: torch.Tensor,
        per_expert_token_counts: torch.Tensor | None = None,
    ) -> Routing:
        """Refresh preallocated routing with the same token and top-k shape.

        Create the routing before CUDA graph capture. Use the returned routing
        for dispatch/combine, and keep it and its input storage alive for replay.
        Do not update a handle while an earlier forward still needs its backward.
        """
        handle, counts = _update_routing_op(
            routing._opaque_handle, topk_idx, per_expert_token_counts
        )
        if per_expert_token_counts is not None:
            per_expert_token_counts.copy_(counts)
        return Routing(handle=handle, topk_idx=topk_idx, layout=routing.layout)

    def _dispatch(
        self,
        routing: Routing,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        out_tokens: torch.Tensor,
        out_topk_weights: torch.Tensor | None,
        out_topk_idx: torch.Tensor | None,
    ) -> None:
        capacity = self._max_recv_tokens_per_rank
        for tensor in (out_tokens, out_topk_weights, out_topk_idx):
            if tensor is not None and tensor.ndim > 0 and tensor.shape[0] < capacity:
                raise ValueError(
                    f"NCCL EP dispatch outputs require at least {capacity} rows "
                    f"(max_recv_tokens_per_rank), got {tensor.shape[0]}"
                )
        _dispatch_op(
            routing._opaque_handle,
            tokens,
            topk_weights,
            out_tokens,
            out_topk_weights,
            out_topk_idx,
        )

    def _combine(
        self,
        routing: Routing,
        expert_tokens: torch.Tensor,
        out_tokens: torch.Tensor,
    ) -> None:
        _combine_op(
            routing._opaque_handle,
            expert_tokens,
            out_tokens,
        )

    def _combine_backward(
        self,
        routing: Routing,
        grad_tokens: torch.Tensor,
        out_expert_grads: torch.Tensor,
    ) -> None:
        _dispatch_backward_op(routing._opaque_handle, grad_tokens, out_expert_grads)

    def _dispatch_backward(
        self,
        routing: Routing,
        grad_recv_tokens: torch.Tensor,
        grad_recv_topk_weights: torch.Tensor,
        out_token_grads: torch.Tensor,
        out_topk_weight_grads: torch.Tensor,
    ) -> None:
        _combine_backward_op(
            routing._opaque_handle,
            grad_recv_tokens,
            grad_recv_topk_weights,
            out_token_grads,
            out_topk_weight_grads,
        )

    def dispatch(
        self,
        routing: Routing,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        max_recv_tokens: int | None = None,
        *,
        out: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]
        | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if out is not None:
            return super().dispatch(routing, tokens, topk_weights, out=out)
        if max_recv_tokens is None:
            raise ValueError("max_recv_tokens is required when out= is not provided")
        if max_recv_tokens < self._max_recv_tokens_per_rank:
            raise ValueError(
                f"NCCL EP dispatch outputs require at least "
                f"{self._max_recv_tokens_per_rank} rows (max_recv_tokens_per_rank), "
                f"got {max_recv_tokens}"
            )
        from torch.distributed._functional_collectives import wait_tensor

        received, weights, ids = _dispatch_async_op(
            routing._opaque_handle,
            tokens,
            topk_weights,
            max_recv_tokens,
            routing.layout,
        )
        return (
            wait_tensor(received),
            wait_tensor(weights),
            wait_tensor(ids) if ids is not None else None,
        )

    def combine(
        self,
        routing: Routing,
        expert_tokens: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if out is not None:
            return super().combine(routing, expert_tokens, out=out)
        from torch.distributed._functional_collectives import wait_tensor

        return wait_tensor(
            _combine_async_op(
                routing._opaque_handle,
                expert_tokens,
                routing.topk_idx.shape[0],
                self._max_recv_tokens_per_rank,
            )
        )

    def dispatch_quantized(
        self,
        routing: Routing,
        tokens: torch.Tensor,
        scales: torch.Tensor,
        topk_weights: torch.Tensor,
        max_recv_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Forward physical token and scale rows without converting their values.

        Scales may contain an arbitrary aligned row width, including opaque
        sideband bytes. This raw transport has no autograd formula. A logical
        quantized training layer must define its own surrogate derivative and
        choose forward and backward transport independently.
        """
        if max_recv_tokens < self._max_recv_tokens_per_rank:
            raise ValueError(
                f"NCCL EP dispatch outputs require at least "
                f"{self._max_recv_tokens_per_rank} rows (max_recv_tokens_per_rank), "
                f"got {max_recv_tokens}"
            )
        from torch.distributed._functional_collectives import wait_tensor

        received, weights, ids, received_scales = _dispatch_quantized_async_op(
            routing._opaque_handle,
            tokens,
            scales,
            topk_weights,
            max_recv_tokens,
            routing.layout,
        )
        return (
            wait_tensor(received),
            wait_tensor(weights),
            wait_tensor(ids) if ids is not None else None,
            wait_tensor(received_scales),
        )

    def combine_quantized(
        self,
        routing: Routing,
        expert_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Combine BF16 rows using native MXFP8 communication.

        The native expert-major local-permute path quantizes internally and
        returns BF16. This raw operation has no implicit autograd formula.
        """
        from torch.distributed._functional_collectives import wait_tensor

        return wait_tensor(
            _combine_quantized_async_op(
                routing._opaque_handle,
                expert_tokens,
                routing.topk_idx.shape[0],
            )
        )

    def dispatch_backward_quantized(
        self,
        routing: Routing,
        grad_tokens: torch.Tensor,
        scales: torch.Tensor,
        max_recv_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Redispatch raw token-gradient and scale rows without score metadata."""
        if max_recv_tokens < self._max_recv_tokens_per_rank:
            raise ValueError(
                f"NCCL EP dispatch outputs require at least "
                f"{self._max_recv_tokens_per_rank} rows (max_recv_tokens_per_rank), "
                f"got {max_recv_tokens}"
            )
        from torch.distributed._functional_collectives import wait_tensor

        tokens, received_scales = _dispatch_backward_quantized_async_op(
            routing._opaque_handle, grad_tokens, scales, max_recv_tokens
        )
        return wait_tensor(tokens), wait_tensor(received_scales)

    def combine_backward_quantized(
        self,
        routing: Routing,
        grad_tokens: torch.Tensor,
        grad_topk_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return BF16 token gradients and FP32 scores in one MXFP8 combine."""
        from torch.distributed._functional_collectives import wait_tensor

        tokens, weights = _combine_backward_quantized_async_op(
            routing._opaque_handle,
            grad_tokens,
            grad_topk_weights,
            routing.topk_idx.shape[0],
            routing.topk_idx.shape[1],
        )
        return wait_tensor(tokens), wait_tensor(weights)
