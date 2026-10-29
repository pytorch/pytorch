"""Scoped native library-boundary adapters (pre-AOT, FX-level).

Two coherent adapters, each with its own rewrite entry point:

* BN: ``aten``/python training BatchNorm -> frozen-cuDNN boundary. Stock AOT
  decomposes ``cudnn_batch_norm -(Autograd py_impl)-> native_batch_norm
  -(Autograd py_impl)-> _native_batch_norm_legit`` into generated
  ``var_mean``/``rsqrt`` math (~1e-6 off), so decomp-table filtering alone
  cannot preserve; the boundary installs before AOT joint tracing.
* RNG: training dropout (``0 < p <= 1``) -> opaque boundary preserving exact
  eager mask/RNG order. The stock ``native_dropout`` decomposition
  (``rand_like`` math) re-enters philox lowering and diverges on multi-RNG
  graphs.

Contracts: frozen per-node ``BoundarySpec`` (+ ``BoundaryMutation`` for the
BN ``copy_`` glue), canonical op-identity matching (``OpOverload`` schema /
``OpOverloadPacket`` qualified names, never bare ``str()``), static
backend/reserve verification with loud rejection, ``copy_``-based running
stat updates so AOT functionalization materializes buffer mutations exactly
once at the framework seam.
"""

from __future__ import annotations

import dataclasses
import enum
import operator
from typing import Any

import torch
from torch import Tensor
from torch._ops import OpOverload, OpOverloadPacket

__all__ = [
    "BOUNDARY_MODULE_VERSION",
    "BoundaryKind",
    "BoundaryEffect",
    "BoundarySpec",
    "BoundaryMutation",
    "BoundaryReport",
    "KNOWN_NATIVE_BOUNDARY_SYMBOLS",
    "native_bn_fwd",
    "native_bn_bwd",
    "native_dropout_fwd",
    "canonical_target_name",
    "is_native_boundary_target",
    "boundary_kind_of",
    "select_batch_norm_backend",
    "batch_norm_reserve_size",
    "rewrite_native_boundaries",
    "rewrite_batch_norm_boundaries",
    "rewrite_dropout_boundaries",
    "BatchNormNativeBoundary",
    "DropoutNativeBoundary",
    "NativeBoundaryPreGradPass",
    "SUPPORTED_BN_DTYPES",
]

BOUNDARY_MODULE_VERSION = "0.2.0"

BN_FWD_NS = "loops_ir_native_bn"
DO_FWD_NS = "loops_ir_native_dropout"

_NativeTarget = OpOverload | OpOverloadPacket


def _resolve_target(qualified: str) -> _NativeTarget:
    # Packets materialize on attribute access after the @custom_op decorator
    # runs below; callers must resolve *after* module import completes (tests
    # call this function directly, never at module top level).
    ns, rest = qualified.split("::", 1)
    packet = getattr(getattr(torch.ops, ns), rest)
    if not isinstance(packet, OpOverloadPacket):
        raise RuntimeError(f"not a packet: {qualified}")
    return packet


def native_bn_fwd() -> _NativeTarget:
    return _resolve_target(f"{BN_FWD_NS}::bn_cudnn_fwd")


def native_bn_bwd() -> _NativeTarget:
    return _resolve_target(f"{BN_FWD_NS}::bn_cudnn_bwd")


def native_dropout_fwd() -> _NativeTarget:
    return _resolve_target(f"{DO_FWD_NS}::dropout_fwd")


KNOWN_NATIVE_BOUNDARY_SYMBOLS: frozenset[str] = frozenset(
    {
        f"{BN_FWD_NS}::bn_cudnn_fwd",
        f"{BN_FWD_NS}::bn_cudnn_fwd.default",
        f"{BN_FWD_NS}::bn_cudnn_bwd",
        f"{BN_FWD_NS}::bn_cudnn_bwd.default",
        f"{DO_FWD_NS}::dropout_fwd",
        f"{DO_FWD_NS}::dropout_fwd.default",
    }
)


class BoundaryKind(enum.Enum):
    BN_FUNCTIONAL = "bn_functional"
    RNG_STATE_UPDATE = "rng_state_update"


class BoundaryEffect(enum.Enum):
    BN_RUNNING_STATS_FUNCTIONAL = "bn_running_stats_functional"
    RNG_CONSUMES_ORDERED = "rng_consumes_ordered"
    PURE_BACKWARD = "pure_backward"


@dataclasses.dataclass(frozen=True)
class BoundarySpec:
    kind: BoundaryKind
    symbol: str
    version: str
    expected_impl: int = -1
    effect: BoundaryEffect = BoundaryEffect.PURE_BACKWARD


@dataclasses.dataclass(frozen=True)
class BoundaryMutation:
    """Typed record of the BN ``copy_`` glue (actual FX users, not meta)."""

    running_mean_target: str
    running_var_target: str


@dataclasses.dataclass(frozen=True)
class BoundaryReport:
    adapter: str
    version: str
    numerics: str
    rewrites: int
    rejections: tuple[str, ...]


class LoopsIrNativeBoundaryError(RuntimeError):
    """Scoped rejection (never a silent substitution)."""


SUPPORTED_BN_DTYPES: frozenset[torch.dtype] = frozenset({torch.float32})


def canonical_target_name(target: Any) -> str | None:
    """Canonical ``ns::name[.overload]`` for overloads/packets, else None."""
    if isinstance(target, OpOverload):
        return f"{target.namespace}::{target._opname}.{target._overloadname}"
    if isinstance(target, OpOverloadPacket):
        return target._qualified_op_name
    return None


def is_native_boundary_target(target: Any) -> bool:
    return canonical_target_name(target) in KNOWN_NATIVE_BOUNDARY_SYMBOLS


def boundary_kind_of(target: Any) -> BoundaryKind | None:
    name = canonical_target_name(target)
    if name in (
        f"{BN_FWD_NS}::bn_cudnn_fwd",
        f"{BN_FWD_NS}::bn_cudnn_fwd.default",
        f"{BN_FWD_NS}::bn_cudnn_bwd",
        f"{BN_FWD_NS}::bn_cudnn_bwd.default",
    ):
        return BoundaryKind.BN_FUNCTIONAL
    if name in (f"{DO_FWD_NS}::dropout_fwd", f"{DO_FWD_NS}::dropout_fwd.default"):
        return BoundaryKind.RNG_STATE_UPDATE
    return None


def select_batch_norm_backend(
    inp: Tensor,
    weight: Tensor,
    bias: Tensor,
    mean: Tensor,
    var: Tensor,
    training: bool,
    eps: float,
) -> Any:
    select = getattr(torch._C, "_select_batch_norm_backend", None)
    if select is None:
        raise LoopsIrNativeBoundaryError("_select_batch_norm_backend unavailable")
    try:
        return select(inp, weight, bias, mean, var, training, eps)
    except Exception as exc:
        raise LoopsIrNativeBoundaryError(f"backend selection failed: {exc}") from exc


def batch_norm_reserve_size(inp: Tensor, training: bool) -> int:
    fn = getattr(torch._C, "_get_cudnn_batch_norm_reserve_space_size", None)
    if fn is None:
        raise LoopsIrNativeBoundaryError("reserve-size query unavailable")
    return int(fn(inp, training))


def _backend_to_impl_index(backend: Any) -> int:
    name = getattr(backend, "name", None) or str(backend).rsplit(".", 1)[-1]
    if name == "Cudnn":
        return 1
    if name == "Miopen":
        return 2
    return 0


# ---------------------------------------------------------------------------
# Registered opaque symbols (bodies call only registered ATen symbols).
# ---------------------------------------------------------------------------


@torch.library.custom_op(f"{BN_FWD_NS}::bn_cudnn_fwd", mutates_args=())
def _bn_cudnn_fwd(
    x: Tensor,
    w: Tensor,
    b: Tensor,
    rm: Tensor,
    rv: Tensor,
    training: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
    cudnn_enabled: bool = True,
    expected_impl: int = 1,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Functional cuDNN BN: (out, save_mean, save_var, reserve, new_rm, new_rv).

    Clones running stats, dispatches ``aten._batch_norm_impl_index``, rejects
    runtime/frozen impl mismatch. The runtime index is not an output.
    """
    rmc = rm.clone()
    rvc = rv.clone()
    o, sm, sv, reserve, idx = torch.ops.aten._batch_norm_impl_index(
        x, w, b, rmc, rvc, training, momentum, eps, cudnn_enabled
    )
    if int(idx) != int(expected_impl):
        raise RuntimeError(
            f"bn boundary impl mismatch: runtime {int(idx)} != frozen {int(expected_impl)}"
        )
    return o, sm, sv, reserve, rmc, rvc


@torch.library.register_fake(f"{BN_FWD_NS}::bn_cudnn_fwd")
def _bn_cudnn_fwd_fake(
    x,
    w,
    b,
    rm,
    rv,
    training=True,
    momentum=0.1,
    eps=1e-5,
    cudnn_enabled=True,
    expected_impl=1,
):
    c = x.shape[1]
    return (
        torch.empty_like(x),
        torch.empty(c, dtype=torch.float32, device=x.device),
        torch.empty(c, dtype=torch.float32, device=x.device),
        torch.empty(0, dtype=torch.uint8, device=x.device),
        torch.empty_like(rm),
        torch.empty_like(rv),
    )


def _bn_setup_context(ctx: Any, inputs: Any, output: Any) -> None:
    x, w, b, rm, rv, training, momentum, eps, cudnn_enabled, expected_impl = inputs
    o, sm, sv, reserve, new_rm, new_rv = output
    ctx.save_for_backward(x, w, sm, sv, reserve)
    ctx.expected_impl = int(expected_impl)
    ctx.training = bool(training)
    ctx.eps = float(eps)


def _bn_backward(
    ctx: Any, g0: Tensor, g1: Tensor, g2: Tensor, g3: Tensor, g4: Tensor, g5: Tensor
) -> tuple[Any, ...]:
    x, w, sm, sv, reserve = ctx.saved_tensors
    gi, gw, gb = torch.ops.loops_ir_native_bn.bn_cudnn_bwd.default(
        x,
        g0.contiguous(),
        w,
        sm,
        sv,
        reserve,
        ctx.expected_impl,
        ctx.training,
        ctx.eps,
    )
    return gi, gw, gb, None, None, None, None, None, None, None


torch.library.register_autograd(
    f"{BN_FWD_NS}::bn_cudnn_fwd", _bn_backward, setup_context=_bn_setup_context
)


@torch.library.custom_op(f"{BN_FWD_NS}::bn_cudnn_bwd", mutates_args=())
def _bn_cudnn_bwd(
    x: Tensor,
    grad_out: Tensor,
    w: Tensor,
    save_mean: Tensor,
    save_var: Tensor,
    reserve: Tensor,
    expected_impl: int = 1,
    training: bool = True,
    eps: float = 1e-5,
) -> tuple[Tensor, Tensor, Tensor]:
    """Opaque BN backward over the frozen impl (cuDNN ``1`` proven here)."""
    impl = int(expected_impl)
    if impl == 1:
        return torch.ops.aten._batch_norm_impl_index_backward(
            impl,
            x,
            grad_out,
            w,
            None,
            None,
            save_mean,
            save_var,
            bool(training),
            float(eps),
            [True, True, True],
            reserve,
        )
    raise RuntimeError(f"bn boundary backward: unsupported impl {impl}")


@torch.library.register_fake(f"{BN_FWD_NS}::bn_cudnn_bwd")
def _bn_cudnn_bwd_fake(
    x,
    grad_out,
    w,
    save_mean,
    save_var,
    reserve,
    expected_impl=1,
    training=True,
    eps=1e-5,
):
    return torch.empty_like(x), torch.empty_like(w), torch.empty_like(w)


@torch.library.custom_op(
    f"{DO_FWD_NS}::dropout_fwd",
    mutates_args=(),
    tags=(torch.Tag.nondeterministic_seeded,),
)
def _dropout_fwd(
    x: Tensor, p: float = 0.5, training: bool = True
) -> tuple[Tensor, Tensor]:
    """Opaque training dropout: eager ``(out, bool_mask)``, one RNG consume."""
    if not bool(training) or not (0.0 < float(p) < 1.0):
        raise RuntimeError("dropout boundary only for training with 0 < p < 1")
    out, mask = torch.ops.aten.native_dropout(x, float(p), bool(training))
    return out, mask


@torch.library.register_fake(f"{DO_FWD_NS}::dropout_fwd")
def _dropout_fwd_fake(x, p=0.5, training=True):
    return torch.empty_like(x), torch.empty_like(x, dtype=torch.bool)


def _dropout_setup_context(ctx: Any, inputs: Any, output: Any) -> None:
    x, p, training = inputs
    out, mask = output
    ctx.save_for_backward(mask)
    pf = float(p)
    ctx.scale = 0.0 if pf == 1.0 else (1.0 / (1.0 - pf))
    ctx.training = bool(training)


def _dropout_backward(ctx: Any, g0: Tensor, g1: Tensor) -> tuple[Any, ...]:
    (mask,) = ctx.saved_tensors
    if not ctx.training:
        return g0, None, None
    out = torch.ops.aten.native_dropout_backward(g0, mask, float(ctx.scale))
    return out, None, None


torch.library.register_autograd(
    f"{DO_FWD_NS}::dropout_fwd",
    _dropout_backward,
    setup_context=_dropout_setup_context,
)


# ---------------------------------------------------------------------------
# FX rewrite (pre-AOT / pre-joint).
# ---------------------------------------------------------------------------

_PY_BATCH_NORM_FN = torch.nn.functional.batch_norm
_ATEN_BN_TARGETS: tuple[Any, ...] = (
    torch.ops.aten.batch_norm.default,
    torch.ops.aten.batch_norm,
)
_DROPOUT_TARGETS: tuple[Any, ...] = (
    torch.nn.functional.dropout,
    torch.ops.aten.dropout.default,
    torch.ops.aten.dropout,
    torch.ops.aten.native_dropout,
    torch.ops.aten.native_dropout.default,
)


def _frozen_numerics(numerics: str | None) -> str:
    if numerics is None:
        try:
            import torch._inductor.config as _cfg

            numerics = str(_cfg.experimental_cpp_ir_numerics)
        except Exception:
            numerics = "strict_eager"
    if numerics not in ("strict_eager", "inductor"):
        raise LoopsIrNativeBoundaryError(f"unknown numerics {numerics!r}")
    return numerics


def _node_fake(node: Any) -> Tensor | None:
    for key in ("example_value", "val"):
        ex = node.meta.get(key, None)
        fake = ex[0] if isinstance(ex, (tuple, list)) and ex else ex
        if isinstance(fake, torch.Tensor):
            return fake
    return None


def _static_bn_args(node: Any) -> tuple[Any, ...] | None:
    """Canonical (x, w, b, rm, rv, training, momentum, eps, cudnn_enabled).

    ``F.batch_norm`` order is (input, running_mean, running_var, weight,
    bias, training=False, momentum, eps); ``aten.batch_norm`` order is
    (input, weight, bias, running_mean, running_var, training, momentum,
    eps, cudnn_enabled). Both are normalized here; dynamic (non-bool/float)
    training/momentum/eps/cudnn_enabled values reject (None return).
    """
    args = tuple(node.args)
    kwargs = dict(node.kwargs)
    if node.target == _PY_BATCH_NORM_FN:
        pos = ["input", "running_mean", "running_var", "weight", "bias"]
        vals: dict[str, Any] = {}
        for i, nm in enumerate(pos):
            if i < len(args):
                vals[nm] = args[i]
            elif nm in kwargs:
                vals[nm] = kwargs[nm]
            elif nm in ("weight", "bias"):
                vals[nm] = None
            else:
                return None
        training = kwargs.get("training", args[5] if len(args) > 5 else False)
        momentum = kwargs.get("momentum", args[6] if len(args) > 6 else 0.1)
        eps = kwargs.get("eps", args[7] if len(args) > 7 else 1e-5)
        if (
            not isinstance(training, bool)
            or not isinstance(momentum, float)
            or not isinstance(eps, float)
        ):
            return None
        cudnn_enabled = kwargs.get("cudnn_enabled", torch.backends.cudnn.enabled)
        if not isinstance(cudnn_enabled, bool):
            return None
        return (
            vals["input"],
            vals["weight"],
            vals["bias"],
            vals["running_mean"],
            vals["running_var"],
            training,
            momentum,
            eps,
            cudnn_enabled,
        )
    if len(args) >= 9:
        x, w, b, rm, rv = args[0], args[1], args[2], args[3], args[4]
        training, momentum, eps, cudnn_enabled = args[5], args[6], args[7], args[8]
        if (
            not isinstance(training, bool)
            or not isinstance(momentum, float)
            or not isinstance(eps, float)
            or not isinstance(cudnn_enabled, bool)
        ):
            return None
        return (x, w, b, rm, rv, training, momentum, eps, cudnn_enabled)
    return None


class BatchNormNativeBoundary:
    """Scoped adapter: training BN -> frozen cuDNN boundary + ``copy_`` glue."""

    version = BOUNDARY_MODULE_VERSION + ":bn2"

    def __init__(self, numerics: str | None = None) -> None:
        self.numerics = _frozen_numerics(numerics)
        self.rewrites = 0
        self.rejections: list[str] = []

    def applies(self, node: Any) -> bool:
        return (
            node.op == "call_function"
            and node.target in (_PY_BATCH_NORM_FN,) + _ATEN_BN_TARGETS
        )

    def rewrite(self, gm: torch.fx.GraphModule, node: Any) -> bool:
        vals = _static_bn_args(node)
        if vals is None:
            self.rejections.append(f"{node.name}: unparsable/dynamic BN args")
            return False
        x, w, b, rm, rv, training, momentum, eps, cudnn_enabled = vals
        for name, t in (
            ("input", x),
            ("weight", w),
            ("bias", b),
            ("running_mean", rm),
            ("running_var", rv),
        ):
            if not isinstance(t, torch.fx.Node):
                self.rejections.append(f"{node.name}: {name} not a graph tensor")
                return False
        if not training:
            self.rejections.append(f"{node.name}: eval BN left to stock path")
            return False
        if not cudnn_enabled:
            self.rejections.append(f"{node.name}: cudnn_enabled=False cannot be impl 1")
            return False
        fake = _node_fake(node) if isinstance(x, torch.fx.Node) else None
        if fake is None:
            # Fall back to the input node's own metadata (the selector arg).
            fake = _node_fake(x)
        if not isinstance(fake, torch.Tensor):
            self.rejections.append(f"{node.name}: no static fake metadata")
            return False
        if fake.dtype not in SUPPORTED_BN_DTYPES or fake.device.type != "cuda":
            self.rejections.append(
                f"{node.name}: unsupported {fake.dtype}/{fake.device} (fp32 CUDA only)"
            )
            return False
        if len(fake.shape) < 3:
            self.rejections.append(f"{node.name}: input.dim < 3 (no cuDNN BN)")
            return False
        if not (
            fake.is_contiguous()
            or (
                len(fake.shape) == 4
                and fake.is_contiguous(memory_format=torch.channels_last)
            )
        ):
            # Reserve/size selection is proven for contiguous and channels-last
            # 4D only; anything else leaves the node for the stock path.
            self.rejections.append(f"{node.name}: unsupported memory format")
            return False
        # Static backend selection on the *input node's* metadata (proves the
        # frozen impl); per-tensor metadata required, never zero stubs.
        try:
            stats = []
            for t in (w, b, rm, rv):
                f = _node_fake(t)
                if not isinstance(f, torch.Tensor):
                    raise LoopsIrNativeBoundaryError("missing per-tensor metadata")
                stats.append(f)
            backend = select_batch_norm_backend(
                fake, stats[0], stats[1], stats[2], stats[3], True, eps
            )
        except LoopsIrNativeBoundaryError as exc:
            self.rejections.append(f"{node.name}: {exc}")
            return False
        impl = _backend_to_impl_index(backend)
        if impl != 1:
            self.rejections.append(
                f"{node.name}: backend {backend} (impl {impl}) is not proven cuDNN"
            )
            return False
        try:
            reserve = batch_norm_reserve_size(fake, True)
        except LoopsIrNativeBoundaryError as exc:
            self.rejections.append(f"{node.name}: {exc}")
            return False
        if reserve != 0:
            self.rejections.append(f"{node.name}: nonzero reserve {reserve}")
            return False
        # Sequential topological anchor: build the new chain with explicit
        # inserting_before(node) placements *before* erasing `node`, so the
        # chain lands exactly where `node` was (inserting_after chains can
        # invert when the anchor node is erased).
        with gm.graph.inserting_before(node):
            call = gm.graph.call_function(
                torch.ops.loops_ir_native_bn.bn_cudnn_fwd.default,
                (x, w, b, rm, rv, True, momentum, eps, True, int(impl)),
            )
            call.meta["loops_ir_boundary"] = BoundarySpec(
                kind=BoundaryKind.BN_FUNCTIONAL,
                symbol=f"{BN_FWD_NS}::bn_cudnn_fwd",
                version=self.version,
                expected_impl=int(impl),
                effect=BoundaryEffect.BN_RUNNING_STATS_FUNCTIONAL,
            )
            out0 = gm.graph.call_function(operator.getitem, (call, 0))
            new_rm = gm.graph.call_function(operator.getitem, (call, 4))
            new_rv = gm.graph.call_function(operator.getitem, (call, 5))
            # Buffer-update glue: explicit copy_ users preserve storage/version
            # semantics; AOT functionalization converts these into returned
            # mutation outputs exactly once at the framework seam. The copied
            # values are detached: running-stat updates are by definition not
            # differentiable (eager raises on rm/rv grad), and an attached
            # edge makes the one-shot compiled backward unreusable across
            # steps (saved-tensor double-free on the second backward).
            det_rm = gm.graph.call_function(torch.ops.aten.detach.default, (new_rm,))
            det_rv = gm.graph.call_function(torch.ops.aten.detach.default, (new_rv,))
            gm.graph.call_function(torch.ops.aten.copy_.default, (rm, det_rm))
            gm.graph.call_function(torch.ops.aten.copy_.default, (rv, det_rv))
        call.meta["loops_ir_mutation"] = BoundaryMutation(
            running_mean_target=str(rm), running_var_target=str(rv)
        )
        node.replace_all_uses_with(out0)
        gm.graph.erase_node(node)
        self.rewrites += 1
        return True


class DropoutNativeBoundary:
    """Scoped adapter: training dropout (``0 < p < 1``) -> opaque boundary."""

    version = BOUNDARY_MODULE_VERSION + ":do3"

    def __init__(self, numerics: str | None = None) -> None:
        self.numerics = _frozen_numerics(numerics)
        self.rewrites = 0
        self.rejections: list[str] = []

    def applies(self, node: Any) -> bool:
        # Never match our own boundary symbol (idempotency: a second pass
        # must not treat dropout_fwd as another dropout site and duplicate
        # the RNG consume).
        if is_native_boundary_target(node.target):
            return False
        return node.op == "call_function" and node.target in _DROPOUT_TARGETS

    def _parse(self, node: Any) -> tuple[Any, float, bool] | None:
        args = tuple(node.args)
        kwargs = dict(node.kwargs)
        # Non-default Generator kwarg: unmodeled RNG source; never silently
        # rewrite (the boundary always consumes the global CUDA RNG).
        if kwargs.get("generator") is not None:
            return None
        if not args:
            return None
        if node.target == torch.nn.functional.dropout:
            if kwargs.get("inplace", args[3] if len(args) > 3 else False):
                return None  # input mutation: never silently rewritten
            training = kwargs.get("training", args[2] if len(args) > 2 else True)
        else:
            training = kwargs.get(
                "train", kwargs.get("training", args[2] if len(args) > 2 else True)
            )
            if training is None and node.target in (
                torch.ops.aten.native_dropout,
                torch.ops.aten.native_dropout.default,
            ):
                training = True
        probability = kwargs.get("p", args[1] if len(args) > 1 else 0.5)
        if not isinstance(probability, (int, float)) or not isinstance(training, bool):
            return None
        return args[0], float(probability), training

    def rewrite(self, gm: torch.fx.GraphModule, node: Any) -> bool:
        parsed = self._parse(node)
        if parsed is None:
            self.rejections.append(f"{node.name}: unparsable/inplace dropout args")
            return False
        x, p, training = parsed
        if not training or p == 0.0 or p == 1.0:
            # Identity (p == 0 / eval) or p == 1 edge-value profiles: leave the
            # node (no alias lie, no silent sign/NaN resemanticization).
            if p == 1.0 and training:
                self.rejections.append(f"{node.name}: p==1 edge semantics unproven")
            return False
        if not (0.0 < p < 1.0):
            self.rejections.append(f"{node.name}: bad p={p}")
            return False
        # Each original site consumes the CUDA RNG exactly once.
        with gm.graph.inserting_before(node):
            call = gm.graph.call_function(
                torch.ops.loops_ir_native_dropout.dropout_fwd.default,
                (x, float(p), bool(training)),
            )
            call.meta["loops_ir_boundary"] = BoundarySpec(
                kind=BoundaryKind.RNG_STATE_UPDATE,
                symbol=f"{DO_FWD_NS}::dropout_fwd",
                version=self.version,
                effect=BoundaryEffect.RNG_CONSUMES_ORDERED,
            )
            if node.target in (
                torch.ops.aten.native_dropout,
                torch.ops.aten.native_dropout.default,
            ):
                getitem_users = [
                    u for u in list(node.users) if u.target == operator.getitem
                ]
                for u in getitem_users:
                    u.args = (call,) + tuple(u.args[1:])
                node.replace_all_uses_with(call)
                gm.graph.erase_node(node)
            else:
                node.replace_all_uses_with(
                    gm.graph.call_function(operator.getitem, (call, 0))
                )
                gm.graph.erase_node(node)
        self.rewrites += 1
        return True


def rewrite_batch_norm_boundaries(
    gm: torch.fx.GraphModule, numerics: str | None = None
) -> BoundaryReport:
    adapter = BatchNormNativeBoundary(numerics=numerics)
    for node in list(gm.graph.nodes):
        if adapter.applies(node):
            adapter.rewrite(gm, node)
    gm.graph.lint()
    gm.recompile()
    return BoundaryReport(
        adapter="bn",
        version=BatchNormNativeBoundary.version,
        numerics=adapter.numerics,
        rewrites=adapter.rewrites,
        rejections=tuple(adapter.rejections),
    )


def rewrite_dropout_boundaries(
    gm: torch.fx.GraphModule, numerics: str | None = None
) -> BoundaryReport:
    adapter = DropoutNativeBoundary(numerics=numerics)
    for node in list(gm.graph.nodes):
        if adapter.applies(node):
            adapter.rewrite(gm, node)
    gm.graph.lint()
    gm.recompile()
    return BoundaryReport(
        adapter="dropout",
        version=DropoutNativeBoundary.version,
        numerics=adapter.numerics,
        rewrites=adapter.rewrites,
        rejections=tuple(adapter.rejections),
    )


def rewrite_native_boundaries(
    gm: torch.fx.GraphModule, numerics: str | None = None
) -> tuple[BoundaryReport, BoundaryReport]:
    """BN then RNG (BN is RNG-pure; no seed/offset nodes are introduced)."""
    return (
        rewrite_batch_norm_boundaries(gm, numerics=numerics),
        rewrite_dropout_boundaries(gm, numerics=numerics),
    )


class NativeBoundaryPreGradPass:
    """``pre_grad_custom_pass`` body (stable method uuid)."""

    def __init__(self, numerics: str | None = None) -> None:
        self.numerics = _frozen_numerics(numerics)

    def __call__(self, graph: torch.fx.graph.Graph) -> None:
        gm = graph.owning_module
        rewrite_native_boundaries(gm, numerics=self.numerics)

    def uuid(self) -> str:
        return f"loops-ir-native-boundaries:{BOUNDARY_MODULE_VERSION}:{self.numerics}"
