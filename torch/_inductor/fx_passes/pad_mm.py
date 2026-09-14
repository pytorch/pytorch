import functools
import itertools
import operator
import typing
from contextvars import ContextVar
from dataclasses import dataclass
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch._inductor.runtime.runtime_utils
from torch import Tensor
from torch._dynamo.utils import counters
from torch._higher_order_ops.flex_gemm import _PRESERVE_FLEX_GEMM_GEMM_OP
from torch._inductor import utils
from torch._inductor.autoheuristic.autoheuristic import (
    AHContext,
    AutoHeuristic,
    LocalFeedback,
)
from torch._inductor.autoheuristic.autoheuristic_utils import (
    context_add_strides,
    context_add_using_tf32,
    pad_mm_operations,
    pad_mm_precondition,
)
from torch._inductor.runtime.caching import encoders, memoizers
from torch._subclasses.fake_tensor import is_fake_tensor
from torch.fx.experimental.symbolic_shapes import statically_known_true
from torch.utils._mode_utils import no_dispatch

from ...utils._triton import has_triton
from ..pattern_matcher import (
    fwd_only,
    gen_register_replacement,
    joint_fwd_bwd,
    Match,
    ReplaceFn,
    SearchFn,
)


aten = torch.ops.aten


# This flag is only used for testing purpose.
# Changing it to True will ignore comparing do_bench times
# between original pattern and padded one.
_skip_do_bench_times = False


@dataclass(frozen=True)
class PaddingPlan:
    """The dimensions padded by one shape-padding replacement."""

    pad_m: bool = False
    pad_k: bool = False
    pad_n: bool = False

    @property
    def name(self) -> str:
        if self.pad_m:
            return "legacy-all"
        if self.pad_k and self.pad_n:
            return "k+n"
        if self.pad_k:
            return "k"
        if self.pad_n:
            return "n"
        return "none"


NO_PADDING = PaddingPlan()
K_PADDING = PaddingPlan(pad_k=True)
N_PADDING = PaddingPlan(pad_n=True)
K_N_PADDING = PaddingPlan(pad_k=True, pad_n=True)
LEGACY_ALL_PADDING = PaddingPlan(pad_m=True, pad_k=True, pad_n=True)
FORCE_PADDING = LEGACY_ALL_PADDING

_PADDING_PLANS_BY_NAME = {
    plan.name: plan
    for plan in (NO_PADDING, K_PADDING, N_PADDING, K_N_PADDING, LEGACY_ALL_PADDING)
}

# Replacement graphs are retraced immediately after their extra_check succeeds.
# Keep the exact selected plan in compilation-local state so the replacement does
# not independently reconstruct a different plan.
_selected_padding_plan: ContextVar[PaddingPlan | None] = ContextVar(
    "selected_padding_plan", default=None
)


def _consume_selected_padding_plan() -> PaddingPlan:
    """Return one accepted plan and clear it before tracing graph operations."""
    plan = _selected_padding_plan.get()
    _selected_padding_plan.set(None)
    if plan is None:
        raise AssertionError("padding replacement traced without a selected plan")
    return plan


def _clear_selected_padding_plan() -> None:
    _selected_padding_plan.set(None)


def _padding_plan_result_encoder_factory(
    fn: Callable[..., PaddingPlan],
) -> Callable[..., Callable[[PaddingPlan], str]]:
    del fn

    def params_to_encoder(
        *args: object, **kwargs: object
    ) -> Callable[[PaddingPlan], str]:
        del args, kwargs
        return lambda plan: plan.name

    return params_to_encoder


def _padding_plan_result_decoder_factory(
    fn: Callable[..., PaddingPlan],
) -> Callable[..., Callable[[object], PaddingPlan]]:
    del fn

    def params_to_decoder(
        *args: object, **kwargs: object
    ) -> Callable[[object], PaddingPlan]:
        del args, kwargs

        def decode(value: object) -> PaddingPlan:
            # Unknown values and pre-v3 boolean results fail closed.
            if not isinstance(value, str):
                return NO_PADDING
            return _PADDING_PLANS_BY_NAME.get(value, NO_PADDING)

        return decode

    return params_to_decoder


def fetch_fake_tensors(match: Match, kwarg_names: Sequence[str]) -> list[Tensor]:
    kwargs = match.kwargs
    return [kwargs[name].meta["val"] for name in kwarg_names]


def unwrap_fake_args(
    *arg_names: str,
) -> Callable[[Callable[..., Any]], Callable[[Match], Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[[Match], Any]:
        def wrapper(match: Match) -> Any:
            fake_tensors = fetch_fake_tensors(match, arg_names)
            return func(*fake_tensors)

        return wrapper

    return decorator


def get_alignment_size(x: Tensor) -> int:
    return get_alignment_size_dtype(x.dtype)


def get_alignment_size_dtype(dtype: torch.dtype) -> int:
    if dtype == torch.float16 or dtype == torch.half or dtype == torch.bfloat16:
        return 8
    elif dtype == torch.float32 or dtype == torch.float:
        return 4
    else:
        return 0


def check_device(a: Tensor, b: Tensor) -> bool:
    return (a.is_cuda and b.is_cuda) or (a.is_xpu and b.is_xpu)


def check_dtype(a: Tensor, b: Tensor) -> bool:
    return a.is_floating_point() and b.is_floating_point()


def hint_symbols(
    ds: Sequence[int | torch.SymInt],
) -> list[int]:
    """Helper to convert symbolic dimensions to their concrete hint values."""
    from torch.fx.experimental.symbolic_shapes import optimization_hint

    return [optimization_hint(d) for d in ds]


def can_pad(
    mat1: Tensor,
    mat2: Tensor,
    op: torch._ops.OpOverloadPacket,
    input: Tensor | None = None,
) -> bool:
    """
    Determines if an operation CAN be padded (safety checks).
    All logic related to whether it's safe to pad should be here.
    """

    # Can't pad if there is no static dims, we pad static dims only.
    def has_one_static_dim(t: Tensor) -> bool:
        """Return False if all dimensions are symbolic — nothing concrete to pad."""
        for x in t.size():
            if isinstance(x, int):
                return True
            elif not isinstance(x, torch.SymInt):
                raise RuntimeError("not expected size")
        return False

    # Basic safety checks
    if not torch._inductor.config.shape_padding:
        return False

    if not check_device(mat1, mat2):
        return False

    if not check_dtype(mat1, mat2):
        return False

    # For padding to be vaible each tensor should have at least one static dim.
    tensors = [t for t in (mat1, mat2, input) if t is not None]
    if not all(has_one_static_dim(t) for t in tensors):
        return False

    # Skip zero-sized dimensions — padding would be wasteful (mm on empty tensors)
    from torch.fx.experimental.symbolic_shapes import optimization_hint

    if any(
        optimization_hint(dim) == 0 for dim in itertools.chain(mat1.shape, mat2.shape)
    ):
        return False

    # Calculate padding lengths to check if padding is needed
    with no_dispatch():
        if op is torch.ops.aten.mm or op is torch.ops.aten.addmm:
            m = mat1.shape[0]
            k = mat1.shape[1]
            n = mat2.shape[1]
        elif op is torch.ops.aten.bmm:
            m = mat1.shape[1]
            k = mat1.shape[2]
            n = mat2.shape[2]
        else:
            return False

        k_padded_length = get_padded_length(k, get_alignment_size(mat1))
        n_padded_length = get_padded_length(n, get_alignment_size(mat2))
        m_padded_length = get_padded_length(m, get_alignment_size(mat1))

        # No padding needed - can't pad if there's nothing to pad
        if m_padded_length == k_padded_length == n_padded_length == 0:
            return False

    # In deterministic mode, we can't safely benchmark - disallow padding
    # Check this after other basic checks so force_shape_pad/autoheuristic can override
    if (
        torch._inductor.config.deterministic
        and not torch._inductor.config.force_shape_pad
        and not torch._inductor.config.use_autoheuristic("pad_mm")
    ):
        return False

    # Triton availability check - required for padding to work
    if not has_triton():
        return False

    return True


def get_padded_length(x: int | torch.SymInt, alignment_size: int) -> int:
    # we don't pad x if it is symbolic
    if isinstance(x, torch.SymInt) or alignment_size == 0 or x % alignment_size == 0:
        return 0

    # ignore dim that can be squeezed away
    if x == 1:
        return 0

    return int((x // alignment_size + 1) * alignment_size) - x


def get_padding_lengths(
    mat1: Tensor,
    mat2: Tensor,
    op: torch._ops.OpOverloadPacket,
    plan: PaddingPlan,
) -> tuple[int, int, int]:
    """Return (M, K, N) padding for exactly ``plan``."""
    if op is torch.ops.aten.bmm:
        m, k, n = mat1.shape[1], mat1.shape[2], mat2.shape[2]
    else:
        m, k, n = mat1.shape[0], mat1.shape[1], mat2.shape[1]
    return (
        get_padded_length(m, get_alignment_size(mat1)) if plan.pad_m else 0,
        get_padded_length(k, get_alignment_size(mat1)) if plan.pad_k else 0,
        get_padded_length(n, get_alignment_size(mat2)) if plan.pad_n else 0,
    )


def get_normal_padding_plans(
    mat1: Tensor,
    mat2: Tensor,
    op: torch._ops.OpOverloadPacket,
) -> tuple[PaddingPlan, ...]:
    """Return non-M subsets plus the one legacy all-required-dim plan."""
    m_pad, k_pad, n_pad = get_padding_lengths(
        mat1, mat2, op, LEGACY_ALL_PADDING
    )
    plans: list[PaddingPlan] = [NO_PADDING]
    if k_pad:
        plans.append(K_PADDING)
    if n_pad:
        plans.append(N_PADDING)
    if k_pad and n_pad:
        plans.append(K_N_PADDING)
    if m_pad and (k_pad or n_pad):
        # Keep the old combined candidate without introducing M-only subsets.
        # Appending it makes exact ties prefer the less invasive non-M plan.
        plans.append(LEGACY_ALL_PADDING)
    return tuple(plans)


def get_full_non_m_padding_plan(plans: Sequence[PaddingPlan]) -> PaddingPlan:
    if K_N_PADDING in plans:
        return K_N_PADDING
    if K_PADDING in plans:
        return K_PADDING
    if N_PADDING in plans:
        return N_PADDING
    return NO_PADDING


def pad_dim(x: Tensor, padded_length: int, dim: int) -> Tensor:
    if padded_length == 0:
        return x
    pad = x.new_zeros(*x.shape[:dim], padded_length, *x.shape[dim + 1 :])
    return torch.cat([x, pad], dim=dim)


def addmm_pattern(
    input: Tensor, mat1: Tensor, mat2: Tensor, beta: float, alpha: float
) -> Tensor:
    return aten.addmm(input, mat1, mat2, beta=beta, alpha=alpha)


def _is_statically_expandable_to(shape: torch.Size, desired: Sequence[Any]) -> bool:
    if len(shape) > len(desired):
        return False
    return all(
        statically_known_true(dim == desired_dim) or statically_known_true(dim == 1)
        for dim, desired_dim in zip(reversed(shape), reversed(desired))
    )


def should_pad_addmm(match: Match) -> bool:
    _clear_selected_padding_plan()
    mat1, mat2, input = fetch_fake_tensors(match, ("mat1", "mat2", "input"))
    beta = match.kwargs["beta"]
    if (
        beta == 0
        and input.is_cuda
        and not _is_statically_expandable_to(
            input.shape, (mat1.shape[0], mat2.shape[1])
        )
    ):
        return False
    return should_pad(match, mat1, mat2, torch.ops.aten.addmm, input=input)


def pad_addmm(
    input: Tensor | None,
    mat1: Tensor,
    mat2: Tensor,
    m_padded_length: int,
    k_padded_length: int,
    n_padded_length: int,
    beta: float = 1.0,
    alpha: float = 1.0,
    mat1_pre_padded: bool = False,
    mat2_pre_padded: bool = False,
) -> Tensor:
    # for paddings, dim order is reversed for some reasons
    # and for every dim, we need to specify left and right padding
    if not mat1_pre_padded:
        mat1 = pad_mat1(
            mat1, m_padded_length=m_padded_length, k_padded_length=k_padded_length
        )
    if not mat2_pre_padded:
        mat2 = pad_mat2(
            mat2, k_padded_length=k_padded_length, n_padded_length=n_padded_length
        )

    # the add broadcasts, so we only pad if the dimension != 1
    if input is not None:
        if n_padded_length != 0:
            if input.dim() == 2 and input.shape[1] != 1:
                input = pad_dim(input, n_padded_length, 1)
            elif input.dim() == 1 and input.shape[0] != 1:
                input = pad_dim(input, n_padded_length, 0)
        if m_padded_length != 0 and input.dim() == 2 and input.shape[0] != 1:
            input = pad_dim(input, m_padded_length, 0)

    res = aten.addmm(input, mat1, mat2, beta=beta, alpha=alpha)

    if m_padded_length != 0:
        res = res[:-m_padded_length, :]
    if n_padded_length != 0:
        res = res[:, :-n_padded_length]
    return res


def addmm_replace(
    input: Tensor | None,
    mat1: Tensor,
    mat2: Tensor,
    beta: float = 1.0,
    alpha: float = 1.0,
) -> Tensor:
    m_padded_length, k_padded_length, n_padded_length = get_padding_lengths(
        mat1, mat2, torch.ops.aten.addmm, _consume_selected_padding_plan()
    )
    return pad_addmm(
        input,
        mat1,
        mat2,
        m_padded_length,
        k_padded_length,
        n_padded_length,
        beta,
        alpha,
    )


def is_mm_compute_bound(M: int, K: int, N: int, dtype: torch.dtype) -> bool:
    denominator = M * K + N * K + M * N
    if denominator == 0:
        return False
    arithmetic_intensity = (M * N * K) / denominator

    # we have experienced some large perf hits in this case, even in bandwidth bound regimes
    if (
        dtype is torch.bfloat16
        and K > M
        and K > N
        and (torch.xpu.is_available() or torch.cuda.get_device_capability() < (9, 0))
    ):  # doesn't repro on h100s:
        return True

    # Fails with AMD
    try:
        machine_balance = (
            1000 * utils.get_device_tflops(dtype)
        ) / utils.get_gpu_dram_gbps()
    except Exception:
        return True

    # dram_gbps might be underestimating bandwidth because of cache.
    # if we estimate machine balance too low we might miss some speedups,
    # if we estimate too high there will be unnecessary compilation time increase.
    # TODO - finetune coefficient here. As a reference point, Triton mm model assumes
    # 80% of reads are in cache and cache is 4x faster than dram_gbps
    machine_balance = machine_balance * 0.5

    return arithmetic_intensity > machine_balance


@functools.cache
def get_pad_cache() -> torch._inductor.codecache.LocalCache:
    return torch._inductor.codecache.LocalCache()


def get_cached_padding_plan(key: str) -> PaddingPlan | None:
    value = get_pad_cache().lookup(key)
    if not isinstance(value, str):
        return None
    return {
        "none": NO_PADDING,
        "k": K_PADDING,
        "n": N_PADDING,
        "k+n": K_N_PADDING,
        "legacy-all": LEGACY_ALL_PADDING,
    }.get(value)


def set_cached_padding_plan(key: str, plan: PaddingPlan) -> None:
    get_pad_cache().set_value(key, value=plan.name)


def get_cached_base_mm_benchmark_time(key: str) -> float:
    return get_pad_cache().lookup(key)  # type: ignore[return-value]


def set_cached_base_mm_benchmark_time(key: str, value: float) -> None:
    return get_pad_cache().set_value(key, value=value)


def padding_selection_policy() -> tuple[object, float, bool, bool]:
    """Configuration values that can change the selected padding plan."""
    options = torch._inductor.config.post_grad_fusion_options
    return (
        options.get("pad_aten_mm_pass"),
        options.get("shape_padding_multiplier", {}).get("value", 1.1),
        _should_run_pad_autoheuristic(),
        torch._inductor.config.deterministic,
    )


def should_pad_bench_key(
    match: Match,
    mat1: Tensor,
    mat2: Tensor,
    op: torch._ops.OpOverloadPacket,
    input: Tensor | None = None,
    is_base_time_key: bool = False,
) -> str:
    def tensor_key(t: Tensor) -> tuple[object, ...]:
        return (
            t.shape,
            t.stride(),
            t.dtype,
            encoders.get_device_identity(t.device),
        )

    fp32_precision = encoders.get_matmul_precision_for_cache(mat1)
    addmm_scalars = (
        (match.kwargs.get("beta", 1.0), match.kwargs.get("alpha", 1.0))
        if op is torch.ops.aten.addmm
        else None
    )

    def fmt_pad(name: str) -> str | None:
        if is_base_time_key:
            return None
        return f"exclude_pad:{should_exclude_padding_time(match, name)}"

    key = (
        tensor_key(mat1),
        tensor_key(mat2),
        fmt_pad("mat1"),
        fmt_pad("mat2"),
        op,
        input if input is None else tensor_key(input),
        addmm_scalars,
        fp32_precision,
        None if is_base_time_key else padding_selection_policy(),
    )

    if not is_base_time_key:
        key = ("padding_plan_v3", *key)

    key = str(key)
    if is_base_time_key:
        key = f"base mm time: {key}"
    return key


def get_non_view_def(node: torch.fx.Node) -> torch.fx.Node:
    if node.op == "call_function" and node.target is operator.getitem:
        return get_non_view_def(node.args[0])  # type: ignore[arg-type]

    if (
        node.op == "call_function"
        and isinstance(node.target, torch._ops.OpOverload)
        and utils.is_view(node.target)
    ):
        return get_non_view_def(node.all_input_nodes[0])

    return node


def should_exclude_padding_time(match: Match, arg_name: str) -> bool:
    from torch._prims_common import is_contiguous_or_false

    node_def = get_non_view_def(match.kwargs[arg_name])

    # constant padding converts tensors to contiguous so even if the input tensor
    # can be planned layout transform is not free. TODO - way to pad and preserve layout ?
    # Use is_contiguous_or_false to avoid guarding on data-dependent expressions
    # with unbacked symints - returns False instead of raising an error.
    if not is_contiguous_or_false(fetch_fake_tensors(match, (arg_name,))[0]):
        return False

    # TODO - see issue https://github.com/pytorch/pytorch/issues/128889
    # We would only able to completely plan these out if we were only doing
    # first dimension padding. non-first we would still need a copy
    # because these outputs are fixed dense.
    cannot_plan_output = [
        aten.mm.default,
        aten.convolution.default,
        aten.convolution_backward.default,
        aten.bmm.default,
        aten.addmm.default,
        aten._scaled_dot_product_flash_attention.default,
        aten._scaled_dot_product_efficient_attention.default,
    ]

    if node_def.target in cannot_plan_output:
        return False

    if (
        node_def.target is aten.cat.default
        and len(node_def.all_input_nodes)
        > torch._inductor.config.max_pointwise_cat_inputs
    ):
        return False

    # optimistically assume we should be able to memory plan away
    # all non inputs
    return node_def.op != "placeholder"


def is_padded_faster(key: str, ori_time: float, pad_time: float) -> bool:
    """
    Determines if padding is beneficial by comparing benchmark times.
    Helper function that applies a multiplier to account for memory ops overhead.
    """
    multiplier = 1.1
    # Shape padding introduces additional memory ops. Based on microbenchmarks, 1.1x represents a reasonable
    # tradeoff between performance improvement from shape padding and overhead from additional memory ops
    # TODO: Build a learned model which would be better than this heuristic
    if "shape_padding_multiplier" in torch._inductor.config.post_grad_fusion_options:
        multiplier = torch._inductor.config.post_grad_fusion_options[
            "shape_padding_multiplier"
        ].get("value", 1.1)
        counters["inductor"]["shape_padding_multiplier"] += 1
    return _skip_do_bench_times or ori_time > pad_time * multiplier


def should_pad_mm_bf16(dtype: torch.dtype, M: int, N: int, K: int) -> bool:
    # always force pad for mm with bf16 when the following are satisfied to avoid perf regression
    large_k_threshold_to_pad = torch._inductor.config.post_grad_fusion_options[
        "pad_aten_mm_pass"
    ].get("k_threshold_to_pad", 8388608)
    if (
        dtype is torch.bfloat16
        and K > M
        and K > N
        and N % 2 == 1
        and K >= large_k_threshold_to_pad
        and (torch.xpu.is_available() or torch.cuda.get_device_capability() < (9, 0))
    ):  # doesn't repro on h100s:
        return True
    return False


def should_pad(
    match: Match,
    mat1: Tensor,
    mat2: Tensor,
    op: torch._ops.OpOverloadPacket,
    input: Tensor | None = None,
) -> bool:
    # A prior replacement trace may have failed before consuming its handoff.
    # Clear at the start so every rejection and exception fails closed.
    _clear_selected_padding_plan()
    if match.output_node().meta.get(_PRESERVE_FLEX_GEMM_GEMM_OP):
        return False
    if not can_pad(mat1, mat2, op, input):
        return False

    if torch._inductor.config.force_shape_pad:
        _selected_padding_plan.set(FORCE_PADDING)
        return True

    # Small-K/N mm is lowered to a fused pointwise kernel in tuned_mm.
    # Leave those shapes unpadded and let the pointwise lowering handle them.
    if op is torch.ops.aten.mm:
        from ..kernel.mm_common import _use_small_mm_pointwise

        m, k, n = mat1.shape[0], mat1.shape[1], mat2.shape[1]
        if _use_small_mm_pointwise(
            m, k, n, mat1.device.type, statically_known_true=statically_known_true
        ):
            return False

    plan = _should_pad(match, mat1, mat2, op, input)
    if plan == NO_PADDING:
        return False
    _selected_padding_plan.set(plan)
    return True


def get_do_bench() -> Callable[[Callable[[], Any]], float]:
    return functools.partial(
        # pyrefly: ignore [bad-argument-type]
        torch._inductor.runtime.benchmarking.benchmarker.benchmark_gpu,
        warmup=5,
    )


def _should_run_pad_autoheuristic() -> bool:
    return torch._inductor.config.run_autoheuristic("pad_mm")


def _realize_tensor(t: Tensor) -> Tensor:
    if is_fake_tensor(t):
        size_hints = hint_symbols(t.size())
        stride_hint = hint_symbols(t.stride())
        real_size = sum((d - 1) * s for d, s in zip(size_hints, stride_hint)) + 1
        real_t = torch.randn(real_size, dtype=t.dtype, device=t.device)
        return torch.as_strided(real_t, size_hints, stride_hint)
    return torch.randn_like(t)


def _padding_bench_fn(
    match: Match,
    mat1: Tensor,
    mat2: Tensor,
    op: torch._ops.OpOverloadPacket,
    plan: PaddingPlan,
    input: Tensor | None,
) -> Callable[[], Any]:
    m_pad, k_pad, n_pad = get_padding_lengths(mat1, mat2, op, plan)
    is_bmm = op is torch.ops.aten.bmm
    mat1_pre_padded = should_exclude_padding_time(match, "mat1")
    mat2_pre_padded = should_exclude_padding_time(match, "mat2")
    mat1_pad, mat2_pad = mat1, mat2
    prepare: list[Callable[[], Any]] = []
    beta, alpha = (
        (match.kwargs.get("beta", 1.0), match.kwargs.get("alpha", 1.0))
        if op is torch.ops.aten.addmm
        else (1.0, 1.0)
    )

    if mat1_pre_padded and (m_pad or k_pad):
        mat1_pad = pad_mat1(
            mat1, m_padded_length=m_pad, k_padded_length=k_pad, is_bmm=is_bmm
        )
        if m_pad:
            prepare.append(
                lambda: mat1_pad[:, -m_pad:, :].zero_()
                if is_bmm
                else mat1_pad[-m_pad:, :].zero_()
            )
        if k_pad:
            prepare.append(
                lambda: mat1_pad[:, :, -k_pad:].zero_()
                if is_bmm
                else mat1_pad[:, -k_pad:].zero_()
            )

    if mat2_pre_padded and (k_pad or n_pad):
        mat2_pad = pad_mat2(
            mat2, k_padded_length=k_pad, n_padded_length=n_pad, is_bmm=is_bmm
        )
        if k_pad:
            prepare.append(
                lambda: mat2_pad[:, -k_pad:, :].zero_()
                if is_bmm
                else mat2_pad[-k_pad:, :].zero_()
            )
        if n_pad:
            prepare.append(
                lambda: mat2_pad[:, :, -n_pad:].zero_()
                if is_bmm
                else mat2_pad[:, -n_pad:].zero_()
            )

    def run() -> Any:
        for fn in prepare:
            fn()
        if op is torch.ops.aten.mm:
            return pad_mm(
                mat1_pad,
                mat2_pad,
                m_pad,
                k_pad,
                n_pad,
                mat1_pre_padded=mat1_pre_padded,
                mat2_pre_padded=mat2_pre_padded,
            )
        if op is torch.ops.aten.bmm:
            return pad_bmm(
                mat1_pad,
                mat2_pad,
                m_pad,
                k_pad,
                n_pad,
                mat1_pre_padded=mat1_pre_padded,
                mat2_pre_padded=mat2_pre_padded,
            )
        return pad_addmm(
            input,
            mat1_pad,
            mat2_pad,
            m_pad,
            k_pad,
            n_pad,
            beta=beta,
            alpha=alpha,
            mat1_pre_padded=mat1_pre_padded,
            mat2_pre_padded=mat2_pre_padded,
        )

    return run


def _select_padding_plan_uncached(
    match: Match,
    mat1: Tensor,
    mat2: Tensor,
    op: torch._ops.OpOverloadPacket,
    input: Tensor | None = None,
) -> PaddingPlan:
    """Choose the fastest profitable normal-mode padding plan."""
    with no_dispatch():
        if op is torch.ops.aten.mm or op is torch.ops.aten.addmm:
            m = mat1.shape[0]
            k = mat1.shape[1]
            n = mat2.shape[1]
        elif op is torch.ops.aten.bmm:
            m = mat1.shape[1]
            k = mat1.shape[2]
            n = mat2.shape[2]
        else:
            return NO_PADDING

        # Resolve symbolic dims to concrete hints for heuristic checks below.
        # These are performance decisions, not correctness — optimization_hint is safe.
        m_concrete, k_concrete, n_concrete = hint_symbols((m, k, n))

        plans = get_normal_padding_plans(mat1, mat2, op)
        if len(plans) == 1:
            return NO_PADDING

        if (
            "pad_aten_mm_pass" in torch._inductor.config.post_grad_fusion_options
            and should_pad_mm_bf16(mat1.dtype, m_concrete, n_concrete, k_concrete)
        ):
            return (
                LEGACY_ALL_PADDING
                if LEGACY_ALL_PADDING in plans
                else get_full_non_m_padding_plan(plans)
            )

        # Check if operation is compute bound (performance check)
        if not is_mm_compute_bound(m_concrete, k_concrete, n_concrete, mat1.dtype):
            return NO_PADDING

        key = should_pad_bench_key(match, mat1, mat2, op, input)
        cached_plan = get_cached_padding_plan(key)
        if cached_plan is not None:
            return cached_plan

        mat1 = _realize_tensor(mat1)
        mat2 = _realize_tensor(mat2)

        # since we key on whether or not the inputs can be memory planned, set cache for the
        # original time which is unaffected by whether or not the input can be planned
        ori_time_key = should_pad_bench_key(
            match, mat1, mat2, op, input, is_base_time_key=True
        )
        ori_time = get_cached_base_mm_benchmark_time(ori_time_key)
        if op is torch.ops.aten.addmm and input is not None:
            input = _realize_tensor(input)

        mat1_pre_padded = should_exclude_padding_time(match, "mat1")
        mat2_pre_padded = should_exclude_padding_time(match, "mat2")
        do_bench = get_do_bench()

        def orig_bench_fn():
            if op is torch.ops.aten.bmm or op is torch.ops.aten.mm:
                return op(mat1, mat2)
            return op(
                input,
                mat1,
                mat2,
                beta=match.kwargs.get("beta", 1.0),
                alpha=match.kwargs.get("alpha", 1.0),
            )

        padded_fns = {
            plan: _padding_bench_fn(match, mat1, mat2, op, plan, input)
            for plan in plans[1:]
        }

        autoheuristic_plan: PaddingPlan | None = None
        if _should_run_pad_autoheuristic() and op is torch.ops.aten.mm:
            if len(padded_fns) == 1:
                autoheuristic_plan = next(iter(padded_fns))
            elif torch._inductor.config.deterministic:
                autoheuristic_plan = get_full_non_m_padding_plan(plans)

        if autoheuristic_plan is not None:
            pad_bench_fn = padded_fns[autoheuristic_plan]
            plan_m, plan_k, plan_n = get_padding_lengths(
                mat1, mat2, op, autoheuristic_plan
            )
            ah_should_pad = run_autoheuristic(
                mat1,
                mat2,
                orig_bench_fn,
                pad_bench_fn,
                plan_m,
                plan_k,
                plan_n,
                do_bench,
                mat1_pre_padded,
                mat2_pre_padded,
                ori_time,
                ori_time_key,
                key,
            )
            if ah_should_pad is not None:
                selected_plan = (
                    autoheuristic_plan if ah_should_pad else NO_PADDING
                )
                set_cached_padding_plan(key, selected_plan)
                return selected_plan

        # AH didn't make a decision, so if we're in deterministic mode, we should return false
        if torch._inductor.config.deterministic:
            return NO_PADDING

        if ori_time is None:
            ori_time = do_bench(orig_bench_fn)
            set_cached_base_mm_benchmark_time(ori_time_key, ori_time)

        plan_times = {plan: do_bench(fn) for plan, fn in padded_fns.items()}
        counters["inductor"]["pad_mm_bench"] += 1
        best_plan, best_time = min(plan_times.items(), key=operator.itemgetter(1))
        selected_plan = (
            best_plan if is_padded_faster(key, ori_time, best_time) else NO_PADDING
        )
        set_cached_padding_plan(key, selected_plan)
        return selected_plan


@memoizers.should_pad_memoizer.memoize(
    custom_params_encoder=encoders.should_pad_params_encoder,
    custom_result_encoder=_padding_plan_result_encoder_factory,
    custom_result_decoder=_padding_plan_result_decoder_factory,
)
def _should_pad(
    match: Match,
    mat1: Tensor,
    mat2: Tensor,
    op: torch._ops.OpOverloadPacket,
    input: Tensor | None = None,
) -> PaddingPlan:
    return _select_padding_plan_uncached(match, mat1, mat2, op, input)


def get_context(
    mat1: Tensor,
    mat2: Tensor,
    mat1_pre_padded: bool,
    mat2_pre_padded: bool,
    m_padded_length: int,
    k_padded_length: int,
    n_padded_length: int,
) -> AHContext:
    context = AHContext()

    context.add_feature("m", mat1.shape[0])
    context.add_feature("k", mat1.shape[1])
    context.add_feature("n", mat2.shape[1])

    context_add_strides(context, "mat1", mat1.stride())
    context_add_strides(context, "mat2", mat2.stride())

    context.add_feature("m_padded_length", m_padded_length)
    context.add_feature("k_padded_length", k_padded_length)
    context.add_feature("n_padded_length", n_padded_length)

    context.add_feature("mat1_align_size", get_alignment_size(mat1))
    context.add_feature("mat2_align_size", get_alignment_size(mat2))

    context.add_feature("mat1_dtype", mat1.dtype, is_categorical=True)
    context.add_feature("mat2_dtype", mat2.dtype, is_categorical=True)

    context.add_feature("prepadded_mat1", mat1_pre_padded, is_categorical=True)
    context.add_feature("prepadded_mat2", mat2_pre_padded, is_categorical=True)

    context_add_using_tf32(context, mat1.dtype)
    return context


def run_autoheuristic(
    mat1: Tensor,
    mat2: Tensor,
    orig_bench_fn: Callable[[], None],
    pad_bench_fn: Callable[[], None],
    m_padded_length: int,
    k_padded_length: int,
    n_padded_length: int,
    do_bench: Callable[[Callable[[], Any]], float],
    mat1_pre_padded: bool,
    mat2_pre_padded: bool,
    ori_time: float,
    ori_time_key: str,
    key: str,
) -> bool | None:
    def feedback_fn(
        choice: str,
    ) -> float | None:
        if choice == orig_choice:
            return do_bench(orig_bench_fn)
        elif choice == pad_choice:
            return do_bench(pad_bench_fn)
        return None

    def fallback() -> str:
        return "autotune"

    orig_choice = "orig"
    pad_choice = "pad"
    choices = [orig_choice, pad_choice]
    feedback = LocalFeedback(feedback_fn)  # type: ignore[arg-type]
    context = get_context(
        mat1,
        mat2,
        mat1_pre_padded,
        mat2_pre_padded,
        m_padded_length,
        k_padded_length,
        n_padded_length,
    )
    name = "pad_mm"
    autoheuristic = AutoHeuristic(
        fallback=fallback,
        choices=choices,
        feedback=feedback,
        context=context,
        name=name,
        augment_context=pad_mm_operations(),
        precondition=pad_mm_precondition,
    )
    choice = autoheuristic.get_choice()
    choice2should_pad = {orig_choice: False, pad_choice: True, "autotune": None}
    ah_should_pad = choice2should_pad.get(choice)

    if torch._inductor.config.collect_autoheuristic(name):
        ah_ori_time = autoheuristic.get_collected_feedback(orig_choice)
        ah_pad_time = autoheuristic.get_collected_feedback(pad_choice)

        # if precondition is not satisfied, autoheuristic does not collect data
        if ah_ori_time is not None and ah_pad_time is not None:
            if ori_time is None:
                set_cached_base_mm_benchmark_time(ori_time_key, ah_ori_time)
            return is_padded_faster(key, ah_ori_time, ah_pad_time)
    return ah_should_pad


def mm_pattern(mat1: Tensor, mat2: Tensor) -> Tensor:
    return aten.mm(mat1, mat2)


def should_pad_mm(match: Match) -> bool:
    _clear_selected_padding_plan()
    mat1, mat2 = fetch_fake_tensors(match, ("mat1", "mat2"))
    return should_pad(match, mat1, mat2, torch.ops.aten.mm)


def pad_mat1(
    mat1: Tensor, *, m_padded_length: int, k_padded_length: int, is_bmm: bool = False
) -> Tensor:
    if k_padded_length != 0 or m_padded_length != 0:
        # dim order is reversed for constant_pad_nd, for every dim we specify right and left padding
        pad_arg = [0, k_padded_length, 0, m_padded_length]
        if is_bmm:
            pad_arg.extend((0, 0))
        return aten.constant_pad_nd(mat1, pad_arg)
    else:
        return mat1


def pad_mat2(
    mat2: Tensor, *, k_padded_length: int, n_padded_length: int, is_bmm: bool = False
) -> Tensor:
    if k_padded_length != 0 or n_padded_length != 0:
        # dim order is reversed for constant_pad_nd, for every dim we specify right and left padding
        pad_arg = [0, n_padded_length, 0, k_padded_length]
        if is_bmm:
            pad_arg.extend((0, 0))
        return aten.constant_pad_nd(mat2, pad_arg)
    else:
        return mat2


def pad_mm(
    mat1: Tensor,
    mat2: Tensor,
    m_padded_length: int,
    k_padded_length: int,
    n_padded_length: int,
    mat1_pre_padded: bool = False,
    mat2_pre_padded: bool = False,
) -> Tensor:
    if not mat1_pre_padded:
        mat1 = pad_mat1(
            mat1, m_padded_length=m_padded_length, k_padded_length=k_padded_length
        )
    if not mat2_pre_padded:
        mat2 = pad_mat2(
            mat2, k_padded_length=k_padded_length, n_padded_length=n_padded_length
        )
    res = aten.mm(mat1, mat2)
    if m_padded_length != 0:
        res = res[:-m_padded_length, :]
    if n_padded_length != 0:
        res = res[:, :-n_padded_length]
    return res


def mm_replace(mat1: Tensor, mat2: Tensor) -> Tensor:
    m_padded_length, k_padded_length, n_padded_length = get_padding_lengths(
        mat1, mat2, torch.ops.aten.mm, _consume_selected_padding_plan()
    )
    return pad_mm(
        mat1,
        mat2,
        m_padded_length,
        k_padded_length,
        n_padded_length,
    )


def bmm_pattern(mat1: Tensor, mat2: Tensor) -> Tensor:
    return aten.bmm(mat1, mat2)


def should_pad_bmm(match: Match) -> bool:
    _clear_selected_padding_plan()
    mat1, mat2 = fetch_fake_tensors(match, ("mat1", "mat2"))
    return should_pad(match, mat1, mat2, torch.ops.aten.bmm)


def pad_bmm(
    mat1: Tensor,
    mat2: Tensor,
    m_padded_length: int,
    k_padded_length: int,
    n_padded_length: int,
    mat1_pre_padded: bool = False,
    mat2_pre_padded: bool = False,
) -> Tensor:
    if not mat1_pre_padded:
        mat1 = pad_mat1(
            mat1,
            m_padded_length=m_padded_length,
            k_padded_length=k_padded_length,
            is_bmm=True,
        )
    if not mat2_pre_padded:
        mat2 = pad_mat2(
            mat2,
            k_padded_length=k_padded_length,
            n_padded_length=n_padded_length,
            is_bmm=True,
        )
    res = aten.bmm(mat1, mat2)
    if m_padded_length != 0:
        res = res[:, :-m_padded_length, :]
    if n_padded_length != 0:
        res = res[:, :, :-n_padded_length]
    return res


def bmm_replace(mat1: Tensor, mat2: Tensor) -> Tensor:
    m_padded_length, k_padded_length, n_padded_length = get_padding_lengths(
        mat1, mat2, torch.ops.aten.bmm, _consume_selected_padding_plan()
    )
    return pad_bmm(
        mat1,
        mat2,
        m_padded_length,
        k_padded_length,
        n_padded_length,
    )


@functools.cache
def _pad_mm_init(input_device: torch.device | None = None) -> None:
    from .joint_graph import patterns

    if input_device:
        device = str(input_device)
    else:
        if torch.cuda.is_available():
            # workaround https://github.com/pytorch/pytorch/issues/97894
            device = "cuda"
        elif torch.xpu.is_available():
            device = "xpu"
        else:
            device = "cpu"

    # sizes/values don't actually matter for initial trace
    # once we get a possible match we re-trace with the actual values and verify the match still holds

    dim2a = functools.partial(torch.empty, (4, 4), device=device, requires_grad=True)
    dim2b = functools.partial(torch.empty, (4, 4), device=device, requires_grad=True)

    dim3a = functools.partial(torch.empty, (4, 4, 4), device=device, requires_grad=True)
    dim3b = functools.partial(torch.empty, (4, 4, 4), device=device, requires_grad=True)

    dim1a = functools.partial(torch.empty, (4), device=device, requires_grad=True)

    # workaround https://github.com/pytorch/pytorch/issues/97894
    # 0.113377 is a "magic" value that lets us recover the lost input arg relationship
    rep = {"beta": 0.213377, "alpha": 0.113377}

    for pattern, replacement, args, workaround, extra_check in [
        (
            typing.cast(SearchFn, mm_pattern),
            typing.cast(ReplaceFn, mm_replace),
            [dim2a(), dim2b()],
            {},
            should_pad_mm,
        ),
        (
            typing.cast(SearchFn, bmm_pattern),
            typing.cast(ReplaceFn, bmm_replace),
            [dim3a(), dim3b()],
            {},
            should_pad_bmm,
        ),
        (
            typing.cast(SearchFn, addmm_pattern),
            typing.cast(ReplaceFn, addmm_replace),
            [dim1a(), dim2a(), dim2b()],
            rep,
            should_pad_addmm,
        ),
    ]:
        if not isinstance(
            workaround, dict
        ):  # mypy is unable to infer the type properly
            raise AssertionError(
                f"expected workaround to be a dict, got {type(workaround)}"
            )
        name = pattern.__name__

        gen_register_replacement(
            f"{name}_training",
            pattern,
            replacement,
            args,
            # pyrefly: ignore [bad-argument-type]
            joint_fwd_bwd,
            # pyrefly: ignore [bad-argument-type]
            patterns,
            extra_check=extra_check,
            scalar_workaround=workaround,
            skip_duplicates=True,
        )

        gen_register_replacement(
            f"{name}_inference",
            pattern,
            replacement,
            args,
            # pyrefly: ignore [bad-argument-type]
            fwd_only,
            # pyrefly: ignore [bad-argument-type]
            patterns,
            extra_check=extra_check,
            scalar_workaround=workaround,
            skip_duplicates=True,
        )
