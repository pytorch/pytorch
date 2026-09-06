import statistics
import threading

import torch
from torch import Tensor

import cutlass.cute as cute
from cutlass import Float32

from .cache import jit_cache
from .compile_utils import make_fake_tensor as fake_tensor
from .cute_dsl_utils import (
    get_device_capacity,
    get_max_active_clusters,
    torch2cute_dtype_map,
)
from .gemm_default_epi import GemmDefaultEpiMixin
from .gemm_act import GemmActMixin
from .gemm_sm100 import GemmSm100
from .gemm_tvm_ffi_utils import (
    compile_gemm_kernel,
    div_for_dtype,
    get_major,
    make_fake_scheduler_args,
    make_scheduler_args,
)
from .tile_scheduler import TriangularTileScheduler


class GemmSymmetricMixin(GemmActMixin):
    EpilogueArguments = GemmActMixin.EpilogueArguments

    def get_scheduler_class(self, varlen_m: bool = False):
        return TriangularTileScheduler

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        GemmDefaultEpiMixin.epi_visit_subtile(
            self, params, epi_loop_tensors, tRS_rD, tRS_rC
        )
        return tRS_rD


class GemmSymmetricSm100(GemmSymmetricMixin, GemmSm100):
    pass


_AUTOTUNE_CACHE: dict[
    tuple[torch.device, torch.dtype, int, int, int, bool],
    tuple[tuple[int, int], tuple[int, int, int], int],
] = {}
_AUTOTUNE_LOCK = threading.Lock()


@jit_cache
def _compile_gemm_symmetric(
    a_dtype,
    d_dtype,
    a_major,
    d_major,
    postact_major,
    tile_shape_mn,
    cluster_shape_mnk,
    device_capacity,
    has_c,
):
    m, k, l = cute.sym_int(), cute.sym_int(), cute.sym_int()
    div_a = div_for_dtype(a_dtype)
    div_d = div_for_dtype(d_dtype)
    mA = fake_tensor(
        a_dtype,
        (m, k, l),
        leading_dim=1 if a_major == "k" else 0,
        divisibility=div_a,
    )
    mB = fake_tensor(
        a_dtype,
        (m, k, l),
        leading_dim=1 if a_major == "k" else 0,
        divisibility=div_a,
    )
    mD = fake_tensor(
        d_dtype,
        (m, m, l),
        leading_dim=1 if d_major == "n" else 0,
        divisibility=div_d,
    )
    mC = (
        fake_tensor(
            d_dtype,
            (m, m, l),
            leading_dim=1 if d_major == "n" else 0,
            divisibility=div_d,
        )
        if has_c
        else None
    )
    mAuxOut = fake_tensor(
        d_dtype,
        (m, m, l),
        leading_dim=1 if postact_major == "n" else 0,
        divisibility=div_d,
    )
    epi_args = GemmSymmetricMixin.EpilogueArguments(
        mAuxOut,
        alpha=Float32(0.0) if has_c else None,
        beta=Float32(0.0) if has_c else None,
    )
    scheduler_args = make_fake_scheduler_args(False, False, l)
    return compile_gemm_kernel(
        GemmSymmetricSm100,
        a_dtype,
        tile_shape_mn,
        cluster_shape_mnk,
        False,
        True,
        False,
        True,
        device_capacity,
        mA,
        mB,
        mD,
        mC,
        epi_args,
        scheduler_args,
        None,
    )


def gemm_symmetric(
    A: Tensor,
    out: Tensor,
    C: Tensor | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    *,
    tile_shape_mn: tuple[int, int] = (256, 256),
    cluster_shape_mnk: tuple[int, int, int] = (2, 1, 1),
    scheduler_group_size: int = 8,
    autotune: bool = False,
) -> None:
    """Compute a symmetric Gram matrix, optionally with a symmetric C epilogue.

    Inputs are contiguous `(M, K)` tensors or homogeneous `(L, M, K)` batches.
    When C is supplied, it must be symmetric and alpha and beta must also be
    supplied; the result is `alpha * A @ A.mT + beta * C`.
    """
    if A.ndim not in (2, 3) or out.ndim != A.ndim:
        raise NotImplementedError(
            "vendored symmetric GEMM currently supports 2-D tensors or homogeneous batches"
        )
    if A.dtype not in (torch.float16, torch.bfloat16) or out.dtype != A.dtype:
        raise NotImplementedError("vendored symmetric GEMM requires matching fp16 or bf16 tensors")
    if not A.is_contiguous() or not out.is_contiguous():
        raise ValueError("symmetric GEMM requires contiguous input and output tensors")
    if out.device != A.device:
        raise ValueError("symmetric GEMM input and output must be on the same device")
    if not (
        (C is None and alpha is None and beta is None)
        or (C is not None and alpha is not None and beta is not None)
    ):
        raise ValueError("symmetric GEMM requires C, alpha, and beta together")
    device_capacity = get_device_capacity(A.device)
    if device_capacity[0] not in (10, 11):
        raise NotImplementedError("vendored symmetric GEMM prototype requires SM100 or SM110")

    if A.ndim == 2:
        m, k = A.shape
        batch = 1
        batch_stride = 0
        out_batch_stride = 0
    else:
        batch, m, k = A.shape
        batch_stride = A.stride(0)
        out_batch_stride = out.stride(0)
    if out.shape != (*A.shape[:-2], m, m):
        raise ValueError("symmetric GEMM output has an invalid shape")
    if C is not None and (
        C.shape != out.shape
        or C.dtype != out.dtype
        or C.device != out.device
        or not C.is_contiguous()
    ):
        raise ValueError("symmetric GEMM C tensor has incompatible metadata")
    if autotune and (C is None or C.data_ptr() != out.data_ptr()):
        key = (A.device, A.dtype, m, k, batch, C is not None)
        config = _AUTOTUNE_CACHE.get(key)
        if config is None:
            with torch.cuda.device(A.device):
                capturing = torch.cuda.is_current_stream_capturing()
            if not capturing:
                with _AUTOTUNE_LOCK:
                    config = _AUTOTUNE_CACHE.get(key)
                    if config is None:
                        config = _autotune_gemm_symmetric(A, out, C, alpha, beta)
                        _AUTOTUNE_CACHE[key] = config
        if config is not None:
            tile_shape_mn, cluster_shape_mnk, scheduler_group_size = config
    A_p = A.as_strided((m, k, batch), (A.stride(-2), A.stride(-1), batch_stride))
    D_p = out.as_strided(
        (m, m, batch), (out.stride(-2), out.stride(-1), out_batch_stride)
    )
    PostAct_p = out.as_strided(
        (m, m, batch), (out.stride(-1), out.stride(-2), out_batch_stride)
    )
    C_p = (
        C.as_strided(
            (m, m, batch),
            (C.stride(-2), C.stride(-1), C.stride(0) if C.ndim == 3 else 0),
        )
        if C is not None
        else None
    )
    a_major = get_major(A_p, "m", "k")
    d_major = get_major(D_p, "m", "n")
    postact_major = get_major(PostAct_p, "m", "n")
    a_dtype = torch2cute_dtype_map[A.dtype]
    d_dtype = torch2cute_dtype_map[out.dtype]
    compiled_fn = _compile_gemm_symmetric(
        a_dtype,
        d_dtype,
        a_major,
        d_major,
        postact_major,
        tile_shape_mn,
        cluster_shape_mnk,
        device_capacity,
        C is not None,
    )

    from .cache import is_compile_only

    if is_compile_only():
        return
    epi_args = GemmSymmetricMixin.EpilogueArguments(
        PostAct_p,
        act_fn=None,
        tensor_epilogue_fn=None,
        tensor_epilogue_arg_kinds=None,
        tensor_epilogue_returns_aux=None,
        tensor_epilogue_returns_local_reduce=None,
        local_reduce_feeds_main=None,
        local_reduce_group=None,
        local_reduce_axis=None,
        local_reduce_combine_fn=None,
        local_reduce_finalize_fn=None,
        alpha=Float32(alpha) if alpha is not None else None,
        beta=Float32(beta) if beta is not None else None,
        rounding_mode=None,
    )
    device_index = A.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    with torch.cuda.device(A.device):
        scheduler_args = make_scheduler_args(
            get_max_active_clusters(
                cluster_shape_mnk[0] * cluster_shape_mnk[1],
                device_capacity=device_capacity,
                device_id=device_index,
            ),
            scheduler_group_size,
            None,
        )
        compiled_fn(
            A_p,
            A_p,
            D_p,
            C_p,
            epi_args,
            scheduler_args,
            None,
            None,
            None,
            None,
        )


def _autotune_gemm_symmetric(
    A: Tensor,
    out: Tensor,
    C: Tensor | None,
    alpha: float | None,
    beta: float | None,
) -> tuple[tuple[int, int], tuple[int, int, int], int]:
    if A.shape[-2] >= 4096:
        candidates = tuple(
            ((256, 256), (2, 1, 1), group_size)
            for group_size in (2, 4, 8, 16, 32)
        )
    else:
        candidates = (
            ((256, 256), (2, 1, 1), 8),
            ((128, 128), (2, 1, 1), 8),
            ((128, 128), (1, 1, 1), 8),
        )
    samples = {candidate: [] for candidate in candidates}
    with torch.cuda.device(A.device):
        for _ in range(25):
            gemm_symmetric(A, out, C, alpha, beta)
        for tile, cluster, group_size in candidates:
            for _ in range(5):
                gemm_symmetric(
                    A,
                    out,
                    C,
                    alpha,
                    beta,
                    tile_shape_mn=tile,
                    cluster_shape_mnk=cluster,
                    scheduler_group_size=group_size,
                )
        for round_index in range(8):
            ordered = candidates if round_index % 2 == 0 else candidates[::-1]
            for tile, cluster, group_size in ordered:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(5):
                    gemm_symmetric(
                        A,
                        out,
                        C,
                        alpha,
                        beta,
                        tile_shape_mn=tile,
                        cluster_shape_mnk=cluster,
                        scheduler_group_size=group_size,
                    )
                end.record()
                end.synchronize()
                samples[(tile, cluster, group_size)].append(start.elapsed_time(end))
    timings = [
        (statistics.median(values), *candidate)
        for candidate, values in samples.items()
    ]
    best = min(timings)
    default = next(timing for timing in timings if timing[-1] == 8)
    selected = default
    if default[0] > 1.05 * best[0]:
        confirmation = {tuple(default[1:]): [], tuple(best[1:]): []}
        with torch.cuda.device(A.device):
            for round_index in range(10):
                ordered = tuple(confirmation)
                if round_index % 2:
                    ordered = ordered[::-1]
                for tile, cluster, group_size in ordered:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(10):
                        gemm_symmetric(
                            A,
                            out,
                            C,
                            alpha,
                            beta,
                            tile_shape_mn=tile,
                            cluster_shape_mnk=cluster,
                            scheduler_group_size=group_size,
                        )
                    end.record()
                    end.synchronize()
                    confirmation[(tile, cluster, group_size)].append(
                        start.elapsed_time(end)
                    )
        default_time = statistics.median(confirmation[tuple(default[1:])])
        best_time = statistics.median(confirmation[tuple(best[1:])])
        if default_time > 1.05 * best_time:
            selected = best
    _, tile, cluster, group_size = selected
    return tile, cluster, group_size
