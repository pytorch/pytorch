import functools
from typing import List, Optional

import torch
from torch import Tensor
from torch._inductor import utils
from torch.utils._mode_utils import no_dispatch
from ...utils._triton import has_triton
from ..ir import FixedLayout

from ..pattern_matcher import fwd_only, joint_fwd_bwd, Match, register_replacement
from ..utils import use_cutlass_template

aten = torch.ops.aten


def fetch_fake_tensors(match, kwarg_names) -> List[Tensor]:
    kwargs = match.kwargs
    return [kwargs[name].meta["val"] for name in kwarg_names]


def unwrap_fake_args(*arg_names):
    def decorator(func):
        def wrapper(match):
            fake_tensors = fetch_fake_tensors(match, arg_names)
            return func(*fake_tensors)

        return wrapper

    return decorator


def get_alignment_size(x: Tensor) -> int:
    if x.dtype == torch.float16 or x.dtype == torch.half or x.dtype == torch.bfloat16:
        return 8
    elif x.dtype == torch.float32 or x.dtype == torch.float:
        return 4
    else:
        return 0


def check_device(a: Tensor, b: Tensor) -> bool:
    return a.is_cuda and b.is_cuda


def check_dtype(a: Tensor, b: Tensor) -> bool:
    return a.is_floating_point() and b.is_floating_point()


def should_pad_common(
    mat1: Tensor, mat2: Tensor, input: Optional[Tensor] = None
) -> bool:
    return (
        torch._inductor.config.shape_padding
        and check_device(mat1, mat2)
        and check_dtype(mat1, mat2)
        and not utils.any_is_symbolic(mat1, mat2, input)
    )


def get_padded_length(x: int, alignment_size) -> int:
    if alignment_size == 0 or x % alignment_size == 0:
        return 0
    return int((x // alignment_size + 1) * alignment_size) - x


def pad_dim(x: Tensor, padded_length: int, dim: int) -> Tensor:
    if padded_length == 0:
        return x
    pad = x.new_zeros(*x.shape[:dim], padded_length, *x.shape[dim + 1 :])
    return torch.cat([x, pad], dim=dim)


def addmm_pattern(
    input: Tensor, mat1: Tensor, mat2: Tensor, beta: float, alpha: float
) -> Tensor:
    return aten.addmm(input, mat1, mat2, beta=beta, alpha=alpha)


def should_pad_addmm(match: Match) -> bool:
    mat1, mat2, input = fetch_fake_tensors(match, ("mat1", "mat2", "input"))
    return should_pad_common(mat1, mat2, input) and should_pad_bench(
        mat1, mat2, torch.ops.aten.addmm, input=input
    )


def pad_addmm(
    input: Optional[Tensor],
    mat1: Tensor,
    mat2: Tensor,
    m_padded_length: int,
    k_padded_length: int,
    n_padded_length: int,
    beta=1.0,
    alpha=1.0,
    explicit_transpose=False,
):
    # for paddings, dim order is reversed for some reasons
    # and for every dim, we need to specify left and right padding
    if k_padded_length != 0 or m_padded_length != 0:
        mat1_padded = aten.constant_pad_nd(
            mat1, [0, k_padded_length, 0, m_padded_length]
        )
    else:
        mat1_padded = mat1
    if k_padded_length != 0 or n_padded_length != 0:
        mat2_padded = aten.constant_pad_nd(
            mat2, [0, n_padded_length, 0, k_padded_length]
        )
    else:
        mat2_padded = mat2
    if input is not None:
        if len(input.shape) < 2:
            # make sure we have at least two dimensions
            # the first one to be broadcasted over is sometimes implicit
            input = input.unsqueeze(0)
        if n_padded_length != 0 or m_padded_length != 0:
            bias_n_padded_length = n_padded_length
            bias_m_padded_length = m_padded_length
            # What if we're broadcasting?
            if input.shape[0] == 1 and mat1.shape[0] > 1:
                bias_m_padded_length = 0
            if input.shape[1] == 1 and mat2.shape[1] > 1:
                bias_n_padded_length = 0
            if bias_m_padded_length > 0 or bias_n_padded_length > 0:
                input_padded = aten.constant_pad_nd(
                    input, [0, bias_n_padded_length, 0, bias_m_padded_length]
                )
            else:
                input_padded = input
        else:
            input_padded = input
    else:
        input_padded = None
    if explicit_transpose:
        # If M dimension is aligned but N is not, this is an alternative to a padding N
        # which has the advantage of enabling downstream epilogue fusions
        # padding on K dim, transpose and contiguous should be fuseable into a single op

        res = aten.addmm(
            input_padded.transpose(-1, -2),
            mat2_padded.transpose(-1, -2).contiguous(),
            mat1_padded.transpose(-1, -2),
        ).transpose(-1, -2)
    else:
        try:
            res = aten.addmm(
                input_padded, mat1_padded, mat2_padded, beta=beta, alpha=alpha
            )
        except RuntimeError as e:
            if input_padded is not None:
                note1 = f"\npad_addmm was called with argument shapes: input.shape={input.shape}, mat1.shape={mat1.shape}, mat2.shape={mat2.shape}, m_padded_length={m_padded_length}, k_padded_length={k_padded_length}, n_padded_length={n_padded_length}, explicit_transpose={explicit_transpose}"  # type: ignore[union-attr] # noqa: B950
            else:
                note1 = f"pad_addmm was called with argument shapes: input_padded=None, mat1.shape={mat1.shape}, mat2.shape={mat2.shape}, m_padded_length={m_padded_length}, k_padded_length={k_padded_length}, n_padded_length={n_padded_length}, explicit_transpose={explicit_transpose}"  # noqa: B950

            note2 = f"\naten.addmm was called with shapes: input_padded.shape={input_padded.shape}, mat1_padded.shape={mat1_padded.shape}, mat2_padded.shape={mat2_padded.shape}, beta={beta}, alpha={alpha}"  # noqa: B950
            raise RuntimeError(str(e) + note1 + note2) from e

    if m_padded_length != 0:
        res = res[:-m_padded_length, :]
    if n_padded_length != 0:
        res = res[:, :-n_padded_length]
    return res


def addmm_replace(
    input: Optional[Tensor], mat1: Tensor, mat2: Tensor, beta=1.0, alpha=1.0
) -> Tensor:
    k_padded_length = get_padded_length(mat1.shape[1], get_alignment_size(mat1))
    if not torch._inductor.config.shape_pad_only_k_dim:
        n_padded_length = get_padded_length(mat2.shape[1], get_alignment_size(mat2))
        m_padded_length = get_padded_length(mat1.shape[0], get_alignment_size(mat1))
    else:
        n_padded_length = 0
        m_padded_length = 0
    explicit_transpose = 0
    if torch._inductor.config.shape_pad_use_transpose:
        if m_padded_length == 0 and n_padded_length != 0:
            explicit_transpose = True
            n_padded_length = 0
            m_padded_length = 0
        elif m_padded_length != 0 and n_padded_length == 0:
            m_padded_length = 0
    return pad_addmm(
        input,
        mat1,
        mat2,
        m_padded_length,
        k_padded_length,
        n_padded_length,
        beta,
        alpha,
        explicit_transpose=explicit_transpose,
    )


def is_mm_compute_bound(M: int, K: int, N: int, dtype: torch.dtype) -> bool:
    denominator = M * K + N * K + M * N
    if denominator == 0:
        return False
    arithmetic_intensity = (M * N * K) / denominator

    # Fails with AMD
    try:
        machine_balance = (
            1000 * utils.get_device_tflops(dtype)
        ) / utils.get_gpu_dram_gbps()
    except Exception:
        return True

    # dram_gbps might be underestimating bandwidth because of cache.
    # if we estimate machine balance too low we might miss some speedups,
    # if we extimate too high there will be unnecessary compilation time increase.
    # TODO - finetune coefficient here. As a reference point, Triton mm model assumes
    # 80% of reads are in cache and cache is 4x faster than dram_gbps
    machine_balance = machine_balance * 0.5

    return arithmetic_intensity > machine_balance


@functools.lru_cache(None)
def get_pad_cache():
    return torch._inductor.codecache.LocalCache()


def get_cached_should_pad(key):
    return get_pad_cache().lookup(key)


def set_cached_should_pad(key, value):
    return get_pad_cache().set_value(key, value=value)


def should_pad_bench_key(
    mat1: Tensor, mat2: Tensor, op, input: Optional[Tensor] = None
) -> str:
    def tensor_key(t):
        return (t.shape, t.stride(), t.dtype)

    tf32_key = (
        None if mat1.dtype != torch.float32 else torch.backends.cuda.matmul.allow_tf32
    )
    key = (
        tensor_key(mat1),
        tensor_key(mat2),
        op,
        input if input is None else tensor_key(input),
        tf32_key,
        torch._inductor.config.shape_pad_only_k_dim,
        torch._inductor.config.shape_pad_always,
    )

    return str(key)


def should_pad_bench(
    mat1: Tensor, mat2: Tensor, op, input: Optional[Tensor] = None
) -> bool:
    if torch._inductor.config.shape_pad_always:
        return True

    do_bench = functools.partial(
        utils.do_bench,
        warmup=5,
    )
    m_padded_length = 0
    n_padded_length = 0
    batchsize = 1
    explicit_transpose = False
    with no_dispatch():
        if op is torch.ops.aten.mm or op is torch.ops.aten.addmm:
            m = mat1.shape[0]
            k = mat1.shape[1]
            n = mat2.shape[1]
            k_padded_length = get_padded_length(k, get_alignment_size(mat1))
            if not torch._inductor.config.shape_pad_only_k_dim:
                n_padded_length = get_padded_length(n, get_alignment_size(mat2))
                m_padded_length = get_padded_length(m, get_alignment_size(mat1))
        elif op is torch.ops.aten.bmm:
            batchsize = mat1.shape[0]
            m = mat1.shape[1]
            k = mat1.shape[2]
            n = mat2.shape[2]
            k_padded_length = get_padded_length(k, get_alignment_size(mat1))
            if not torch._inductor.config.shape_pad_only_k_dim:
                m_padded_length = get_padded_length(m, get_alignment_size(mat1))
                n_padded_length = get_padded_length(n, get_alignment_size(mat2))
        else:
            return False

        if torch._inductor.config.shape_pad_use_transpose:
            if m_padded_length == 0 and n_padded_length != 0:
                n_padded_length = 0
                m_padded_length = 0
                explicit_transpose = True
            elif n_padded_length == 0 and m_padded_length != 0:
                m_padded_length = 0
        if (
            m_padded_length == k_padded_length == n_padded_length == 0
        ) and not explicit_transpose:
            return False

        fake_layout = FixedLayout(
            device=mat1.device,
            dtype=mat1.dtype,
            size=[batchsize, m, n],
            stride=[n * m, n, 1],
        )
        if use_cutlass_template(fake_layout, m, n, k):
            # We cannot use I/O efficient Cutlass templates if the alignment doesn't meet TMA requirements
            return True

        if not has_triton():
            return False

        if not is_mm_compute_bound(m, k, n, mat1.dtype):
            return False

        # We don't want to look up the cache for cases that are trivially false
        # since it does file io
        key = should_pad_bench_key(mat1, mat2, op, input)

        cached_pad = get_cached_should_pad(key)
        if cached_pad is not None:
            return cached_pad

        mat1 = torch.randn_like(mat1)
        mat2 = torch.randn_like(mat2)
        if op is torch.ops.aten.bmm or op is torch.ops.aten.mm:
            ori_time = do_bench(
                lambda: op(mat1, mat2),
            )
        else:
            if input is not None:
                input = torch.randn_like(input)
            ori_time = do_bench(
                lambda: op(input, mat1, mat2),
            )

        mat1_pad = torch.randn_like(mat1)
        mat2_pad = torch.randn_like(mat2)

        if op is torch.ops.aten.addmm:
            input_pad = None
            if input is not None and input.is_cuda:
                input_pad = torch.randn_like(input)
            pad_time = do_bench(
                lambda: pad_addmm(
                    input_pad,
                    mat1_pad,
                    mat2_pad,
                    m_padded_length,
                    k_padded_length,
                    n_padded_length,
                    explicit_transpose=explicit_transpose,
                ),
            )
        elif op is torch.ops.aten.mm:
            pad_time = do_bench(
                lambda: pad_mm(
                    mat1_pad,
                    mat2_pad,
                    m_padded_length,
                    k_padded_length,
                    n_padded_length,
                    explicit_transpose=explicit_transpose,
                ),
            )
        else:
            pad_time = do_bench(
                lambda: pad_bmm(
                    mat1_pad,
                    mat2_pad,
                    m_padded_length,
                    k_padded_length,
                    n_padded_length,
                    explicit_transpose=explicit_transpose,
                ),
            )

        # Shape padding introduces additional memory ops. Based on microbenchmarks, 1.1x represents a reasonable
        # tradeoff between performance improvement from shape padding and overhead from additional memory ops
        # TODO: Build a learned model which would be better than this heuristic
        should_pad = ori_time > pad_time * 1.1
        set_cached_should_pad(key, should_pad)

        return should_pad


def mm_pattern(mat1: Tensor, mat2: Tensor) -> Tensor:
    return aten.mm(mat1, mat2)


def should_pad_mm(match: Match) -> bool:
    mat1, mat2 = fetch_fake_tensors(match, ("mat1", "mat2"))
    return should_pad_common(mat1, mat2) and should_pad_bench(
        mat1, mat2, torch.ops.aten.mm
    )


def pad_mm(
    mat1: Tensor,
    mat2: Tensor,
    m_padded_length: int,
    k_padded_length: int,
    n_padded_length: int,
    explicit_transpose: bool = False,
) -> Tensor:
    if k_padded_length != 0 or m_padded_length != 0:
        # dim order is reversed for constant_pad_nd, for every dim we specify right and left padding
        mat1_padded = aten.constant_pad_nd(
            mat1, [0, k_padded_length, 0, m_padded_length]
        )
    else:
        mat1_padded = mat1
    if k_padded_length != 0 or n_padded_length != 0:
        # dim order is reversed for constant_pad_nd, for every dim we specify right and left padding
        mat2_padded = aten.constant_pad_nd(
            mat2, [0, n_padded_length, 0, k_padded_length]
        )
    else:
        mat2_padded = mat2
    if explicit_transpose:
        # If M dimension is aligned but N is not, this is an alternative to a padding N
        # which has the advantage of enabling downstream epilogue fusions
        # padding on K dim, transpose and contiguous should be fuseable into a single op
        res = torch.ops.aten.mm(
            mat2_padded.transpose(-1, -2).contiguous(), mat1_padded.transpose(-1, -2)
        ).transpose(-1, -2)
    else:
        res = torch.ops.aten.mm(mat1_padded, mat2_padded)
    if m_padded_length != 0:
        res = res[:-m_padded_length, :]
    if n_padded_length != 0:
        res = res[:, :-n_padded_length]
    return res


def mm_replace(mat1: Tensor, mat2: Tensor) -> Tensor:
    k_padded_length = get_padded_length(mat1.shape[1], get_alignment_size(mat1))
    explicit_transpose = False
    if not torch._inductor.config.shape_pad_only_k_dim:
        m_padded_length = get_padded_length(mat1.shape[0], get_alignment_size(mat1))
        n_padded_length = get_padded_length(mat2.shape[1], get_alignment_size(mat2))
    else:
        m_padded_length = 0
        n_padded_length = 0
    if torch._inductor.config.shape_pad_use_transpose:
        if m_padded_length == 0 and n_padded_length != 0:
            explicit_transpose = True
            n_padded_length = 0
            m_padded_length = 0
        elif m_padded_length != 0 and n_padded_length == 0:
            m_padded_length = 0
    return pad_mm(
        mat1,
        mat2,
        m_padded_length,
        k_padded_length,
        n_padded_length,
        explicit_transpose=explicit_transpose,
    )


def bmm_pattern(mat1: Tensor, mat2: Tensor) -> Tensor:
    return aten.bmm(mat1, mat2)


def should_pad_bmm(match: Match) -> bool:
    mat1, mat2 = fetch_fake_tensors(match, ("mat1", "mat2"))
    return should_pad_common(mat1, mat2) and should_pad_bench(
        mat1, mat2, torch.ops.aten.bmm
    )


def pad_bmm(
    mat1: Tensor,
    mat2: Tensor,
    m_padded_length: int,
    k_padded_length: int,
    n_padded_length: int,
    explicit_transpose: bool = False,
) -> Tensor:
    if k_padded_length != 0 or m_padded_length != 0:
        mat1_padded = aten.constant_pad_nd(
            mat1, [0, k_padded_length, 0, m_padded_length, 0, 0]
        )
    else:
        mat1_padded = mat1
    if k_padded_length != 0 or n_padded_length != 0:
        mat2_padded = aten.constant_pad_nd(
            mat2, [0, n_padded_length, 0, k_padded_length, 0, 0]
        )
    else:
        mat2_padded = mat2
    if explicit_transpose:
        # If M dimension is aligned but N is not, this is an alternative to a padding N
        # which has the advantage of enabling downstream epilogue fusions
        # padding on K dim, transpose and contiguous should be fuseable into a single op
        res = aten.bmm(
            mat2_padded.transpose(-1, -2).contiguous(), mat1_padded.transpose(-1, -2)
        ).transpose(-1, -2)
    else:
        res = aten.bmm(mat1_padded, mat2_padded)
    if m_padded_length != 0:
        res = res[:, :-m_padded_length, :]
    if n_padded_length != 0:
        res = res[:, :, :-n_padded_length]
    return res


def bmm_replace(mat1: Tensor, mat2: Tensor) -> Tensor:
    k_padded_length = get_padded_length(mat1.shape[2], get_alignment_size(mat1))
    if not torch._inductor.config.shape_pad_only_k_dim:
        n_padded_length = get_padded_length(mat2.shape[2], get_alignment_size(mat2))
        m_padded_length = get_padded_length(mat1.shape[1], get_alignment_size(mat1))
    else:
        m_padded_length = 0
        n_padded_length = 0
    explicit_transpose = False
    if torch._inductor.config.shape_pad_use_transpose:
        if m_padded_length == 0 and n_padded_length != 0:
            explicit_transpose = True
            n_padded_length = 0
            m_padded_length = 0
        elif m_padded_length != 0 and n_padded_length == 0:
            m_padded_length = 0
    return pad_bmm(
        mat1,
        mat2,
        m_padded_length,
        k_padded_length,
        n_padded_length,
        explicit_transpose=explicit_transpose,
    )


@functools.lru_cache(None)
def _pad_mm_init():
    from .joint_graph import patterns

    if torch.cuda.is_available():
        # workaround https://github.com/pytorch/pytorch/issues/97894
        device = "cuda"
    else:
        device = "cpu"

    # sizes/values dont actually matter for initial trace
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
            mm_pattern,
            mm_replace,
            [dim2a(), dim2b()],
            {},
            should_pad_mm,
        ),
        (
            bmm_pattern,
            bmm_replace,
            [dim3a(), dim3b()],
            {},
            should_pad_bmm,
        ),
        (
            addmm_pattern,
            addmm_replace,
            [dim1a(), dim2a(), dim2b()],
            rep,
            should_pad_addmm,
        ),
    ]:
        assert isinstance(workaround, dict)  # mypy is unable to infer the type properly
        register_replacement(
            pattern,
            replacement,
            args,
            joint_fwd_bwd,
            patterns,
            extra_check=extra_check,
            scalar_workaround=workaround,
        )
        register_replacement(
            pattern,
            replacement,
            args,
            fwd_only,
            patterns,
            extra_check=extra_check,
            scalar_workaround=workaround,
        )
