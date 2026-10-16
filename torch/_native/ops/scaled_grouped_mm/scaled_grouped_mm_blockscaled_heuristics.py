from typing import NamedTuple


class _KernelConfig(NamedTuple):
    mma_tile_mn: tuple[int, int]
    cluster_shape_mn: tuple[int, int]
    transpose_ab: bool


def _select_kernel_config_fp8(M: int, N: int, K: int) -> _KernelConfig:
    # Port of MSLK mx8mx8bf16_grouped::get_kernel_via_heuristics()
    cfg_256_64_ba = _KernelConfig((256, 64), (2, 1), True)
    cfg_256_128_ba = _KernelConfig((256, 128), (2, 1), True)
    cfg_256_256_ab = _KernelConfig((256, 256), (2, 1), False)
    cfg_256_256_ba = _KernelConfig((256, 256), (2, 1), True)

    M = int(M)
    N = int(N)
    K = int(K)

    # CuTeDSL-tuned overrides for large per-group workloads where the
    # MSLK heuristic tends to over-select transpose_ab=True.
    if M >= 2048 and N >= 6144 and K >= 2048:
        return cfg_256_256_ab
    if N == 5120 and K == 8192 and M <= 4096:
        return cfg_256_256_ab

    if M <= 64:
        return cfg_256_64_ba
    elif M <= 128:
        if N <= 1024:
            if K <= 5120:
                return cfg_256_64_ba
            else:
                return cfg_256_128_ba
        else:
            return cfg_256_128_ba
    elif M <= 1024:
        if N <= 1024:
            if K <= 1024:
                return cfg_256_256_ab
            elif K <= 5120:
                return cfg_256_256_ba
            elif K <= 6144:
                return cfg_256_256_ab
            elif K <= 7168:
                return cfg_256_256_ba
            elif K <= 8192:
                return cfg_256_256_ab
            else:
                return cfg_256_256_ba
        elif N <= 2048:
            if K <= 1024:
                return cfg_256_256_ba
            elif K <= 2048:
                return cfg_256_256_ab
            elif K <= 7168:
                return cfg_256_256_ba
            elif K <= 8192:
                return cfg_256_256_ab
            else:
                return cfg_256_256_ba
        elif N <= 4096:
            if K <= 5120:
                return cfg_256_256_ba
            elif K <= 6144:
                return cfg_256_256_ab
            elif K <= 8192:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
        elif N <= 5120:
            if K <= 1024:
                return cfg_256_256_ab
            elif K <= 5120:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
        elif N <= 6144:
            if K <= 1024:
                return cfg_256_256_ba
            elif K <= 2048:
                return cfg_256_256_ab
            elif K <= 6144:
                return cfg_256_256_ba
            elif K <= 8192:
                return cfg_256_256_ab
            else:
                return cfg_256_256_ba
        elif N <= 7168:
            if K <= 5120:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
        elif N <= 8192:
            if K <= 4096:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
        else:
            if K <= 1024:
                return cfg_256_256_ab
            elif K <= 2048:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
    elif M <= 2048:
        if N <= 1024:
            if K <= 1024:
                return cfg_256_256_ba
            elif K <= 2048:
                return cfg_256_256_ab
            elif K <= 7168:
                return cfg_256_256_ba
            elif K <= 8192:
                return cfg_256_256_ab
            else:
                return cfg_256_256_ba
        elif N <= 2048:
            return cfg_256_256_ba
        elif N <= 4096:
            if K <= 1024:
                return cfg_256_256_ba
            elif K <= 2048:
                return cfg_256_256_ab
            elif K <= 4096:
                return cfg_256_256_ba
            elif K <= 7168:
                return cfg_256_256_ab
            elif K <= 8192:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
        elif N <= 6144:
            if K <= 1024:
                return cfg_256_256_ab
            elif K <= 6144:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
        elif N <= 7168:
            if K <= 1024:
                return cfg_256_256_ab
            elif K <= 5120:
                return cfg_256_256_ba
            elif K <= 8192:
                return cfg_256_256_ab
            else:
                return cfg_256_256_ab
        elif N <= 8192:
            if K <= 4096:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
        else:
            if K <= 2048:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
    elif M <= 4096:
        if N <= 1024:
            if K <= 1024:
                return cfg_256_256_ab
            else:
                return cfg_256_256_ba
        elif N <= 2048:
            if K <= 2048:
                return cfg_256_256_ab
            elif K <= 5120:
                return cfg_256_256_ba
            elif K <= 6144:
                return cfg_256_256_ab
            else:
                return cfg_256_256_ba
        elif N <= 4096:
            if K <= 1024:
                return cfg_256_256_ab
            elif K <= 8192:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
        elif N <= 5120:
            if K <= 8192:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
        elif N <= 6144:
            if K <= 1024:
                return cfg_256_256_ab
            elif K <= 4096:
                return cfg_256_256_ba
            elif K <= 5120:
                return cfg_256_256_ab
            elif K <= 7168:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
        elif N <= 7168:
            if K <= 2048:
                return cfg_256_256_ba
            elif K <= 4096:
                return cfg_256_256_ab
            elif K <= 5120:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
        elif N <= 8192:
            if K <= 1024:
                return cfg_256_256_ba
            elif K <= 2048:
                return cfg_256_256_ab
            elif K <= 4096:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
        else:
            if K <= 1024:
                return cfg_256_256_ab
            elif K <= 2048:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
    elif M <= 5120:
        if N <= 1024:
            if K <= 1024:
                return cfg_256_256_ab
            else:
                return cfg_256_256_ba
        elif N <= 2048:
            return cfg_256_256_ba
        elif N <= 4096:
            if K <= 1024:
                return cfg_256_256_ba
            elif K <= 2048:
                return cfg_256_256_ab
            else:
                return cfg_256_256_ba
        elif N <= 5120:
            if K <= 1024:
                return cfg_256_256_ba
            elif K <= 2048:
                return cfg_256_256_ab
            elif K <= 6144:
                return cfg_256_256_ba
            elif K <= 7168:
                return cfg_256_256_ab
            else:
                return cfg_256_256_ba
        elif N <= 6144:
            if K <= 1024:
                return cfg_256_256_ab
            else:
                return cfg_256_256_ba
        elif N <= 7168:
            if K <= 5120:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
        elif N <= 8192:
            if K <= 1024:
                return cfg_256_256_ab
            elif K <= 5120:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
        else:
            if K <= 2048:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
    elif M <= 6144:
        if N <= 1024:
            return cfg_256_256_ba
        elif N <= 2048:
            if K <= 1024:
                return cfg_256_256_ab
            else:
                return cfg_256_256_ba
        elif N <= 4096:
            return cfg_256_256_ba
        elif N <= 5120:
            if K <= 7168:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
        elif N <= 6144:
            return cfg_256_256_ba
        elif N <= 7168:
            if K <= 4096:
                return cfg_256_256_ba
            elif K <= 5120:
                return cfg_256_256_ab
            elif K <= 6144:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
        elif N <= 8192:
            if K <= 5120:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
        else:
            if K <= 1024:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
    elif M <= 7168:
        if N <= 2048:
            if K <= 1024:
                return cfg_256_256_ba
            elif K <= 2048:
                return cfg_256_256_ab
            else:
                return cfg_256_256_ba
        elif N <= 4096:
            if K <= 1024:
                return cfg_256_256_ab
            else:
                return cfg_256_256_ba
        elif N <= 6144:
            return cfg_256_256_ba
        elif N <= 7168:
            if K <= 8192:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
        elif N <= 8192:
            if K <= 2048:
                return cfg_256_256_ba
            elif K <= 5120:
                return cfg_256_256_ab
            elif K <= 7168:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
        else:
            if K <= 1024:
                return cfg_256_256_ab
            elif K <= 2048:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
    elif M <= 8192:
        if N <= 1024:
            return cfg_256_256_ba
        elif N <= 2048:
            if K <= 1024:
                return cfg_256_256_ab
            else:
                return cfg_256_256_ba
        elif N <= 8192:
            return cfg_256_256_ba
        else:
            if K <= 2048:
                return cfg_256_256_ba
            else:
                return cfg_256_256_ab
    else:
        if N <= 1024:
            if K <= 1024:
                return cfg_256_256_ab
            else:
                return cfg_256_256_ba
        else:
            return cfg_256_256_ba


def _select_kernel_config_fp4(M: int, N: int, K: int) -> _KernelConfig:
    # Port of MSLK f4f4bf16_grouped::get_kernel_via_heuristics().  The
    # FP4 kernel family carries its K-tiling in the compiled
    # dtype-specific kernel, so the MSLK 256x256x128 vs 256x256x256
    # distinction collapses to a single 256x256 choice here.  Unlike
    # MSLK, groups of up to 128 rows are transposed onto the N tile,
    # as in the FP8 heuristic.
    cfg_128_64 = _KernelConfig((128, 64), (1, 1), True)
    cfg_256_64 = _KernelConfig((256, 64), (2, 1), True)
    cfg_256_128_ba = _KernelConfig((256, 128), (2, 1), True)
    cfg_256_256 = _KernelConfig((256, 256), (2, 1), False)

    M = int(M)
    N = int(N)
    K = int(K)

    if M <= 1:
        return cfg_256_64
    if M <= 64:
        if N <= 1024:
            return cfg_256_64
        if N <= 2048:
            return cfg_128_64 if K <= 2048 else cfg_256_64
        if N <= 7168:
            return cfg_256_64
        if N <= 8192:
            if K <= 6144:
                return cfg_256_64
            if K <= 7168:
                return cfg_128_64
            return cfg_256_64
        return cfg_256_64
    if M <= 128:
        return cfg_256_128_ba
    return cfg_256_256
