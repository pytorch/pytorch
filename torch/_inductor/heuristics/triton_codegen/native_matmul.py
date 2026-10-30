from __future__ import annotations

from typing import Any, cast, TYPE_CHECKING

from torch._inductor.runtime.hints import TRITON_MAX_TENSOR_NUMEL
from torch._inductor.runtime.runtime_utils import next_power_of_2


if TYPE_CHECKING:
    from triton import Config
else:
    from torch._inductor.runtime.triton_compat import Config


def native_matmul_configs(
    size_hints: dict[str, int],
    inductor_meta: dict[str, Any],
    triton_meta: dict[str, Any],
    *,
    r0_block: int | None = None,
) -> list[Config]:
    r"""native_matmul_configs(size_hints, inductor_meta, triton_meta,
    *, r0_block=None) -> list[Config]

    Generate up to nine shape-dependent native matmul configurations.
    Size hints use x=N, y=M, r0_=K, and optional z=batch. A fixed r0_block
    selects persistent configurations. Older kernels without GROUP_M use the
    existing configurations.
    """
    from torch._inductor.runtime.triton_heuristics import (
        _cap_native_matmul_configs,
        _config_helper,
        make_matmul_triton_config,
        unique_configs,
    )

    if len(size_hints) not in (3, 4):
        raise NotImplementedError("native matmul only supports mm/bmm pattern")
    device = triton_meta["device"]
    signature = triton_meta.get("signature", {})
    tune_group = "GROUP_M" in signature
    bmm = "z" in size_hints
    persistent = r0_block is not None
    compact = (
        inductor_meta.get("native_matmul_compact_tensors", False) and not persistent
    )
    if not tune_group:
        configs = [
            make_matmul_triton_config(sizes, warps, stages)
            for sizes, warps, stages in _config_helper(bmm=bmm, persistent=persistent)
        ]
        if r0_block is not None:
            configs = cast(list[Config], _cap_native_matmul_configs(configs, r0_block))
        return configs

    shape = inductor_meta.get("native_matmul_shape", size_hints)
    m, n, k = shape["y"], shape["x"], shape["r0_"]
    batch = max(1, shape.get("z", 1))
    sms = device.multi_processor_count or 132
    wide_blocks = batch * ((m + 127) // 128) * ((n + 255) // 256)
    wide_grid = compact and min(m, n) > 64 and wide_blocks >= sms
    dots = max(1, inductor_meta.get("native_matmul_num_dot", 1))
    m_hint, n_hint, k_hint = (max(16, next_power_of_2(d)) for d in (m, n, k))
    min_xblock = max(16, next_power_of_2(inductor_meta.get("min_xblock") or 16))
    proposals: list[tuple[int, int, int, int, int]] = []

    if persistent:
        proposals = [
            (16, 16, k_hint, 2, 1),
            (16, 16, k_hint, 4, 1),
            (16, 32, k_hint, 4, 1),
            (32, 16, k_hint, 4, 1),
            (32, 32, k_hint, 4, 1),
            (64, 32, k_hint, 4, 1),
            (32, 64, k_hint, 4, 1),
            (64, 128, k_hint, 4, 1),
            (128, 64, k_hint, 4, 1),
        ]
        if m <= n and batch * ((m + 31) // 32) * ((n + 63) // 64) >= sms:
            proposals[-1] = (32, 64, k_hint, 8, 1)
        if min(m, n) > 64:
            proposals[0] = (64, 64, k_hint, 4, 1)
        proposals = [
            (min(bm, m_hint), min(bn, n_hint), bk, w, s)
            for bm, bn, bk, w, s in proposals
        ]
    elif min(m, n) <= 64 and k > 128:
        if k <= 512:
            proposals = [
                (64, 64, 64, 4, 3),
                (32, 64, 64, 4, 3),
                (64, 32, 64, 4, 3),
                (64, 64, 128, 4, 5),
                (64, 64, 256, 4, 3),
                (16, 64, 128, 4, 5),
                (16, 128, 128, 4, 5),
                (32, 64, 128, 4, 5),
            ]
        else:
            short = 16 if m <= n or n <= 16 else 32
            stages = 6 if dots == 1 and (m <= n or n < 32) else 5
            proposals = [(64, 64, 256, 4, 3), (64, 64, 128, 4, 5)]
            if dots > 1 or min(m, n) <= 32:
                proposals.extend([(16, 128, 128, 8, 2), (16, 128, 256, 8, 1)])
            else:
                proposals.extend([(32, 64, 128, 4, 5), (128, 64, 128, 4, 5)])
            proposals.extend(
                [
                    (short, 64, 128, 4, stages),
                    (16, 32, 128, 4, 6)
                    if (m <= n and m % 16 == 0) or (n < m and n <= 16)
                    else (16, 32, 256, 4, 3),
                    (short, 64, 256, 4, 3),
                    (short, 128, 128, 4, 5),
                ]
            )
            if dots == 1 and m < n and 8 < m <= 32:
                # A 64-row tile lets Hopper use a warp-group MMA layout.
                proposals[-1] = (64, 64, 128, 4, 6)
            if dots == 1 and m <= n:
                proposals = [proposals[4], *proposals[:4], *proposals[5:]]
        if n < m:
            proposals = [(bn, bm, bk, w, s) for bm, bn, bk, w, s in proposals]
    elif batch * ((m + 31) // 32) * ((n + 31) // 32) < (2 if k >= 512 else 4) * sms:
        if k >= 512:
            proposals = [
                (64, 32, 128, 4, 5),
                (32, 64, 128, 4, 5),
                (32, 16, 128, 4, 3),
                (16, 32, 128, 4, 3),
                (64, 16, 128, 4, 3),
                (16, 64, 128, 4, 3),
                (16, 16, 128, 4, 3),
                (32, 32, 128, 4, 3),
            ]
        else:
            proposals = [
                (16, 16, min(128, k_hint), 4, 3),
                (16, 16, min(128, k_hint), 2, 2),
                (64, 16, min(128, k_hint), 4, 3),
                (16, 64, min(128, k_hint), 4, 3),
                (32, 16, min(128, k_hint), 4, 3),
                (16, 32, min(128, k_hint), 4, 3),
                (64, 32, min(128, k_hint), 4, 3),
                (32, 64, min(128, k_hint), 4, 3),
                (32, 32, min(64, k_hint), 4, 3),
            ]
        if batch > 1 and min(m, n) > 32 and k > 128:
            proposals = [(64, 64, 64, 8, 3), (64, 32, 32, 8, 5), *proposals[:6]]
        if k < 512 and k % 16:
            proposals = [
                (
                    bm,
                    bn,
                    min(bk, 64) if min(bm, bn) == 16 and max(bm, bn) == 32 else bk,
                    w,
                    s,
                )
                for bm, bn, bk, w, s in proposals
            ]
        proposals = [
            (min(bm, m_hint), min(bn, n_hint), bk, w, s)
            for bm, bn, bk, w, s in proposals
        ]
    else:
        if k <= 256:
            proposals = [
                (128, 128, 64, 8, 3),
                (128, 128, 32, 8, 2),
                (64, 128, 64, 4, 3),
                (128, 64, 64, 4, 3),
                (64, 64, 64, 4, 3),
                (64, 64, 32, 4, 2),
                (64, 128, 128, 4, 3),
                (128, 64, 128, 4, 3),
            ]
        elif any(d % 16 for d in (m, n, k)):
            # Unaligned rows benefit from smaller K tiles or a shallow pipeline.
            proposals = [
                (128, 128, 32, 8, 2),
                (256, 128, 128, 8, 1)
                if compact and dots == 1
                else (64, 128, 256, 8, 2),
                (64, 128, 32, 4, 3),
                (256, 128, 32, 8, 3) if dots == 1 else (64, 32, 128, 4, 3),
                (64, 128, 128, 4, 4),
                (128, 64, 128, 4, 4),
                (64, 64, 128, 4, 3),
                (128, 128, 64, 8, 5),
            ]
            if batch * ((m + 127) // 128) * ((n + 127) // 128) <= sms:
                proposals = [proposals[-1], *proposals[:-1]]
        elif wide_grid and dots == 1:
            proposals = [
                (128, 256, 64, 8, 3),
                (128, 256, 64, 8, 4),
                (256, 128, 64, 8, 4),
                (128, 128, 64, 8, 5),
                (128, 128, 64, 4, 3),
                (64, 128, 128, 4, 4),
                (128, 64, 128, 4, 4),
                (64, 64, 128, 4, 3),
            ]
        else:
            proposals = [
                (128, 128, 64, 8, 5),
                (128, 128, 64, 8 if dots > 1 else 4, 3),
                (64, 128, 128, 4, 4),
                (128, 64, 128, 4, 4),
                (64, 64, 128, 4, 5),
                (64, 64, 128, 4, 3),
                (64, 128, 64, 4, 3),
                (128, 64, 64, 4, 3),
            ]
            if dots == 1 and wide_blocks >= sms:
                proposals[-1] = (128, 256, 32, 8, 3) if n >= m else (256, 128, 32, 8, 3)
            elif wide_grid:
                proposals = [(128, 256, 64, 16, 4), *proposals[:7]]
        proposals = [
            (min(bm, m_hint), min(bn, n_hint), min(bk, k_hint), w, s)
            for bm, bn, bk, w, s in proposals
        ]

    configs = []
    for bm, bn, bk, warps, stages in proposals:
        bn = max(bn, min_xblock)
        block_k = bk if r0_block is None else r0_block
        if bm * bn * dots > warps * device.warp_size_or_default * 192:
            continue
        kwargs = {"XBLOCK": bn, "YBLOCK": bm}
        if bmm:
            kwargs["ZBLOCK"] = 1
        if not persistent:
            kwargs["R0_BLOCK"] = bk
        numel = bm * bn * block_k
        if compact:
            numel = max(bm * bn, bm * block_k, bn * block_k)
        if numel > TRITON_MAX_TENSOR_NUMEL:
            continue
        cfg = Config(kwargs, num_warps=warps, num_stages=stages)
        if tune_group:
            rows = max(1, (m + bm - 1) // bm)
            cfg.kwargs["GROUP_M"] = (
                1 if n <= bn or rows <= 4 else min(8, next_power_of_2(rows))
            )
        configs.append(cfg)
    configs = unique_configs(configs)[:9]
    if not configs:
        configs = [
            make_matmul_triton_config(sizes, warps, stages)
            for sizes, warps, stages in _config_helper(bmm=bmm, persistent=persistent)
        ]
        if r0_block is not None:
            configs = cast(list[Config], _cap_native_matmul_configs(configs, r0_block))
        for cfg in configs:
            cfg.kwargs["GROUP_M"] = 8
        return configs[:9]

    if tune_group and len(configs) < 9:
        for cfg in configs:
            rows = max(1, (m + cfg.kwargs["YBLOCK"] - 1) // cfg.kwargs["YBLOCK"])
            if n <= cfg.kwargs["XBLOCK"] or rows <= 1:
                continue
            if rows <= 8:
                group = next_power_of_2(rows) if rows <= 4 else 1
            else:
                span = cfg.kwargs["YBLOCK"] * k * 32 * 2
                group = 16 if span > 40 * 1024 * 1024 else 32
            kwargs = {**cfg.kwargs, "GROUP_M": min(group, next_power_of_2(rows))}
            if kwargs == cfg.kwargs:
                continue
            configs.append(
                Config(kwargs, num_warps=cfg.num_warps, num_stages=cfg.num_stages)
            )
            break
    return unique_configs(configs)
