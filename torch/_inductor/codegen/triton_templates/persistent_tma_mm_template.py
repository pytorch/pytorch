# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional

from torch._inductor.kernel.mm_common import persistent_mm_grid
from torch._inductor.select_algorithm import TritonTemplate

from ...utils import get_tma_workspace_arg


class PersistentTMATritonTemplate(TritonTemplate):
    """
    Triton template for persistent matrix multiplication using TMA descriptors.

    This template allows passing TMA workspace arguments as extra parameters that are
    not dependent on any particular config choice.
    """

    def maybe_append_choice(
        self, choices: list[Any], **kwargs: Any
    ) -> Optional[NotImplementedError]:
        input_nodes = kwargs.get("input_nodes", None)
        if (
            input_nodes is None
            or not hasattr(input_nodes, "__getitem__")
            or len(input_nodes) == 0
        ):
            raise RuntimeError("sequence of input_nodes required in kwargs")
        workspace_arg = get_tma_workspace_arg(
            num_tma_descriptors=2,
            device=input_nodes[0].get_device(),
        )
        return super().maybe_append_choice(
            choices, workspace_arg=workspace_arg, **kwargs
        )


persistent_tma_mm_template = PersistentTMATritonTemplate(
    name="mm_persistent_tma",
    grid=persistent_mm_grid,
    source=r"""
{{def_kernel("A", "B")}}
    M = {{size("A", 0)}}
    N = {{size("B", 1)}}
    K = {{size("A", 1)}}
    if M * N == 0:
        # early exit due to zero-size input(s)
        return

    start_pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = grid_m * grid_n
    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    width = GROUP_M * grid_n
    rk_for_mask = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    workspace_base = ws_ptr + start_pid * 2 * TMA_SIZE
    a_desc_ptr = workspace_base
    b_desc_ptr = workspace_base + TMA_SIZE

    triton.language.extra.cuda.experimental_device_tensormap_create2d(
        desc_ptr=a_desc_ptr,
        global_address=A,
        load_size=[BLOCK_M, BLOCK_K] if A_ROW_MAJOR else [BLOCK_K, BLOCK_M],
        global_size=[M, K] if A_ROW_MAJOR else [K, M],
        element_ty=A.dtype.element_ty,
    )
    triton.language.extra.cuda.experimental_device_tensormap_create2d(
        desc_ptr=b_desc_ptr,
        global_address=B,
        load_size=[BLOCK_K, BLOCK_N] if B_ROW_MAJOR else [BLOCK_N, BLOCK_K],
        global_size=[K, N] if B_ROW_MAJOR else [N, K],
        element_ty=B.dtype.element_ty,
    )

    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(a_desc_ptr)
    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(b_desc_ptr)

    pid_m = 0
    pid_n = 0
    rm = 0
    rn = 0

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            # re-order program ID for better L2 performance
            group_id = tile_id // width
            group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
            pid_m = group_id * GROUP_M + (tile_id % group_size)
            pid_n = (tile_id % width) // (group_size)

            rm = pid_m * BLOCK_M
            rn = pid_n * BLOCK_N

        rk = ki * BLOCK_K

        a = tl._experimental_descriptor_load(
            a_desc_ptr,
            [rm, rk] if A_ROW_MAJOR else [rk, rm],
            [BLOCK_M, BLOCK_K] if A_ROW_MAJOR else [BLOCK_K, BLOCK_M],
            A.dtype.element_ty,
        )
        b = tl._experimental_descriptor_load(
            b_desc_ptr,
            [rk, rn] if B_ROW_MAJOR else [rn, rk],
            [BLOCK_K, BLOCK_N] if B_ROW_MAJOR else [BLOCK_N, BLOCK_K],
            B.dtype.element_ty,
        )
        acc += tl.dot(
            a if A_ROW_MAJOR else a.T,
            b if B_ROW_MAJOR else b.T,
            allow_tf32=ALLOW_TF32,
        )

        if ki == k_tiles - 1:
            # rematerialize rm and rn to save registers
            rcm = rm + tl.arange(0, BLOCK_M)
            rcn = rn + tl.arange(0, BLOCK_N)
            idx_m = rcm[:, None]
            idx_n = rcn[None, :]
            mask = (idx_m < M) & (idx_n < N)

            # inductor generates a suffix
            {{store_output(("idx_m", "idx_n"), "acc", "mask", indent_width=12)}}
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

""",
)
