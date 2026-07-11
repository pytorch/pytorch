# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# SM120 (Blackwell GeForce / DGX Spark) backward pass.
#
# SM120 uses the same SM80-era MMA instructions (mma.sync.aligned.m16n8k16) but has
# a smaller shared memory capacity (99 KB vs 163 KB on SM80). This module subclasses
# FlashAttentionBackwardSm80 and overrides the SMEM capacity check accordingly.

import cutlass
import cutlass.utils as utils_basic

from .flash_bwd import FlashAttentionBackwardSm80


class FlashAttentionBackwardSm120(FlashAttentionBackwardSm80):
    @staticmethod
    def can_implement(
        dtype,
        head_dim,
        head_dim_v,
        m_block_size,
        n_block_size,
        num_stages_Q,
        num_stages_dO,
        num_threads,
        is_causal,
        V_in_regs=False,
    ) -> bool:
        """Check if the kernel can be implemented on SM120.

        Same logic as SM80 but uses SM120's shared memory capacity (99 KB).
        """
        if dtype not in [cutlass.Float16, cutlass.BFloat16]:
            return False
        if head_dim % 8 != 0:
            return False
        if head_dim_v % 8 != 0:
            return False
        if n_block_size % 16 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        # Shared memory usage: Q tile + dO tile + K tile + V tile
        smem_usage_Q = m_block_size * head_dim * num_stages_Q * 2
        smem_usage_dO = m_block_size * head_dim_v * num_stages_dO * 2
        smem_usage_K = n_block_size * head_dim * 2
        smem_usage_V = n_block_size * head_dim_v * 2
        smem_usage_QV = (
            (smem_usage_Q + smem_usage_V) if not V_in_regs else max(smem_usage_Q, smem_usage_V)
        )
        smem_usage = smem_usage_QV + smem_usage_dO + smem_usage_K
        # SM120 has 99 KB shared memory (vs 163 KB on SM80)
        smem_capacity = utils_basic.get_smem_capacity_in_bytes("sm_120")
        if smem_usage > smem_capacity:
            return False
        return True
