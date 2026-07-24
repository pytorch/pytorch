# Hardware capability struct for launch heuristics.
#
# WHY: the reduction launch heuristics were full of magic numbers (2*sm, block_x=64,
# the 3/4*sm gate) that are really PROXIES for hardware quantities. Expressing them
# as formulas in device properties makes the SAME heuristic code reason correctly
# across architectures -- Hopper (132 SM, 228KB smem), Blackwell (148 SM, 232KB),
# Rubin (future, unknown counts) -- instead of being silently tuned to one GPU.
#
# Everything here is read from torch.cuda.get_device_properties; nothing is baked.
# Cached per device index (properties don't change at runtime).

import torch

from .plan_cache import cached_plan


_CACHE = {}


class HWCaps:
    # Raw, portable device facts + a few derived quantities the heuristics want.
    def __init__(self, device=None):
        p = torch.cuda.get_device_properties(device)
        # --- raw, all architecture-portable ---
        self.name = p.name
        self.cc = (p.major, p.minor)  # compute capability
        self.sm_count = p.multi_processor_count  # # of SMs (132 H100, 148 B200)
        self.warp = p.warp_size  # 32 (stable, but read it)
        self.max_threads_per_sm = (
            p.max_threads_per_multi_processor
        )  # 2048 occupancy cap
        self.max_threads_per_block = p.max_threads_per_block  # 1024
        self.regs_per_sm = p.regs_per_multiprocessor
        self.smem_per_block_optin = p.shared_memory_per_block_optin  # 228KB H, 232KB B
        self.smem_per_sm = p.shared_memory_per_multiprocessor
        self.l2_bytes = p.L2_cache_size
        # peak DRAM bandwidth (bytes/s): bus_width(bits)/8 * memclock(kHz)*1e3 * 2 (DDR)
        self.peak_bw_bytes = p.memory_bus_width // 8 * p.memory_clock_rate * 1000 * 2

    # --- derived quantities the launch heuristics reason in ---
    @property
    def max_warps_per_sm(self):
        return self.max_threads_per_sm // self.warp

    def waves(self, total_blocks, threads_per_block):
        # How many occupancy "waves" a grid of `total_blocks` spans. A wave = the
        # device running its max concurrent blocks once. Concurrency is occupancy-
        # bound: blocks_per_sm = max_threads_per_sm // threads_per_block (ignoring
        # reg/smem limits, which the caller checks separately for big kernels).
        blocks_per_sm = max(1, self.max_threads_per_sm // max(threads_per_block, 1))
        concurrent = self.sm_count * blocks_per_sm
        return total_blocks / max(concurrent, 1)

    def fill_blocks(self, threads_per_block, waves=1.0):
        # Number of blocks needed to fill the device to `waves` occupancy waves.
        blocks_per_sm = max(1, self.max_threads_per_sm // max(threads_per_block, 1))
        return int(self.sm_count * blocks_per_sm * waves)


def caps(device=None):
    idx = (
        torch.cuda.current_device()
        if device is None
        else (device if isinstance(device, int) else torch.cuda.current_device())
    )
    return cached_plan(_CACHE, idx, lambda: HWCaps(device))
