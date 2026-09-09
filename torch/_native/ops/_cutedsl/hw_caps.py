# Hardware capability struct: launch heuristics as formulas in device properties rather than
# magic numbers, so the same rule reasons across architectures instead of being tuned to one
# GPU. All read from get_device_properties, cached per device index.

import functools

import torch


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
        # Peak DRAM bandwidth. Divide LAST: `// 8` first truncates a bus width that is not a multiple
        # of 8. `memory_clock_rate` is NOT read off `p` -- it is registered as a property whose lambda
        # ignores the cudaDeviceProp and queries the CURRENT device, so on a heterogeneous box this
        # would pair one device's bus width with another's clock, and the result is memoized per index.
        with torch.cuda.device(device):
            mem_clock_khz = p.memory_clock_rate
        self.peak_bw_bytes = p.memory_bus_width * mem_clock_khz * 1000 * 2 // 8

    # --- derived quantities the launch heuristics reason in ---
    @property
    def max_warps_per_sm(self):
        return self.max_threads_per_sm // self.warp

    def blocks_per_sm(self, threads_per_block):
        # Concurrent blocks an SM holds, occupancy-bound: register and smem limits are the callers'
        # to check. A non-positive block size raises rather than normalizing, which would return a
        # plausible occupancy for an unlaunchable grid. The clamp at 1 is for a block larger than an
        # SM holds, where callers would otherwise divide by zero.
        if threads_per_block <= 0:
            raise ValueError(
                f"threads_per_block must be positive, got {threads_per_block}"
            )
        return max(1, self.max_threads_per_sm // threads_per_block)

    def waves(self, total_blocks, threads_per_block):
        # How many occupancy "waves" a grid of `total_blocks` spans. A wave = the
        # device running its max concurrent blocks once.
        concurrent = self.sm_count * self.blocks_per_sm(threads_per_block)
        return total_blocks / max(concurrent, 1)

    def fill_blocks(self, threads_per_block, waves=1.0):
        # Number of blocks needed to fill the device to `waves` occupancy waves.
        return int(self.sm_count * self.blocks_per_sm(threads_per_block) * waves)


@functools.cache
def _caps(index: int) -> "HWCaps":
    return HWCaps(index)


def caps(device=None):
    # Memoize on the RESOLVED index: `caps(None)` means the current device, which changes, so
    # caching under None would serve one device's properties for another -- a wrong launch shape
    # rather than an error.
    return _caps(torch.cuda._utils._get_device_index(device, optional=True))
