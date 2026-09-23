# Express launch heuristics in device properties rather than architecture-specific constants.
# Properties come from get_device_properties and are cached by resolved device index.

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
        # Divide last to avoid truncating non-byte-multiple bus widths. memory_clock_rate
        # queries the current device, so request the indexed property separately.
        with torch.cuda.device(device):
            mem_clock_khz = p.memory_clock_rate
        self.peak_bw_bytes = p.memory_bus_width * mem_clock_khz * 1000 * 2 // 8

    # --- derived quantities the launch heuristics reason in ---
    @property
    def max_warps_per_sm(self):
        return self.max_threads_per_sm // self.warp

    def blocks_per_sm(self, threads_per_block):
        # Thread-bound blocks per SM; callers handle register and smem limits. Reject invalid
        # block sizes, but floor oversized blocks at one to keep callers from dividing by zero.
        if threads_per_block <= 0:
            raise ValueError(
                f"threads_per_block must be positive, got {threads_per_block}"
            )
        return max(1, self.max_threads_per_sm // threads_per_block)

    def waves(self, total_blocks, threads_per_block):
        # Grid size in occupancy waves; one wave runs the device's maximum concurrent blocks.
        concurrent = self.sm_count * self.blocks_per_sm(threads_per_block)
        return total_blocks / max(concurrent, 1)

    def fill_blocks(self, threads_per_block, waves=1.0):
        # Number of blocks needed to fill the device to `waves` occupancy waves.
        return int(self.sm_count * self.blocks_per_sm(threads_per_block) * waves)


@functools.cache
def _caps(index: int) -> "HWCaps":
    return HWCaps(index)


def caps(device=None):
    # Resolve before caching because caps(None) follows the mutable current device.
    return _caps(torch.cuda._utils._get_device_index(device, optional=True))
