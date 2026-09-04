# Hardware capability struct for launch heuristics: expresses them as formulas in device properties
# rather than magic numbers, so the same heuristic reasons correctly across architectures instead of
# being silently tuned to one GPU. All read from get_device_properties, cached per device index.

import functools

import torch


# Reference GPU the launch-heuristic constants were TUNED on (B200 / sm_100a). Thresholds multiply
# by a hardware-scale RATIO against these, so the ratio is exactly 1.0 on B200 and the anchors
# reproduce byte-identically. EXACT device-property values -- do not round; smem is 233472 B, not
# 228*1024. Update only when re-tuning on a different reference GPU. A heuristic reads the RATIO,
# never these directly: kernel_xcta scales its sub-row target by `smem_scale`, so a device with more
# or less per-SM smem targets a proportionally longer or shorter sub-row.
_REF_SM_COUNT = 148
_REF_MAX_THREADS_PER_SM = 2048
_REF_SMEM_PER_SM = 233472
_REF_DEVICE_LANES = _REF_SM_COUNT * _REF_MAX_THREADS_PER_SM  # resident-thread capacity


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
        # peak DRAM bandwidth (bytes/s): bus_width(bits)/8 * memclock(kHz)*1e3 * 2 (DDR).
        # Divide LAST: `// 8` first truncates any bus width that is not a multiple of 8.
        #
        # `memory_clock_rate` is NOT read off `p`: torch/csrc/cuda/Module.cpp registers it as a
        # property whose lambda ignores the cudaDeviceProp and queries the CURRENT device, so on
        # a heterogeneous box this would pair `device`'s bus width with another device's clock --
        # and caps() memoizes the result per index. Scope the read.
        with torch.cuda.device(device):
            mem_clock_khz = p.memory_clock_rate
        self.peak_bw_bytes = p.memory_bus_width * mem_clock_khz * 1000 * 2 // 8

    # --- derived quantities the launch heuristics reason in ---
    @property
    def max_warps_per_sm(self):
        return self.max_threads_per_sm // self.warp

    def blocks_per_sm(self, threads_per_block):
        # Concurrent blocks an SM holds, occupancy-bound: reg/smem limits are ignored here
        # and checked separately by the callers that build big kernels. A non-positive block
        # size is a caller bug, so it raises rather than normalizing to one block per SM --
        # which would have returned a plausible occupancy for an unlaunchable grid. The
        # clamp at 1 stays for a block LARGER than an SM holds, where the floor is 0 and
        # callers divide by the result.
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

    @property
    def device_lanes(self):
        # Total resident-thread capacity of the device (SMs * threads/SM). The natural
        # unit for "how much parallel work saturates this GPU"; size thresholds scale
        # with it so a bigger/smaller device fills at a proportionally different numel.
        return self.sm_count * self.max_threads_per_sm

    @property
    def fill_scale(self):
        # Ratio of this device's fill capacity to the tuning reference (B200). 1.0 on
        # B200. A heuristic's element-count thresholds multiply by this so the same rule
        # fills a larger device at a larger numel and a smaller one sooner.
        return self.device_lanes / _REF_DEVICE_LANES

    @property
    def smem_scale(self):
        # Ratio of this device's per-SM smem to the reference. Governs how much
        # accumulator state (nfields * acc_width per lane, or an N-wide row tile) fits
        # before occupancy drops -- the lever behind the nfields sensitivity.
        return self.smem_per_sm / _REF_SMEM_PER_SM


@functools.cache
def _caps(index: int) -> "HWCaps":
    return HWCaps(index)


def caps(device=None):
    # Memoize on the RESOLVED index, not on what the caller passed: `caps(None)` means the current
    # device, which changes, so caching under None would serve one device's properties for another
    # -- a wrong launch shape rather than an error. _get_device_index handles every form (None, an
    # index, "cuda:1", a torch.device with or without an index, a tensor's device).
    return _caps(torch.cuda._utils._get_device_index(device, optional=True))
