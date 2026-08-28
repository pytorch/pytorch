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

# Reference GPU the launch-heuristic constants were TUNED on (B200 / sm_100a). The
# heuristics keep their measured anchor numbers but multiply size/occupancy thresholds
# by a hardware-scale RATIO against this reference, so on B200 every ratio is 1.0 (the
# anchors reproduce exactly, no regression) and on other GPUs (Hopper, Rubin, ...) the
# same formula adapts by device fill capacity + smem. Update only if we re-tune on a
# different reference GPU. Values are B200's device properties.
# These are B200's EXACT device-property values (read from get_device_properties on
# sm_100a), so every *_scale ratio is precisely 1.0 on B200 and the anchor heuristics
# reproduce byte-identically. Do not round -- e.g. smem is 233472 B, not 228*1024.
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


def caps(device=None):
    # Key on the DEVICE INDEX, resolving whatever form the caller passed (an index,
    # torch.device, "cuda:1", or a tensor's device). Falling back to current_device() for
    # anything non-int silently returned ANOTHER device's properties, which is a wrong launch
    # shape rather than an error.
    idx = torch.cuda.current_device() if device is None else torch.device(device).index
    if idx is None:  # torch.device("cuda") carries no index: that means the current one
        idx = torch.cuda.current_device()
    return cached_plan(_CACHE, idx, lambda: HWCaps(idx))
