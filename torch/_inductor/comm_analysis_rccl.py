"""RCCL tuning tables for the analytical collective-runtime estimator.

Data only; consumed by comm_analysis._rccl_algo_time. Transcribed from
RCCL revision bc1c8814476f6feab41d69e97c23bcc4e16f47b2 (rcclTuningModel):
https://github.com/ROCm/rccl/blob/bc1c8814476f6feab41d69e97c23bcc4e16f47b2/src/graph/tuning.cc

RCCL's tables are indexed [algo][proto] over all of NCCL's algorithms; only the
[TREE, RING] x [LL, LL128, SIMPLE] sub-block is kept here, matching the
algo/proto space that comm_analysis models.
"""

from dataclasses import dataclass


RCCL_FACTOR_TABLE_SIZE = 27  # tuning.cc:148


@dataclass(frozen=True)
class RcclTuningModel:
    hwLat: tuple[tuple[tuple[float, ...], ...], ...]  # [hw:3][algo:2][proto:3]
    bwRatio: tuple[tuple[tuple[float, ...], ...], ...]  # [nNodes<=2][algo:2][proto:3]
    treeCorrectionFactor: tuple[tuple[float, ...], ...]  # [proto:3][logSize:27]
    ringCorrectionFactor: tuple[tuple[float, ...], ...]  # [proto:3][logSize:27]


# Correction-factor rows are indexed by
# index = clamp(log2i(nBytes >> 6), 0, 26): 0=64B, 9=32K, 18=16M, 26=4G.

# fmt: off

# tuning.cc:331-375. gfx942 (MI300X).
# The 320 us Ring/Simple network latency is verbatim from RCCL.
TUNING_MODEL_5 = RcclTuningModel(
    hwLat=(
        # NVLINK (XGMI): Tree(LL/LL128/Simple), Ring(LL/LL128/Simple)
        ((0.9, 0.9, 2.3), (0.8, 0.8, 2.1)),
        # PCI
        ((2.2, 2.2, 5.7), (2.2, 2.2, 5.7)),
        # NET
        ((10.5, 10.5, 25.0), (9.5, 9.5, 320.0)),
    ),
    bwRatio=(
        # <= 2 nodes
        ((0.06, 0.06, 0.11), (0.08, 0.08, 1.00)),
        # > 2 nodes
        ((0.06, 0.06, 0.59), (0.08, 0.08, 1.00)),
    ),
    treeCorrectionFactor=(
        # LL
        (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0,
         1.0, 1.0, 0.6, 1.0, 0.9, 1.0, 1.0, 0.1, 0.1,
         0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
        # LL128
        (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0,
         1.0, 1.0, 0.6, 1.0, 0.9, 1.0, 1.0, 0.1, 0.1,
         0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
        # SIMPLE
        (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
         0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.7, 1.0, 1.0,
         1.0, 1.0, 1.0, 0.7, 0.7, 0.5, 0.6, 0.6, 0.6),
    ),
    ringCorrectionFactor=(
        # LL
        (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
         0.1, 0.1, 1.0, 0.2, 1.0, 0.4, 0.4, 0.1, 0.2,
         0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
        # LL128
        (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
         0.1, 0.1, 1.0, 0.2, 1.0, 0.4, 0.4, 0.1, 0.2,
         0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
        # SIMPLE
        (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
         0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
         0.1, 0.1, 0.1, 0.1, 0.7, 0.2, 1.0, 1.0, 1.0),
    ),
)

# tuning.cc:377-425. gfx950 (MI355X).
# hwLat and bwRatio match model 5; only the correction tables differ.
TUNING_MODEL_6 = RcclTuningModel(
    hwLat=TUNING_MODEL_5.hwLat,
    bwRatio=TUNING_MODEL_5.bwRatio,
    treeCorrectionFactor=(
        # LL
        (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0,
         1.0, 1.0, 0.6, 1.0, 0.9, 1.0, 1.0, 0.1, 0.1,
         0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
        # LL128
        (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0,
         1.0, 1.0, 0.6, 1.0, 0.9, 1.0, 1.0, 1.0, 1.0,
         1.0, 2.0, 2.0, 1.0, 0.1, 0.9, 0.9, 0.1, 0.1),
        # SIMPLE
        (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
         0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.7, 1.0, 0.1,
         0.1, 0.1, 0.1, 0.1, 0.7, 0.15, 0.6, 0.7, 0.6),
    ),
    ringCorrectionFactor=(
        # LL
        (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
         0.1, 0.1, 1.0, 0.2, 1.0, 0.4, 0.4, 0.1, 0.2,
         0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
        # LL128
        (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
         0.1, 0.1, 1.0, 0.2, 1.0, 0.4, 0.4, 0.1, 0.2,
         0.1, 0.1, 0.1, 0.1, 5.5, 0.1, 0.1, 1.0, 1.0),
        # SIMPLE
        (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
         0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
         0.1, 0.1, 0.9, 0.1, 1.0, 0.1, 0.6, 1.0, 1.0),
    ),
)

# fmt: on

# rcclGetTuningIndexForArch maps these datacenter architectures to models 5/6.
RCCL_TUNING_MODELS: dict[str, RcclTuningModel] = {
    "gfx942": TUNING_MODEL_5,
    "gfx950": TUNING_MODEL_6,
}

# Aggregate XGMI bus bandwidth per GPU (GB/s), assuming seven active peer
# links on a standard 8-GPU platform. RCCL's topo.h at the revision pinned
# above uses 48 GB/s per link for both gfx942 and gfx950: 7 * 48 = 336 GB/s.
# This approximates topology/channel effects; get_intra_node_bw scales by
# active peers / 7 for smaller groups.
RCCL_XGMI_BW: dict[str, float] = {
    "gfx942": 336.0,
    "gfx950": 336.0,
}

# All-to-all latency per arch: (fixed_us, per_rank_us). Total modelled latency
# is fixed_us + per_rank_us * group_size, plus the architecture-independent
# _RCCL_LAUNCH_OVERHEAD_US applied in comm_analysis.
#
# The original model had only a per-rank term. Measured implied latency was
# roughly flat in group size (~28/29/31us at n=2/4/8 on MI350X), so the fixed
# part dominates and its absence was the single largest all-to-all error.
#
# Caveat on the split between the two terms: with only three group sizes they
# are weakly separable, and holding out n=8 moves the fitted pair
# substantially while barely changing the fit quality. Values are rounded to
# the middle of each flat optimum rather than taken raw, and the pair should
# be re-fitted together rather than either constant adjusted alone.
#
# Measured 2026-08-19: median |log| error 0.062 (gfx950), 0.064 (gfx942).
RCCL_ALL_TO_ALL_LAT_US: dict[str, tuple[float, float]] = {
    "gfx942": (18.0, 1.0),
    "gfx950": (15.5, 0.7),
}
