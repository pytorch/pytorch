import functools
from collections import defaultdict

import torch
from torch._inductor.codegen.cuda.cuda_env import get_cuda_arch


@functools.cache
def gen_cutlass_presets() -> dict[int, dict[str, list[str]]]:
    """
    Generate cutlass presets for the given CUDA arch.
    """
    presets: dict[int, dict[str, list[str]]] = {}

    if not torch._C._has_cuda:
        return presets

    presets[0] = defaultdict(list)
    arch = get_cuda_arch()
    if arch == "90":
        preset = presets[0]
        preset["0"] = [
            r"cutlass3x_sm90_tensorop_.*_128x128x64_2x1x1_0_.*_align.*_stream_k_warpspecialized_cooperative_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_128x256x64_1x2x1_0_.*_align.*_warpspecialized_cooperative_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_128x256x64_1x2x1_0_.*_align.*_stream_k_warpspecialized_cooperative_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_128x128x64_2x1x1_0_.*_align.*_warpspecialized_cooperative_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_128x128x64_1x2x1_0_.*_align.*_stream_k_warpspecialized_cooperative_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_128x256x64_1x2x1_0_.*_align.*_warpspecialized_cooperative_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_128x128x64_2x1x1_0_.*_align.*_warpspecialized_pingpong_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_64x256x64_1x2x1_0_.*_align.*_warpspecialized_pingpong_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_256x128x64_2x1x1_0_.*_align.*_warpspecialized_cooperative_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_128x256x64_2x1x1_0_.*_align.*_warpspecialized_cooperative_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_128x128x64_1x2x1_0_.*_align.*_warpspecialized_pingpong_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_128x256x64_2x1x1_0_.*_align.*_stream_k_warpspecialized_cooperative_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_128x128x64_1x2x1_0_.*_align.*_warpspecialized_cooperative_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_128x256x64_1x2x1_0_.*_align.*_stream_k_warpspecialized_cooperative_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_128x256x64_2x1x1_0_.*_align.*_stream_k_warpspecialized_cooperative_epi_tma",
        ]
        preset["3332"] = [
            r"cutlass3x_sm90_tensorop_.*_64x48x64_1x4x1_0_.*_align.*_warpspecialized_epi_nosmem",
            r"cutlass3x_sm90_tensorop_.*_64x128x64_2x1x1_0_.*_align.*_warpspecialized_pingpong_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_128x128x64_1x2x1_0_.*_align.*_warpspecialized_cooperative_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_64x32x64_1x1x1_0_.*_align.*_cpasync_warpspecialized_epi_nosmem",
            r"cutlass3x_sm90_tensorop_.*_128x128x64_1x4x1_0_.*_align.*_warpspecialized_pingpong_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_128x64x64_1x2x1_0_.*_align.*_warpspecialized_pingpong_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_128x128x64_1x2x1_0_.*_align.*_warpspecialized_cooperative_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_128x64x64_2x1x1_0_.*_align.*_warpspecialized_pingpong_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_64x16x64_2x1x1_0_.*_align.*_warpspecialized_epi_nosmem",
            r"cutlass3x_sm90_tensorop_.*_128x64x64_2x1x1_0_.*_align.*_warpspecialized_pingpong_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_128x256x64_1x2x1_0_.*_align.*_stream_k_warpspecialized_cooperative_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_128x128x64_4x1x1_0_.*_align.*_warpspecialized_cooperative_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_64x32x64_2x1x1_0_.*_align.*_warpspecialized_epi_nosmem",
            r"cutlass3x_sm90_tensorop_.*_128x64x64_1x1x1_0_.*_align.*_warpspecialized_cooperative_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_64x128x64_4x1x1_0_.*_align.*_warpspecialized_pingpong_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_128x128x64_2x1x1_0_.*_align.*_warpspecialized_cooperative_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_128x128x64_2x1x1_0_.*_align.*_warpspecialized_pingpong_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_128x256x64_1x1x1_0_.*_align.*_warpspecialized_cooperative_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_64x16x64_2x2x1_0_.*_align.*_warpspecialized_epi_nosmem",
            r"cutlass3x_sm90_tensorop_.*_64x128x64_1x2x1_0_.*_align.*_warpspecialized_pingpong_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_64x32x64_1x4x1_0_.*_align.*_warpspecialized_epi_nosmem",
            r"cutlass3x_sm90_tensorop_.*_64x16x64_4x1x1_0_.*_align.*_warpspecialized_epi_nosmem",
            r"cutlass3x_sm90_tensorop_.*_128x256x64_1x1x1_0_.*_align.*_warpspecialized_cooperative_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_128x128x64_1x4x1_0_.*_align.*_warpspecialized_pingpong_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_128x64x64_1x2x1_0_.*_align.*_warpspecialized_cooperative_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_128x64x64_1x1x1_0_.*_align.*_warpspecialized_cooperative_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_128x64x64_1x2x1_0_.*_align.*_warpspecialized_pingpong_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_64x16x64_1x2x1_0_.*_align.*_warpspecialized_epi_nosmem",
            r"cutlass3x_sm90_tensorop_.*_256x128x64_1x1x1_0_.*_align.*_stream_k_warpspecialized_cooperative_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_128x64x64_2x1x1_0_.*_align.*_warpspecialized_cooperative_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_256x192x64_1x1x1_0_.*_align.*_warpspecialized_cooperative_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_128x128x64_1x1x1_0_.*_align.*_warpspecialized_pingpong_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_128x256x64_1x1x1_0_.*_align.*_stream_k_warpspecialized_cooperative_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_64x128x64_1x1x1_0_.*_align.*_warpspecialized_pingpong_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_128x128x64_1x2x1_0_.*_align.*_warpspecialized_pingpong_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_256x128x64_1x1x1_0_.*_align.*_warpspecialized_cooperative_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_128x128x64_2x1x1_0_.*_align.*_warpspecialized_pingpong_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_128x64x64_2x1x1_0_.*_align.*_warpspecialized_cooperative_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_128x256x64_1x2x1_0_.*_align.*_warpspecialized_cooperative_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_128x64x64_1x2x1_0_.*_align.*_warpspecialized_cooperative_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_64x16x64_1x4x1_0_.*_align.*_warpspecialized_epi_nosmem",
            r"cutlass3x_sm90_tensorop_.*_64x32x64_2x2x1_0_.*_align.*_warpspecialized_epi_nosmem",
            r"cutlass3x_sm90_tensorop_.*_128x256x64_2x1x1_0_.*_align.*_warpspecialized_cooperative_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_256x192x64_1x2x1_0_.*_align.*_warpspecialized_cooperative_epi_tma",
            r"cutlass3x_sm90_tensorop_.*_64x16x64_1x1x1_0_.*_align.*_warpspecialized_epi_nosmem",
        ]

    return presets
