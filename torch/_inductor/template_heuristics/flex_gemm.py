# mypy: allow-untyped-defs
from __future__ import annotations

from dataclasses import fields
from functools import cache
from typing import Any, TypeAlias

import sympy

import torch
import torch._vendor.quack.gemm_config as quack_gemm_config
from torch.utils._ordered_set import OrderedSet


GemmConfigKey: TypeAlias = tuple[tuple[str, Any], ...]


def gemm_config_key(config: quack_gemm_config.GemmConfig) -> GemmConfigKey:
    """Project a QuACK GEMM config using the dataclass schema as the contract."""
    return tuple(
        (field.name, getattr(config, field.name))
        for field in fields(quack_gemm_config.GemmConfig)
    )


@cache
def dense_gemm_config_priority_keys() -> tuple[GemmConfigKey, ...]:
    """Return the measured dense FlexGEMM QuACK preference order."""
    configs = (
        quack_gemm_config.GemmConfig(
            tile_m=128,
            tile_n=256,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=2,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=128,
            tile_n=192,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=2,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=256,
            tile_n=256,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=2,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=256,
            tile_n=256,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=2,
            cluster_n=2,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=256,
            tile_n=192,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=2,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=128,
            tile_n=128,
            pingpong=False,
            is_dynamic_persistent=False,
            cluster_m=1,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=128,
            tile_n=256,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=1,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=128,
            tile_n=256,
            pingpong=False,
            is_dynamic_persistent=False,
            cluster_m=1,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=128,
            tile_n=128,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=2,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=256,
            tile_n=128,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=2,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=128,
            tile_n=224,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=1,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=128,
            tile_n=160,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=1,
            device_capacity=10,
        ),
    )
    return tuple(gemm_config_key(config) for config in configs)


def candidate_gemm_configs_for_device(device: torch.device):
    """Return all device-compatible QuACK configs before shape-specific ranking."""
    device_capacity = torch.cuda.get_device_capability(device)[0]
    if device_capacity == 11:
        device_capacity = 10
    priority_map = {
        key: priority for priority, key in enumerate(dense_gemm_config_priority_keys())
    }
    configs = sorted(
        (
            config
            for config in quack_gemm_config.get_all_configs()
            if config.device_capacity == device_capacity
            and not config.swap_ab
            and config.cluster_k == 1
            and not config.use_tma_gather
        ),
        key=lambda config: (
            priority_map.get(gemm_config_key(config), len(priority_map)),
            config.tile_m,
            config.tile_n,
            config.cluster_m,
            config.cluster_n,
            int(config.is_dynamic_persistent),
        ),
    )
    if not configs:
        raise RuntimeError(
            f"FlexGEMM found no QuACK configs for CUDA device capability "
            f"SM{device_capacity}0"
        )
    return configs


def default_gemm_config_key(device: torch.device, m, n) -> GemmConfigKey:
    """Return the untuned default QuACK config key for generated code."""
    configs = candidate_gemm_configs_for_device(device)
    config_keys = OrderedSet([gemm_config_key(config) for config in configs])
    default_key, skinny_key, large_rect_key, large_key = (
        dense_gemm_config_priority_keys()[:4]
    )

    from torch._inductor.virtualized import V

    guard_or_false = V.graph.sizevars.guard_or_false
    if guard_or_false(sympy.Le(m, n)):
        min_dim, max_dim = m, n
    elif guard_or_false(sympy.Lt(n, m)):
        min_dim, max_dim = n, m
    else:
        return (
            default_key if default_key in config_keys else gemm_config_key(configs[0])
        )

    if guard_or_false(sympy.Lt(min_dim, 512)):
        preferred_keys = (skinny_key, default_key)
    elif guard_or_false(sympy.And(sympy.Eq(min_dim, 1024), sympy.Eq(max_dim, 1024))):
        preferred_keys = (skinny_key, default_key)
    elif guard_or_false(
        sympy.And(
            sympy.Ge(max_dim, 4096), sympy.Ge(min_dim, 768), sympy.Lt(min_dim, 1024)
        )
    ):
        preferred_keys = (large_key, default_key)
    elif guard_or_false(sympy.And(sympy.Ge(max_dim, 4096), sympy.Eq(min_dim, 1024))):
        preferred_keys = (large_rect_key, default_key)
    elif guard_or_false(sympy.Ge(min_dim, 2048)):
        preferred_keys = (large_key, default_key)
    else:
        preferred_keys = (default_key,)

    for key in preferred_keys:
        if key in config_keys:
            return key
    return gemm_config_key(configs[0])
