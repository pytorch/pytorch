from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FlyDSLHGemmConfig:
    TILE_M: int
    TILE_N: int = 128
    TILE_K: int = 64
    STAGES: int = 2
    SPLIT_K: int = 1
    BLOCK_M_WARPS: int = 1
    BLOCK_N_WARPS: int = 4
    BLOCK_K_WARPS: int = 1
    B_TO_LDS: bool = True


def get_default_hgemm_configs() -> list[FlyDSLHGemmConfig]:
    return [
        FlyDSLHGemmConfig(TILE_M=32),
        FlyDSLHGemmConfig(TILE_M=64),
        FlyDSLHGemmConfig(TILE_M=128),
    ]


def get_hgemm_configs(m: int, n: int, k: int) -> list[dict[str, object]]:
    configs: list[dict[str, object]] = []
    for config in get_default_hgemm_configs():
        if config.TILE_M > max(128, m):
            continue
        if k % config.SPLIT_K != 0:
            continue
        if (k // config.SPLIT_K) // config.TILE_K < config.STAGES:
            continue
        configs.append(
            {
                **asdict(config),
            }
        )
    return configs
