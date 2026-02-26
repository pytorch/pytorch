# mypy: allow-untyped-defs
import numpy as np
import torch

from torch.utils.benchmark import Fuzzer, FuzzedParameter, ParameterAlias, FuzzedSparseTensor


_MIN_DIM_SIZE = 16
_MAX_DIM_SIZE = 16 * 1024 ** 2
_POW_TWO_SIZES = tuple(2 ** i for i in range(
    int(np.log2(_MIN_DIM_SIZE)),
    int(np.log2(_MAX_DIM_SIZE)) + 1,
))


class BinaryOpSparseFuzzer(Fuzzer):
    def __init__(self, seed, dtype=torch.float32, cuda=False) -> None:
        super().__init__(
            parameters=[
                # Dimensionality of x and y. (e.g. 1D, 2D, or 3D.)
                FuzzedParameter("dim_parameter", distribution={1: 0.3, 2: 0.4, 3: 0.3}, strict=True),
                FuzzedParameter(
                    name="sparse_dim",
                    distribution={1: 0.4, 2: 0.4, 3: 0.2},
                    strict=True
                ),
                # Shapes for `x` and `y`.
                #       It is important to test all shapes, however
                #   powers of two are especially important and therefore
                #   warrant special attention. This is done by generating
                #   both a value drawn from all integers between the min and
                #   max allowed values, and another from only the powers of two
                #   (both distributions are loguniform) and then randomly
                #   selecting between the two.
                #       Moreover, `y` will occasionally have singleton
                #   dimensions in order to test broadcasting.
                [
                    FuzzedParameter(
                        name=f"k_any_{i}",
                        minval=_MIN_DIM_SIZE,
                        maxval=_MAX_DIM_SIZE,
                        distribution="loguniform",
                    ) for i in range(3)
                ],
                [
                    FuzzedParameter(
                        name=f"k_pow2_{i}",
                        distribution={size: 1. / len(_POW_TWO_SIZES) for size in _POW_TWO_SIZES}
                    ) for i in range(3)
                ],
                [
                    FuzzedParameter(
                        name=f"k{i}",
                        distribution={
                            ParameterAlias(f"k_any_{i}"): 0.8,
                            ParameterAlias(f"k_pow2_{i}"): 0.2,
                        },
                        strict=True,
                    ) for i in range(3)
                ],
                [
                    FuzzedParameter(
                        name=f"y_k{i}",
                        distribution={
                            ParameterAlias(f"k{i}"): 1.0},
                        strict=True,
                    ) for i in range(3)
                ],
                FuzzedParameter(
                    name="density",
                    distribution={0.1: 0.4, 0.05: 0.3, 0.01: 0.3},
                ),
                FuzzedParameter(
                    name="coalesced",
                    distribution={True: 0.5, False: 0.5},
                ),
                # Repeatable entropy for downstream applications.
                FuzzedParameter(name="random_value", minval=0, maxval=2 ** 32 - 1, distribution="uniform"),
            ],
            tensors=[
                FuzzedSparseTensor(
                    name="x",
                    size=("k0", "k1", "k2"),
                    dim_parameter="dim_parameter",
                    sparse_dim="sparse_dim",
                    density="density",
                    coalesced="coalesced",
                    min_elements=4 * 1024,
                    max_elements=32 * 1024 ** 2,
                    dtype=dtype,
                    cuda=cuda,
                ),
                FuzzedSparseTensor(
                    name="y",
                    size=("y_k0", "y_k1", "y_k2"),
                    dim_parameter="dim_parameter",
                    sparse_dim="sparse_dim",
                    density="density",
                    coalesced="coalesced",
                    min_elements=4 * 1024,
                    max_elements=32 * 1024 ** 2,
                    dtype=dtype,
                    cuda=cuda,
                ),
            ],
            seed=seed,
        )
