import torch
from torch.utils._benchmark import FuzzedParameter, FuzzedTensor, Fuzzer, ParameterAlias
from torch.utils._benchmark.op_fuzzers import constants


X_SIZE = "x_size"


class UnaryOpFuzzer(Fuzzer):
    def __init__(
        self,
        seed,
        dtype=torch.float32,
        cuda=False,
        scale=constants.Scale.LARGE,
        dim=None,
        pow_2_fraction=0.2,
        max_elements=32 * 1024 ** 2,
    ):
        assert scale in (
            constants.Scale.SMALL,
            constants.Scale.MEDIUM,
            constants.Scale.LARGE,
        )
        if dim is None:
            dim = {1: 0.3, 2: 0.4, 3: 0.3}
        elif isinstance(dim, int):
            assert dim >= 1
            dim = {dim: 1}
        elif isinstance(dim, dict):
            assert all(isinstance(k, int) and k >= 1 for k in dim.keys())
        maxdim = max(dim.keys())

        super().__init__(
            parameters=[
                # Dimensionality of x. (e.g. 1D, 2D, etc.)
                FuzzedParameter(
                    "dim",
                    distribution=dim,
                    strict=True,
                ),

                # Shapes for `x`.
                #   It is important to test all shapes, however
                #   powers of two are especially important and therefore
                #   warrant special attention. This is done by generating
                #   both a value drawn from all integers between the min and
                #   max allowed values, and another from only the powers of two
                #   (both distributions are loguniform) and then randomly
                #   selecting between the two.
                [
                    FuzzedParameter(
                        name=f"k_any_{i}",
                        minval=constants.MIN_DIM_SIZE,
                        maxval=constants.MAX_DIM_SIZE[scale],
                        distribution="loguniform",
                    )
                    for i in range(maxdim)
                ],
                [
                    FuzzedParameter(
                        name=f"k_pow2_{i}",
                        distribution=constants.pow_2_values(
                            constants.MIN_DIM_SIZE, constants.MAX_DIM_SIZE[scale]
                        ),
                    )
                    for i in range(maxdim)
                ],
                [
                    FuzzedParameter(
                        name=f"k{i}",
                        distribution={
                            ParameterAlias(f"k_any_{i}"): 1 - pow_2_fraction,
                            ParameterAlias(f"k_pow2_{i}"): pow_2_fraction,
                        },
                        strict=True,
                    )
                    for i in range(maxdim)
                ],

                # Steps for `x`. (Benchmarks strided memory access.)
                [
                    FuzzedParameter(
                        name=f"x_step_{i}",
                        distribution={1: 0.8, 2: 0.06, 4: 0.06, 8: 0.04, 16: 0.04},
                    )
                    for i in range(maxdim)
                ],

                # Repeatable entropy for downstream applications.
                FuzzedParameter(
                    name="random_value",
                    minval=0,
                    maxval=2 ** 32 - 1,
                    distribution="uniform",
                ),
            ],
            tensors=[
                FuzzedTensor(
                    name="x",
                    size=tuple(f"k{i}" for i in range(maxdim)),
                    steps=tuple(f"x_step_{i}" for i in range(maxdim)),
                    probability_contiguous=0.75,
                    min_elements=constants.MIN_ELEMENTS[scale],
                    max_elements=max_elements,
                    max_allocation_bytes=2 * 1024 ** 3,  # 2 GB
                    dim_parameter="dim",
                    dtype=dtype,
                    cuda=cuda,
                )
            ],
            seed=seed,
        )

    @staticmethod
    def structure_params(params: dict):
        params = params.copy()
        dim = params.pop("dim")
        params[X_SIZE] = tuple(params.pop(f"k{i}") for i in range(dim))
        return params
