import torch
from torch.utils._benchmark import FuzzedParameter, FuzzedTensor, Fuzzer, ParameterAlias
from torch.utils._benchmark.op_fuzzers import constants


X_SIZE = "x_size"
Y_SIZE = "y_size"


class BinaryOpFuzzer(Fuzzer):
    def __init__(
        self,
        seed,
        dtype=torch.float32,
        cuda=False,
        scale=constants.Scale.LARGE,
        dim=None,
    ):
        assert scale in (
            constants.Scale.SMALL,
            constants.Scale.MEDIUM,
            constants.Scale.LARGE,
        )
        super().__init__(
            parameters=[
                # Dimensionality of x and y. (e.g. 1D, 2D, or 3D.)
                FuzzedParameter(
                    "dim",
                    distribution={1: 0.3, 2: 0.4, 3: 0.3} if dim is None else {dim: 1},
                    strict=True,
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
                        minval=constants.MIN_DIM_SIZE,
                        maxval=constants.MAX_DIM_SIZE[scale],
                        distribution="loguniform",
                    )
                    for i in range(3)
                ],
                [
                    FuzzedParameter(
                        name=f"k_pow2_{i}",
                        distribution=constants.pow_2_values(
                            constants.MIN_DIM_SIZE, constants.MAX_DIM_SIZE[scale]
                        ),
                    )
                    for i in range(3)
                ],
                [
                    FuzzedParameter(
                        name=f"k{i}",
                        distribution={
                            ParameterAlias(f"k_any_{i}"): 0.8,
                            ParameterAlias(f"k_pow2_{i}"): 0.2,
                        },
                        strict=True,
                    )
                    for i in range(3)
                ],
                [
                    FuzzedParameter(
                        name=f"y_k{i}",
                        distribution={ParameterAlias(f"k{i}"): 0.8, 1: 0.2},
                        strict=True,
                    )
                    for i in range(3)
                ],

                # Steps for `x` and `y`. (Benchmarks strided memory access.)
                [
                    FuzzedParameter(
                        name=f"{name}_step_{i}",
                        distribution={1: 0.8, 2: 0.06, 4: 0.06, 8: 0.04, 16: 0.04},
                    )
                    for i in range(3)
                    for name in ("x", "y")
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
                    size=("k0", "k1", "k2"),
                    steps=("x_step_0", "x_step_1", "x_step_2"),
                    probability_contiguous=0.75,
                    min_elements=constants.MIN_ELEMENTS[scale],
                    max_elements=32 * 1024 ** 2,
                    max_allocation_bytes=2 * 1024 ** 3,  # 2 GB
                    dim_parameter="dim",
                    dtype=dtype,
                    cuda=cuda,
                ),
                FuzzedTensor(
                    name="y",
                    size=("y_k0", "y_k1", "y_k2"),
                    steps=("y_step_0", "y_step_1", "y_step_2"),
                    probability_contiguous=0.75,
                    max_allocation_bytes=2 * 1024 ** 3,  # 2 GB
                    dim_parameter="dim",
                    dtype=dtype,
                    cuda=cuda,
                ),
            ],
            seed=seed,
        )

    @staticmethod
    def structure_params(params: dict):
        params = params.copy()
        dim = params.pop("dim")
        params[X_SIZE] = tuple(params.pop(i) for i in ("k0", "k1", "k2"))[:dim]
        params[Y_SIZE] = tuple(params.pop(i) for i in ("y_k0", "y_k1", "y_k2"))[:dim]
        return params
