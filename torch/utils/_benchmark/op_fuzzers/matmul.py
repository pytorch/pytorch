import torch

from torch.utils._benchmark import Fuzzer, FuzzedParameter, ParameterAlias, FuzzedTensor
from torch.utils._benchmark.op_fuzzers import constants


X_SIZE = "x_size"
Y_SIZE = "y_size"
BATCH_LIMITS = {
    constants.Scale.SMALL: 8,
    constants.Scale.MEDIUM: 256,
    constants.Scale.LARGE: 2048,
    constants.Scale.LARGER: 32 * 1024,
}
ROOFLINE_WORK_LIMITS = {
    constants.Scale.SMALL: 1024 ** 2,
    constants.Scale.MEDIUM: 512 * 1024 ** 2,
    constants.Scale.LARGE: 1024 ** 3,
    constants.Scale.LARGER: 4 * 1024 ** 3,
}
WORK_LOWER_BOUNDS = {
    constants.Scale.SMALL: 0,
    constants.Scale.MEDIUM: 0,
    constants.Scale.LARGE: ROOFLINE_WORK_LIMITS[constants.Scale.SMALL],
    constants.Scale.LARGER: ROOFLINE_WORK_LIMITS[constants.Scale.MEDIUM]
}


class MatMulFuzzer(Fuzzer):
    batch = False
    def __init__(self, seed, dtype=torch.float32, cuda=False, scale=constants.Scale.LARGE):
        assert scale in (constants.Scale.SMALL, constants.Scale.MEDIUM, constants.Scale.LARGE, constants.Scale.LARGER)
        if self.batch:
            batch_params = [
                FuzzedParameter(
                    name=f"B_any",
                    minval=1,
                    maxval=BATCH_LIMITS[scale],
                    distribution="loguniform",
                    strict=True,
                ),
                FuzzedParameter(
                    name=f"B_pow_2",
                    distribution=constants.pow_2_values(1, BATCH_LIMITS[scale]),
                    strict=True,
                ),
                FuzzedParameter(name="B", distribution={
                    ParameterAlias("B_any"): 0.5,
                    ParameterAlias("B_pow_2"): 0.5,
                }, strict=True),
            ]
            x_size = ("B", "K0", "K1")
            y_size = ("B", "K1", "K2")
        else:
            batch_params = []
            x_size = ("K0", "K1")
            y_size = ("K1", "K2")

        super().__init__(
            parameters=[
                batch_params,
                [
                    FuzzedParameter(name=f"K{i}_any", minval=1, maxval=2048, distribution="loguniform")
                    for i in range(3)
                ],
                [
                    FuzzedParameter(
                        name=f"K{i}_pow_2",
                        distribution=constants.pow_2_values(
                            constants.MIN_DIM_SIZE, constants.MAX_DIM_SIZE[scale])
                    )
                    for i in range(3)
                ],

                FuzzedParameter(name="K0", distribution={
                    ParameterAlias("K0_any"): 0.5,
                    ParameterAlias("K0_pow_2"): 0.5,
                }),

                # Square matricies are somewhat common, so we sometimes
                # alias K1 and K2 to other dims.
                FuzzedParameter(name="K1", distribution={
                    ParameterAlias("K1_any"): 0.3,
                    ParameterAlias("K1_pow_2"): 0.4,
                    ParameterAlias("K0"): 0.3,
                }),

                FuzzedParameter(name="K2", distribution={
                    ParameterAlias("K2_any"): 0.3,
                    ParameterAlias("K2_pow_2"): 0.4,
                    ParameterAlias("K0"): 0.15,
                    ParameterAlias("K1"): 0.15,
                }),

            ],
            tensors=[
                FuzzedTensor(
                    name="x",
                    size=x_size,
                    probability_contiguous=1,
                    max_elements=constants.POINTWISE_MAX_ELEMENTS[scale],
                    dtype=dtype,
                    cuda=False,
                ),
                FuzzedTensor(
                    name="y",
                    size=y_size,
                    probability_contiguous=1,
                    max_elements=constants.POINTWISE_MAX_ELEMENTS[scale],
                    dtype=dtype,
                    cuda=False,
                ),
            ],
            constraints=[
                lambda params: (
                    WORK_LOWER_BOUNDS[scale]
                    <= params.get("B", 1) * params["K0"] * params["K1"] * params["K2"]
                    <= ROOFLINE_WORK_LIMITS[scale]),
            ],
            seed=seed,
        )

    @staticmethod
    def structure_params(params: dict):
        params = params.copy()
        B = ("B",) if "B" in params else ()
        params[X_SIZE] = tuple(params[i] for i in B + ("K0", "K1"))
        params[Y_SIZE] = tuple(params[i] for i in B + ("K1", "K2"))
        return params


class BatchMatMulFuzzer(MatMulFuzzer):
    batch = True
