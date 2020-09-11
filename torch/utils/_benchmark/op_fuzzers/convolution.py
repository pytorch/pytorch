import torch

from torch.utils._benchmark import Fuzzer, FuzzedParameter, ParameterAlias, FuzzedTensor
from torch.utils._benchmark.op_fuzzers import constants


X_SIZE = "x_size"
KERNEL_SIZE = "kernel_size"
STRIDE = "stride"
C_OUT = "C_out"
GROUPS = "groups"

BATCH_LIMITS = {
    constants.Scale.SMALL: 8,
    constants.Scale.MEDIUM: 256,
    constants.Scale.LARGE: 1024,
    constants.Scale.LARGER: 4096,
}
CHANNEL_LIMITS = {
    constants.Scale.SMALL: 8,
    constants.Scale.MEDIUM: 256,
    constants.Scale.LARGE: 1024,
    constants.Scale.LARGER: 2048,
}
SIZE_LIMITS = {
    constants.Scale.SMALL: 64,
    constants.Scale.MEDIUM: 256,
    constants.Scale.LARGE: 512,
    constants.Scale.LARGER: 2048,
}
ROOFLINE_WORK_LIMITS = {
    constants.Scale.SMALL: 1024 ** 2,
    constants.Scale.MEDIUM: 512 * 1024 ** 2,
    constants.Scale.LARGE: 1024 ** 3,
    constants.Scale.LARGER: 4 * 1024 ** 3,
}


def mixed_distribution(name, minval, maxval, distribution=None):
    k_any, k_pow_2 = f"{name}_any", f"{name}_pow_2"
    distribution = distribution or {k_any: 0.5, k_pow_2: 0.5}
    assert k_any in distribution and k_pow_2 in distribution
    return [
        FuzzedParameter(name=k_any, minval=minval, maxval=maxval, distribution="loguniform"),
        FuzzedParameter(name=k_pow_2, distribution=constants.pow_2_values(minval, maxval)),
        FuzzedParameter(name=name, distribution={
            ParameterAlias(k) if isinstance(k, str) else k: v
            for k, v in distribution.items()
        })
    ]


class ConvFuzzer(Fuzzer):
    def __init__(self, seed, dtype=torch.float32, cuda=False, scale=constants.Scale.LARGE, dim=4, groups=None):
        assert scale in (constants.Scale.SMALL, constants.Scale.MEDIUM, constants.Scale.LARGE, constants.Scale.LARGER)
        assert dim in (3, 4, 5), "Only Conv1/2/3D are supported."

        l_params = mixed_distribution("L", 4, SIZE_LIMITS[scale])
        hw_params = (
            mixed_distribution("H", 4, SIZE_LIMITS[scale]) +

            # Square images are particularly common, so half the time Width
            # simply mirrors height.
            mixed_distribution(
                "W", 4, SIZE_LIMITS[scale],
                {"H": 0.5, "W_any": 0.25, "W_pow_2": 0.25})
        )
        dhw_params = mixed_distribution("D", 4, SIZE_LIMITS[scale]) + hw_params

        if dim == 3:
            size = ("N", "C", "L")
            spatial_params = l_params
        if dim == 4:
            size = ("N", "C", "H", "W")
            spatial_params = hw_params
        if dim == 5:
            size = ("N", "C", "D", "H", "W")
            spatial_params = dhw_params

        def kernel_size_constraint(params):
            kernel_size = params["kernel_size"]
            return all(
                kernel_size <= params.get(k, kernel_size)
                for k in ("L", "D", "H", "W"))

        def work_constraint(params):
            work = params[C_OUT]
            for i in size:
                work *= params[i]
            work *= params[KERNEL_SIZE] ** (dim - 2)
            work /= params[STRIDE] ** (dim - 2)
            work /= params.get(GROUPS, 1)
            return work <= ROOFLINE_WORK_LIMITS[scale]

        super().__init__(
            parameters=[
                # Batch Size
                mixed_distribution("N", 2, BATCH_LIMITS[scale], {
                    1: 0.1,  # Batch 1 inference is a particularly important use case.
                    "N_any": 0.45,
                    "N_pow_2": 0.45,
                }),

                # L, HW, or DHW for Conv1/2/3d respectively.
                spatial_params,

                mixed_distribution("C", 1, CHANNEL_LIMITS[scale]),
                mixed_distribution(C_OUT, 1, CHANNEL_LIMITS[scale]),
                FuzzedParameter(KERNEL_SIZE, distribution={1: 0.25, 3: 0.25, 5: 0.25, 7: 0.25}, strict=True),
                FuzzedParameter(STRIDE, distribution={1: 0.9, 2: 0.05, 3: 0.05}),
                [FuzzedParameter(GROUPS, distribution=groups, strict=True)] if groups else [],
            ],
            tensors=[
                FuzzedTensor(
                    name="x",
                    size=size,
                    probability_contiguous=1,
                    max_elements=constants.POINTWISE_MAX_ELEMENTS[scale],
                    dtype=dtype,
                    cuda=False,
                ),
            ],
            constraints=[
                # Ensure all dims are >= kernel_size
                kernel_size_constraint,

                # Ensure groups divides channels
                lambda params: not (
                    params["C"] % params.get(GROUPS, 1) or
                    params[C_OUT] % params.get(GROUPS, 1)),

                # Limit compute based on scale
                work_constraint,
            ],
            seed=seed,
        )

    @staticmethod
    def structure_params(params: dict):
        if "L" in params:
            size = ("N", "C", "L")
        elif "D" in params:
            size = ("N", "C", "D", "H", "W")
        else:
            size = ("N", "C", "H", "W")
        return {
            X_SIZE: tuple(params.pop(k) for k in size),
            KERNEL_SIZE: params[KERNEL_SIZE],
            STRIDE: params[STRIDE],
            C_OUT: params[C_OUT],
            GROUPS: params.get(GROUPS),
        }
