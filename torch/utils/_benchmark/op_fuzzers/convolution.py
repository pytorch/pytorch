import torch

from torch.utils._benchmark import Fuzzer, FuzzedParameter, ParameterAlias, FuzzedTensor
from torch.utils._benchmark.op_fuzzers import constants


BATCH_LIMITS = {
    constants.Scale.SMALL: 8,
    constants.Scale.MEDIUM: 256,
    constants.Scale.LARGE: 4096,
}


class ConvFuzzer(Fuzzer):
    def __init__(self, seed, dtype=torch.float32, cuda=False, scale=constants.Scale.LARGE, dim=4):
        assert scale in (constants.Scale.SMALL, constants.Scale.MEDIUM, constants.Scale.LARGE)
        assert dim in (3, 4, 5), "Only Conv1/2/3D are supported."
        raise NotImplementedError("TODO: finish.")

        super().__init__(
            parameters=[
                # Batch Size
                FuzzedParameter(name="N_any", minval=2, maxval=BATCH_LIMITS[scale], distribution="loguniform"),
                FuzzedParameter(name="N_pow_2", distribution=constants.pow_2_values(2, BATCH_LIMITS[scale])),
                FuzzedParameter(name="N", distribution={
                    1: 1,  # Batch 1 inference is an especially important use case.
                    ParameterAlias("N_any"): 0.45,
                    ParameterAlias("N_pow_2"): 0.45,
                }),

                # Channels
                #   Channels are generally either:
                #    A) A power of two due to taking as input the output of a prior
                #       convolution with power of two number of channels.
                #    B) Three, due to RGB
                #   As a result, these two are given extra probability.
                FuzzedParameter(name="C_any", minval=1, maxval=1024, distribution="loguniform"),
                FuzzedParameter(name="C_pow_2", distribution=as_normalized_dict(_POW_TWO_SIZES)),
                FuzzedParameter(name="C", distribution={
                    ParameterAlias("C_any"): 0.25,
                    ParameterAlias("C_pow_2"): 0.65,
                    3: 0.1,
                }),

                # H and W
                FuzzedParameter("H_any", minval=7, maxval=500, distribution="loguniform"),
                FuzzedParameter("H_pow2", distribution=as_normalized_dict(_POW_TWO_SIZES, minval=8)),
                FuzzedParameter("H_resnet", distribution=as_normalized_dict(_RESNET_SIZES)),

                FuzzedParameter("W_any", minval=7, maxval=500, distribution="loguniform"),
                FuzzedParameter("W_pow2", distribution=as_normalized_dict(_POW_TWO_SIZES, minval=8)),

                FuzzedParameter("H", distribution={
                    ParameterAlias("H_any"): 0.4,
                    ParameterAlias("H_pow2"): 0.4,
                    ParameterAlias("H_resnet"): 0.2,
                }),

                # Square images are unusually common, so half the time Width simply
                # mirrors height.
                FuzzedParameter("W", distribution={
                    ParameterAlias("H"): 0.5,
                    ParameterAlias("W_any"): 0.25,
                    ParameterAlias("W_pow2"): 0.25,
                }),

                # Output channels
                FuzzedParameter("out_channels_any", minval=4, maxval=1024, distribution="loguniform"),
                FuzzedParameter("out_channels_pow2", distribution=as_normalized_dict(_POW_TWO_SIZES)),
                FuzzedParameter("out_channels", distribution={
                    ParameterAlias("out_channels_any"): 0.5,
                    ParameterAlias("out_channels_pow2"): 0.5,
                }),

                # Kernel sizes and strides
                FuzzedParameter("kernel_H", minval=1, maxval=7, distribution="uniform"),
                FuzzedParameter("kernel_W_candidate", minval=1, maxval=7, distribution="uniform"),
                FuzzedParameter("kernel_W", distribution={
                    ParameterAlias("kernel_H"): 0.5,
                    ParameterAlias("kernel_W_candidate"): 0.5,
                }),

                FuzzedParameter("stride_H", minval=1, maxval=3, distribution="uniform"),
                FuzzedParameter("stride_W_candidate", minval=1, maxval=3, distribution="uniform"),
                FuzzedParameter("stride_W", distribution={
                    ParameterAlias("stride_H"): 0.5,
                    ParameterAlias("stride_W_candidate"): 0.5,
                }),

            ],
            tensors=[
                FuzzedTensor(
                    name="x",
                    size=("N", "C", "H", "W"),
                    # TODO(robieta): steps
                    probability_contiguous=0.8,
                    max_elements=1024 ** 2,
                    dtype=dtype,
                    cuda=False,
                ),
            ],
            seed=seed,
        )
