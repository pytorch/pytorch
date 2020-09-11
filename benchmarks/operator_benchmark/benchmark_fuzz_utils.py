"""Adapter between op fuzzers and microbenchmarks.
"""
import enum
import hashlib
from typing import Optional, Tuple, Union

from torch.utils._benchmark.op_fuzzers.constants import Scale
from torch.utils._benchmark.op_fuzzers import binary, convolution, matmul, unary

import benchmark_utils


CPU_MEDIUM_CUDA_LARGE = "cpu_medium_cuda_large"
class Fuzzers(enum.Enum):
    UNARY = 0
    BINARY = 1
    MATMUL = 2
    BATCH_MATMUL = 3
    CONV1D = 4
    CONV2D = 5
    CONV3D = 6


def fuzzer_factory(fuzzer: Fuzzers, scale: Scale, seed: Union[int, str], fuzzer_kwargs: Optional[dict]):
    # Convenience wrapper to easily allow different seeds for different ops
    # while keeping the source code readable.
    if isinstance(seed, str):
        m = hashlib.sha256()
        m.update(seed.encode("utf-8"))
        seed = int(m.hexdigest(), 16) % 2**32

    kwargs = {"seed": seed, "scale": scale}
    kwargs.update(fuzzer_kwargs or {})

    if fuzzer == Fuzzers.UNARY:
        attr_names = (unary.X_SIZE,)
        return unary.UnaryOpFuzzer(**kwargs), attr_names

    if fuzzer == Fuzzers.BINARY:
        attr_names = (binary.X_SIZE, binary.Y_SIZE)
        return binary.BinaryOpFuzzer(**kwargs), attr_names

    if fuzzer == Fuzzers.MATMUL:
        attr_names = (matmul.X_SIZE, matmul.Y_SIZE)
        return matmul.MatMulFuzzer(**kwargs), attr_names

    if fuzzer == Fuzzers.BATCH_MATMUL:
        attr_names = (matmul.X_SIZE, matmul.Y_SIZE)
        return matmul.BatchMatMulFuzzer(**kwargs), attr_names

    if fuzzer in (Fuzzers.CONV1D, Fuzzers.CONV2D, Fuzzers.CONV3D):
        dim = {Fuzzers.CONV1D: 3, Fuzzers.CONV2D: 4, Fuzzers.CONV3D: 5}[fuzzer]
        attr_names = (convolution.X_SIZE, convolution.C_OUT, convolution.KERNEL_SIZE, convolution.STRIDE)
        if "groups" in kwargs:
            attr_names += (convolution.GROUPS,)
        return convolution.ConvFuzzer(dim=dim, **kwargs), attr_names

    raise NotImplementedError("Unknown fuzzer.")


def make_fuzzed_config(
    fuzzer: Fuzzers,
    scale=Scale.SMALL,
    n: int = 10,
    seed: Union[int, str] = 0,
    fuzzer_kwargs: Optional[Union[dict, Tuple[dict]]] = None,
    cross_product_configs = None,
    tags=None,
    checksum: Optional[Union[int, Tuple[int]]] = None
):
    if isinstance(scale, str):
        cpu_cuda = ("cpu", "cuda")
        cross_product_configs = (cross_product_configs or {}).copy()
        if not isinstance(fuzzer_kwargs, tuple):
            fuzzer_kwargs = (fuzzer_kwargs, fuzzer_kwargs)
        assert scale == CPU_MEDIUM_CUDA_LARGE
        assert checksum is None or isinstance(checksum, tuple) and len(checksum) == 2
        assert len(fuzzer_kwargs) == 2
        assert (
            "device" not in cross_product_configs
            or tuple(cross_product_configs.pop("device")) == cpu_cuda)
        result = []
        for device, checksum, fuzzer_kwargs in zip(cpu_cuda, checksum or (None, None), fuzzer_kwargs):
            assert device in ("cpu", "cuda"), f"Invalid device: {device}"
            device_scale = {"cpu": Scale.MEDIUM, "cuda": Scale.LARGE}[device]
            cross_product_configs["device"] = [device]
            result += make_fuzzed_config(
                fuzzer,
                device_scale,
                n,
                seed,
                fuzzer_kwargs,
                cross_product_configs,
                tags,
                checksum
            )
        return result

    fuzzer, attr_names = fuzzer_factory(fuzzer, scale, seed, fuzzer_kwargs)
    attrs = []
    for i in range(n):
        params = fuzzer.structure_params(fuzzer.params[i])
        attrs.append([params[a] for a in attr_names])

    # Because the generated tests depend on Fuzzer for random numbers,
    # it is advisable to use a checksum to ensure that the configurations
    # being benchmarked do not silently change.
    if checksum is not None:
        total = 0
        for a in attrs:
            total += sum(i if isinstance(i, int) else sum(i) for i in a)
        if total != checksum:
            raise ValueError(f"Checksum failed: Total {total} != {checksum}")

    return benchmark_utils.config_list(
        attr_names=[a.upper() for a in attr_names],
        attrs=attrs,
        cross_product_configs=cross_product_configs or {},
        tags=tags or [],
    )
