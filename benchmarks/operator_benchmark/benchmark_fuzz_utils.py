"""Adapter between op fuzzers and microbenchmarks.
"""
import enum
import hashlib
from typing import Optional, Union

from torch.utils._benchmark.op_fuzzers.constants import Scale
from torch.utils._benchmark.op_fuzzers import binary, matmul, unary

import benchmark_utils


class Fuzzers(enum.Enum):
    UNARY = 0
    BINARY = 1
    MATMUL = 2
    BATCH_MATMUL = 3


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

    raise NotImplementedError(f"Unknown fuzzer.")


def make_fuzzed_config(
    fuzzer: Fuzzers,
    scale=Scale.SMALL,
    n: int = 10,
    seed: Union[int, str] = 0,
    fuzzer_kwargs: Optional[dict]=None,
    cross_product_configs=None,
    tags=None,
    checksum: Optional[int] = None
):
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
            total += sum(sum(i) for i in a)
        if total != checksum:
            raise ValueError(f"Checksum failed: Total {total} != {checksum}")

    return benchmark_utils.config_list(
        attr_names=[a.upper() for a in attr_names],
        attrs=attrs,
        cross_product_configs=cross_product_configs or {},
        tags=tags or [],
    )
