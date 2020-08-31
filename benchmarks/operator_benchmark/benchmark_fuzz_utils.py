"""Adapter between op fuzzers and microbenchmarks.
"""
import enum
import hashlib
from typing import Union

from torch.utils._benchmark.op_fuzzers import binary


class Fuzzers(enum.Enum):
    BINARY = 1


class Scale(enum.Enum):
    SMALL = 0
    MEDIUM = 1
    LARGE = 2


def fuzzer_factory(fuzzer: Fuzzers, scale: Scale, seed=0: Union[int, str]):
    # Convenience wrapper to easily allow different seeds for different ops
    # while keeping the source code readable.
    if isinstance(seed, str):
        m = hashlib.sha256()
        m.update(seed.encode("utf-8"))
        seed = int(m.hexdigest(), 16) % 2**63

    if fuzzer == Fuzzers.BINARY:
        scale = {
            Scale.SMALL: binary.SMALL,
            Scale.MEDIUM: binary.MEDIUM,
            Scale.LARGE: binary.LARGE
        }[scale]



def make_fuzzed_config(
    scale=Scale.SMALL,
    n: int = 10,
    seed: int = 0,
    cross_product_configs=None,
    tags=None,
    checksum=None
):
    fuzzer = binary.BinaryOpFuzzer(seed=seed, scale=scale)
    attr_names = [binary.X_SIZE, binary.Y_SIZE]
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

    return op_bench.config_list(
        attr_names=[a.upper() for a in attr_names],
        attrs=attrs,
        cross_product_configs=cross_product_configs or {},
        tags=tags or [],
    )
