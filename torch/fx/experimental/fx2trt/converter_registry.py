from typing import Callable, Dict, Any

from torch.fx.node import Target


CONVERTERS: Dict[Target, Any] = {}
NO_IMPLICIT_BATCH_DIM_SUPPORT = {}
NO_EXPLICIT_BATCH_DIM_SUPPORT = {}


def tensorrt_converter(
    key: Target,
    no_implicit_batch_dim: bool = False,
    no_explicit_batch_dim: bool = False,
    enabled: bool = True
) -> Callable[[Any], Any]:
    def register_converter(converter):
        CONVERTERS[key] = converter
        if no_implicit_batch_dim:
            NO_IMPLICIT_BATCH_DIM_SUPPORT[key] = converter
        if no_explicit_batch_dim:
            NO_EXPLICIT_BATCH_DIM_SUPPORT[key] = converter
        return converter

    def disable_converter(converter):
        return converter

    if enabled:
        return register_converter
    else:
        return disable_converter
