from collections.abc import Callable
from typing import Any

import torch


def setup_baseline():
    from torchao.quantization.utils import recommended_inductor_config_setter

    recommended_inductor_config_setter()
    torch._dynamo.config.automatic_dynamic_shapes = False
    torch._dynamo.config.recompile_limit = 10000


def torchao_optimize_ctx(quantization: str):
    from torchao.quantization.quant_api import (
        int4_weight_only,
        int8_dynamic_activation_int8_weight,
        int8_weight_only,
        quantize_,
    )
    from torchao.utils import unwrap_tensor_subclass

    def inner(model_iter_fn: Callable):
        def _torchao_apply(module: torch.nn.Module, example_inputs: Any):
            if getattr(module, "_quantized", None) is None:
                if quantization == "int8dynamic":
                    quantize_(
                        module,
                        int8_dynamic_activation_int8_weight(),
                        set_inductor_config=False,
                    )
                elif quantization == "int8weightonly":
                    quantize_(module, int8_weight_only(), set_inductor_config=False)
                elif quantization == "int4weightonly":
                    quantize_(module, int4_weight_only(), set_inductor_config=False)
                unwrap_tensor_subclass(module)
                setattr(module, "_quantized", True)  # noqa: B010
            model_iter_fn(module, example_inputs)

        return _torchao_apply

    return inner
