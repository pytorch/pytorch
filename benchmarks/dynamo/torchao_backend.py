from typing import Any, Callable

import torch


def setup_baseline():
    torch._dynamo.epilogue_fusion = False
    torch._dynamo.config.automatic_dynamic_shapes = False
    torch._dynamo.config.force_parameter_static_shapes = False
    torch._dynamo.config.cache_size_limit = 10000
    torch._inductor.config.force_fuse_int_mm_with_mul = True
    torch._inductor.config.use_mixed_mm = True


def torchao_optimize_ctx(quantization: str):
    import torchao
    from torchao.quantization.quant_api import (
        change_linear_weights_to_int4_woqtensors,
        change_linear_weights_to_int8_dqtensors,
        change_linear_weights_to_int8_woqtensors,
    )

    def inner(model_iter_fn: Callable):
        def _torchao_apply(module: torch.nn.Module, example_inputs: Any):
            if getattr(module, "_quantized", None) is None:
                if quantization == "int8dynamic":
                    change_linear_weights_to_int8_dqtensors(module)
                elif quantization == "int8weightonly":
                    change_linear_weights_to_int8_woqtensors(module)
                elif quantization == "int4weightonly":
                    change_linear_weights_to_int4_woqtensors(module)
                elif quantization == "autoquant":
                    torchao.autoquant(module, error_on_unseen=False)
                    if isinstance(example_inputs, dict):
                        module(**example_inputs)
                    else:
                        module(*example_inputs)
                    from torchao.quantization.autoquant import AUTOQUANT_CACHE

                    assert (
                        len(AUTOQUANT_CACHE) > 0
                    ), f"Err: found no autoquantizable layers in model {type(module)}, stopping autoquantization"
                elif quantization == "noquant":
                    pass
                else:
                    raise AssertionError(
                        f"Unsupposed quantization mode {quantization}."
                    )
                setattr(module, "_quantized", True)  # noqa: B010
            model_iter_fn(module, example_inputs)

        return _torchao_apply

    return inner
