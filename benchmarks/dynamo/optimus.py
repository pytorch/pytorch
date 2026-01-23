import functools

import torch


def get_baseline_ctx(nopython, inductor_compile_mode):
    return functools.partial(
        torch.compile,
        backend="inductor",
        fullgraph=nopython,
        mode=inductor_compile_mode,
    )


def get_optimus_optimize_ctx(config, nopython, inductor_compile_mode):
    if config == "vertical_opt":
        optimus_inductor_config = {
            "pre_grad_fusion_options": {
                "normalization_pass": {},
                "merge_splits_pass": {},
                "split_cat_pass": {},
                "unbind_stack_pass": {},
                "unbind_cat_to_view_pass": {},
            }
        }
    elif config == "horizontal_opt":
        optimus_inductor_config = {
            "pre_grad_fusion_options": {
                "normalization_pass": {},
                "batch_linear": {},
                "batch_layernorm": {},
            },
        }
    elif config == "all":
        optimus_inductor_config = {
            "pre_grad_fusion_options": {
                "normalization_pass": {},
                "batch_linear": {},
                "batch_layernorm": {},
                "merge_splits_pass": {},
                "split_cat_pass": {},
                "unbind_stack_pass": {},
                "unbind_cat_to_view_pass": {},
            },
        }
    else:
        raise RuntimeError(f"Unknown optimus config: {config}")

    def _inner(fn):
        if "pre_grad_fusion_options" in optimus_inductor_config:
            torch._inductor.config.pre_grad_fusion_options = optimus_inductor_config[
                "pre_grad_fusion_options"
            ]
        if "post_grad_fusion_options" in optimus_inductor_config:
            torch._inductor.config.post_grad_fusion_options = optimus_inductor_config[
                "post_grad_fusion_options"
            ]
        return torch.compile(
            fn, backend="inductor", fullgraph=nopython, mode=inductor_compile_mode
        )

    return _inner
