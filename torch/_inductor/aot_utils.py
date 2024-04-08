# Owner(s): ["module: inductor"]

import torch
import torch._export
import torch._inductor
import torch.export._trace


class _AOTWrapperModule(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def _aot_compile(
    model,
    example_inputs,
    options=None,
    dynamic_shapes=None,
    disable_constraint_solver=False,
):
    if not isinstance(model, torch.nn.Module):
        model = _AOTWrapperModule(model)
    # The exact API is subject to change
    if torch._inductor.config.is_predispatch:
        ep = torch.export._trace._export(
            model, example_inputs, dynamic_shapes=dynamic_shapes, pre_dispatch=True
        )
        gm = ep.module()
    else:
        gm = torch.export._trace._export_to_torch_ir(
            model,
            example_inputs,
            dynamic_shapes=dynamic_shapes,
            disable_constraint_solver=disable_constraint_solver,
            # Disabling this flag, because instead we can rely on the mapping
            # dynamo_flat_name_to_original_fqn which is coming from Dynamo.
            restore_fqn=False,
        )

    with torch.no_grad():
        so_path = torch._inductor.aot_compile(gm, example_inputs, options=options)  # type: ignore[arg-type]

    return so_path
