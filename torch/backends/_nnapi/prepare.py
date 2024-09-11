# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
from typing import List, Optional

import torch
from torch.backends._nnapi.serializer import _NnapiSerializer


ANEURALNETWORKS_PREFER_LOW_POWER = 0
ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER = 1
ANEURALNETWORKS_PREFER_SUSTAINED_SPEED = 2


class NnapiModule(torch.nn.Module):
    """Torch Module that wraps an NNAPI Compilation.

    This module handles preparing the weights, initializing the
    NNAPI TorchBind object, and adjusting the memory formats
    of all inputs and outputs.
    """

    # _nnapi.Compilation is defined
    comp: Optional[torch.classes._nnapi.Compilation]  # type: ignore[name-defined]
    weights: List[torch.Tensor]
    out_templates: List[torch.Tensor]

    def __init__(
        self,
        shape_compute_module: torch.nn.Module,
        ser_model: torch.Tensor,
        weights: List[torch.Tensor],
        inp_mem_fmts: List[int],
        out_mem_fmts: List[int],
        compilation_preference: int,
        relax_f32_to_f16: bool,
    ):
        super().__init__()
        self.shape_compute_module = shape_compute_module
        self.ser_model = ser_model
        self.weights = weights
        self.inp_mem_fmts = inp_mem_fmts
        self.out_mem_fmts = out_mem_fmts
        self.out_templates = []
        self.comp = None
        self.compilation_preference = compilation_preference
        self.relax_f32_to_f16 = relax_f32_to_f16

    @torch.jit.export
    def init(self, args: List[torch.Tensor]):
        assert self.comp is None
        self.out_templates = self.shape_compute_module.prepare(self.ser_model, args)  # type: ignore[operator]
        self.weights = [w.contiguous() for w in self.weights]
        comp = torch.classes._nnapi.Compilation()
        comp.init2(
            self.ser_model,
            self.weights,
            self.compilation_preference,
            self.relax_f32_to_f16,
        )

        self.comp = comp

    def forward(self, args: List[torch.Tensor]) -> List[torch.Tensor]:
        if self.comp is None:
            self.init(args)
        comp = self.comp
        assert comp is not None
        outs = [torch.empty_like(out) for out in self.out_templates]

        assert len(args) == len(self.inp_mem_fmts)
        fixed_args = []
        for idx in range(len(args)):
            fmt = self.inp_mem_fmts[idx]
            # These constants match the values in DimOrder in serializer.py
            # TODO: See if it's possible to use those directly.
            if fmt == 0:
                fixed_args.append(args[idx].contiguous())
            elif fmt == 1:
                fixed_args.append(args[idx].permute(0, 2, 3, 1).contiguous())
            else:
                raise ValueError("Invalid mem_fmt")
        comp.run(fixed_args, outs)
        assert len(outs) == len(self.out_mem_fmts)
        for idx in range(len(self.out_templates)):
            fmt = self.out_mem_fmts[idx]
            # These constants match the values in DimOrder in serializer.py
            # TODO: See if it's possible to use those directly.
            if fmt in (0, 2):
                pass
            elif fmt == 1:
                outs[idx] = outs[idx].permute(0, 3, 1, 2)
            else:
                raise ValueError("Invalid mem_fmt")
        return outs


def convert_model_to_nnapi(
    model,
    inputs,
    serializer=None,
    return_shapes=None,
    use_int16_for_qint16=False,
    compilation_preference=ANEURALNETWORKS_PREFER_SUSTAINED_SPEED,
    relax_f32_to_f16=False,
):
    (
        shape_compute_module,
        ser_model_tensor,
        used_weights,
        inp_mem_fmts,
        out_mem_fmts,
        retval_count,
    ) = process_for_nnapi(
        model, inputs, serializer, return_shapes, use_int16_for_qint16
    )

    nnapi_model = NnapiModule(
        shape_compute_module,
        ser_model_tensor,
        used_weights,
        inp_mem_fmts,
        out_mem_fmts,
        compilation_preference,
        relax_f32_to_f16,
    )

    class NnapiInterfaceWrapper(torch.nn.Module):
        """NNAPI list-ifying and de-list-ifying wrapper.

        NNAPI always expects a list of inputs and provides a list of outputs.
        This module allows us to accept inputs as separate arguments.
        It returns results as either a single tensor or tuple,
        matching the original module.
        """

        def __init__(self, mod):
            super().__init__()
            self.mod = mod

    wrapper_model_py = NnapiInterfaceWrapper(nnapi_model)
    wrapper_model = torch.jit.script(wrapper_model_py)
    # TODO: Maybe make these names match the original.
    arg_list = ", ".join(f"arg_{idx}" for idx in range(len(inputs)))
    if retval_count < 0:
        ret_expr = "retvals[0]"
    else:
        ret_expr = "".join(f"retvals[{idx}], " for idx in range(retval_count))
    wrapper_model.define(
        f"def forward(self, {arg_list}):\n"
        f"    retvals = self.mod([{arg_list}])\n"
        f"    return {ret_expr}\n"
    )
    return wrapper_model


def process_for_nnapi(
    model, inputs, serializer=None, return_shapes=None, use_int16_for_qint16=False
):
    model = torch.jit.freeze(model)

    if isinstance(inputs, torch.Tensor):
        inputs = [inputs]

    serializer = serializer or _NnapiSerializer(
        config=None, use_int16_for_qint16=use_int16_for_qint16
    )
    (
        ser_model,
        used_weights,
        inp_mem_fmts,
        out_mem_fmts,
        shape_compute_lines,
        retval_count,
    ) = serializer.serialize_model(model, inputs, return_shapes)
    ser_model_tensor = torch.tensor(ser_model, dtype=torch.int32)

    # We have to create a new class here every time this function is called
    # because module.define adds a method to the *class*, not the instance.
    class ShapeComputeModule(torch.nn.Module):
        """Code-gen-ed module for tensor shape computation.

        module.prepare will mutate ser_model according to the computed operand
        shapes, based on the shapes of args.  Returns a list of output templates.
        """

    shape_compute_module = torch.jit.script(ShapeComputeModule())
    real_shape_compute_lines = [
        "def prepare(self, ser_model: torch.Tensor, args: List[torch.Tensor]) -> List[torch.Tensor]:\n",
    ] + [f"    {line}\n" for line in shape_compute_lines]
    shape_compute_module.define("".join(real_shape_compute_lines))

    return (
        shape_compute_module,
        ser_model_tensor,
        used_weights,
        inp_mem_fmts,
        out_mem_fmts,
        retval_count,
    )
