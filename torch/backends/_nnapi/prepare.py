from typing import Optional, List

import torch
from torch.backends._nnapi.serializer import serialize_model

class NnapiModule(torch.nn.Module):
    """Torch Module that wraps an NNAPI Compilation.

    This module handles preparing the weights, initializing the
    NNAPI TorchBind object, and adjusting the memory formats
    of all inputs and outputs.
    """

    comp: Optional[torch.classes._nnapi.Compilation]
    weights: List[torch.Tensor]

    def __init__(
            self,
            ser_model: torch.Tensor,
            weights: List[torch.Tensor],
            inp_mem_fmts: List[int],
            out_mem_fmts: List[int],
            out_templates: List[torch.Tensor]):
        super().__init__()
        self.ser_model = ser_model
        self.weights = weights
        self.inp_mem_fmts = inp_mem_fmts
        self.out_mem_fmts = out_mem_fmts
        self.out_templates = out_templates
        self.comp = None

    @torch.jit.export
    def init(self):
        assert self.comp is None
        self.weights = [w.contiguous() for w in self.weights]
        comp = torch.classes._nnapi.Compilation()
        comp.init(self.ser_model, self.weights)
        self.comp = comp

    def forward(self, args: List[torch.Tensor]) -> List[torch.Tensor]:
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
                raise Exception("Invalid mem_fmt")
        comp.run(fixed_args, outs)
        assert len(outs) == len(self.out_mem_fmts)
        for idx in range(len(self.out_templates)):
            fmt = self.out_mem_fmts[idx]
            # These constants match the values in DimOrder in serializer.py
            # TODO: See if it's possible to use those directly.
            if fmt == 0:
                pass
            elif fmt == 1:
                outs[idx] = outs[idx].permute(0, 3, 1, 2)
            else:
                raise Exception("Invalid mem_fmt")
        return outs


class NnapiInitWrapper(torch.nn.Module):
    """Wrapper module to ensure NNAPI init is called."""
    def __init__(self, nnapi_module):
        super().__init__()
        self.nnapi_module = nnapi_module

    def forward(self, args: List[torch.Tensor]) -> List[torch.Tensor]:
        return self.nnapi_module(args)

    @torch.jit.export
    def __getstate__(self):
        return self.nnapi_module

    @torch.jit.export
    def __setstate__(self, nnapi_module):
        self.training = False
        self.nnapi_module = nnapi_module
        self.nnapi_module.init()


def _condensed_zeros_like(t):
    """Get a small-storage deterministic tensor with the same shape and dtype as t

    Similar to `torch.zeros(1, dtype=out.dtype).expand(out.shape)`,
    but this works with quantized dtypes as well.

    Similar to `torch.empty(1, dtype=out.dtype).expand(out.shape)`,
    but always returns the same data.
    """

    ret = torch.empty_like(t).flatten()[1].clone().expand(t.shape)
    assert ret.storage().size() == 1
    ret.storage()[0] = 0
    return ret


def convert_model_to_nnapi(model, inputs):
    model = torch.jit.freeze(model)

    if isinstance(inputs, torch.Tensor):
        inputs = [inputs]

    outputs = model(*inputs)

    if isinstance(outputs, torch.Tensor):
        outputs = [outputs]

    ser_model, used_weights, inp_mem_fmts, out_mem_fmts, retval_count = serialize_model(model, inputs)
    ser_model_tensor = torch.tensor(list(ser_model), dtype=torch.uint8)

    out_templates = [_condensed_zeros_like(out) for out in outputs]
    nnapi_model = NnapiInitWrapper(NnapiModule(
        ser_model_tensor,
        used_weights,
        inp_mem_fmts,
        out_mem_fmts,
        out_templates))

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
