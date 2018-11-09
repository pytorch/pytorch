import torch
import copy


class QuantizedLinear(torch.jit.ScriptModule):
    __constants__ = ['scale', 'zero_point']

    def __init__(self, other):
        super(QuantizedLinear, self).__init__()
        self.in_features = other.in_features
        self.out_features = other.out_features
        # Quantize weight and discard the original
        self.weight, self.col_offsets, self.scale, self.zero_point = torch.fbgemm_linear_quantize_weight(
            other.weight.clone().float())
        self.weight = torch.nn.Parameter(self.weight, requires_grad=False)
        self.col_offsets = torch.nn.Parameter(self.col_offsets, requires_grad=False)
        assert other.bias is not None, 'QuantizedLinear requires a bias'
        self.bias = torch.nn.Parameter(other.bias.clone().float())

        self.register_buffer(
            'packed_tensor_ptr',
            torch.fbgemm_pack_quantized_matrix(self.weight.clone(), self.weight.size(1), self.weight.size(0)))

    @torch.jit.script_method
    def _unpack(self):
        self.packed_tensor_ptr.set_(
            torch.fbgemm_pack_quantized_matrix(
                self.weight, self.weight.size(1), self.weight.size(0)))

    @torch.jit.script_method
    def _pack(self):
        self.packed_tensor_ptr.set_(
            torch.zeros(torch.jit.annotate(List[int], []), dtype=torch.uint8).detach())

    @torch.jit.script_method
    def forward(self, input):
        out = torch.fbgemm_linear_int8_weight(
            input.float(), self.weight, self.packed_tensor_ptr, self.col_offsets,
            self.scale, self.zero_point, self.bias)
        return out.type_as(input)

    def extra_repr(self):
        repr = 'in_features={in_features}, out_features={out_features}, ' \
               'scale={scale}, zero_point={zero_point}'.format(**self.__dict__)
        return repr


def quantize_linear_modules(module):
    for name, mod in module.named_modules():
        if mod is module:
            continue
        if isinstance(mod, torch.nn.Linear):
            setattr(module, name, QuantizedLinear(mod))
        quantize_linear_modules(mod)
