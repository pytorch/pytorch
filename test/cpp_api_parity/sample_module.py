import torch
from cpp_api_parity import CppArgDeclaration

'''
`SampleModule` is used by `test_cpp_api_parity.py` to test that Python / C++ API
parity test harness works for `torch.nn.Module` subclasses.

When `SampleModule.has_parity` is true, behavior of `reset_parameters` / `forward` /
`backward` is the same as the C++ equivalent.

When `SampleModule.has_parity` is false, behavior of `reset_parameters` / `forward` /
`backward` is different from the C++ equivalent.
'''

class SampleModule(torch.nn.Module):
    def __init__(self, has_parity):
        super(SampleModule, self).__init__()
        self.register_parameter('param', torch.nn.Parameter(torch.empty(3, 4)))
        self.register_buffer('buffer', torch.empty(4, 5))
        self.has_parity = has_parity
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.param.fill_(1)
            self.buffer.fill_(1)
            if not self.has_parity:
                self.param.add_(10)
                self.buffer.add_(10)

    def forward(self, x):
        if not self.has_parity:
            return x + self.param * 4 + 3
        else:
            return x + self.param * 2

SAMPLE_MODULE_CPP_SOURCE = """\n
struct SampleModuleImpl : public torch::nn::Cloneable<SampleModuleImpl> {
  SampleModuleImpl() {
    reset();
  }
  void reset() {
    param = register_parameter2("param", torch::ones({3, 4}));
    buffer = register_buffer("buffer", torch::ones({4, 5}));
  }
  torch::Tensor forward(torch::Tensor x) {
    return x + param * 2;
  }
  torch::Tensor param;
  torch::Tensor buffer;
};

namespace torch {
namespace nn{
TORCH_MODULE(SampleModule);
}
}
"""

module_tests = [
    dict(
        module_name='SampleModule',
        constructor_args=(True, ),
        cpp_constructor_args='',
        input_size=(3, 4),
        desc='has_parity',
        expect_parity_error=False,
    ),
    dict(
        module_name='SampleModule',
        constructor_args=(False, ),
        cpp_constructor_args='',
        input_size=(3, 4),
        desc='no_parity',
        expect_parity_error=True,
    ),
]

module_metadata = dict(
    cpp_forward_arg_declarations=[CppArgDeclaration(arg_type='torch::Tensor', arg_name='x')],
    cpp_source=SAMPLE_MODULE_CPP_SOURCE,
)

torch.nn.SampleModule = SampleModule
