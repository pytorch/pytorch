import torch
from cpp_api_parity import CppArgDeclaration
from typing import Dict

'''
`SampleModule` is used by `test_cpp_api_parity.py` to test that Python / C++ API
parity test harness works for `torch.nn.Module` subclasses.

When `SampleModule.has_parity` is true, behavior of `reset_parameters` / `forward` /
`backward` is the same as the C++ equivalent.

When `SampleModule.has_parity` is false, behavior of `reset_parameters` / `forward` /
`backward` is different from the C++ equivalent.
'''

class SampleModule(torch.nn.Module):
    def __init__(self, has_parity, has_submodule):
        super(SampleModule, self).__init__()
        self.register_parameter('param', torch.nn.Parameter(torch.empty(3, 4)))
        self.register_buffer('buffer', torch.empty(4, 5))
        self.has_parity = has_parity
        if has_submodule:
            self.submodule = SampleModule(has_parity, False)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.param.fill_(1)
            self.buffer.fill_(1)
            self.attr = 10
            if not self.has_parity:
                self.param.add_(10)
                self.buffer.add_(10)
                self.attr += 90

    def forward(self, x):
        submodule_forward_result = self.submodule(x) if hasattr(self, 'submodule') else 0
        if not self.has_parity:
            return x + self.param * 4 + submodule_forward_result + 3
        else:
            return x + self.param * 2 + submodule_forward_result

SAMPLE_MODULE_CPP_SOURCE = """\n
struct SampleModuleImpl : public torch::nn::Cloneable<SampleModuleImpl> {
  SampleModuleImpl(bool has_submodule) {
    if (has_submodule) {
      submodule = register_module("submodule", std::make_shared<SampleModuleImpl>(false));
    }
    reset();
  }
  void reset() {
    attr = 10;
    param = register_parameter("param", torch::ones({3, 4}));
    buffer = register_buffer("buffer", torch::ones({4, 5}));
  }
  torch::Tensor forward(torch::Tensor x) {
    return x + param * 2 + (submodule ? submodule->forward(x) : torch::zeros_like(x));
  }
  torch::Tensor param;
  torch::Tensor buffer;
  int attr;
  std::shared_ptr<SampleModuleImpl> submodule{nullptr};
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
        constructor_args=(True, True),
        cpp_constructor_args='(true)',
        input_size=(3, 4),
        desc='has_parity',
        expect_parity_error=False,
    ),
    dict(
        module_name='SampleModule',
        constructor_args=(False, True),
        cpp_constructor_args='(true)',
        input_size=(3, 4),
        desc='no_parity',
        expect_parity_error=True,
    ),
]

module_metadata = dict(
    # yf225 TODO: can we use example_inputs to generate the arg_type here (just support a few common types), and get rid of `cpp_forward_arg_declarations`?
    cpp_forward_arg_declarations=[CppArgDeclaration(arg_type='torch::Tensor', arg_name='x')],
    cpp_source=SAMPLE_MODULE_CPP_SOURCE,
)

torch.nn.SampleModule = SampleModule
