import torch

from cpp_api_parity import torch_nn_modules, TorchNNModuleMetadata

'''
`SampleModule` is used by `test_cpp_api_parity.py` to test that Python / C++ API
parity test harness works for `torch.nn.Module` subclasses.

When `SampleModule.has_parity` is true, behavior of `reset_parameters` / `forward` /
`backward` is the same as the C++ equivalent.

When `SampleModule.has_parity` is false, behavior of `reset_parameters` / `forward` /
`backward` is different from the C++ equivalent.
'''

class SampleModule(torch.nn.Module):
    def __init__(self, has_parity, has_submodule, int_option=0, double_option=0.1,
                 bool_option=False, string_option='0', tensor_option=torch.zeros(1),
                 int_or_tuple_option=0):
        super(SampleModule, self).__init__()
        self.has_parity = has_parity
        if has_submodule:
            self.submodule = SampleModule(self.has_parity, False)

        # The following attributes will be included in the `num_attrs_recursive` count.
        self.has_submodule = has_submodule
        self.int_option = int_option
        self.double_option = double_option
        self.bool_option = bool_option
        self.string_option = string_option
        self.tensor_option = tensor_option
        self.int_or_tuple_option = int_or_tuple_option
        self.register_parameter('param', torch.nn.Parameter(torch.empty(3, 4)))
        self.register_buffer('buffer', torch.empty(4, 5))
        self.attr = 0

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
namespace torch {
namespace nn{
struct C10_EXPORT SampleModuleOptions {
  SampleModuleOptions(bool has_submodule) : has_submodule_(has_submodule) {}
  TORCH_ARG(bool, has_submodule);
  TORCH_ARG(int64_t, int_option) = 0;
  TORCH_ARG(double, double_option) = 0.1;
  TORCH_ARG(bool, bool_option) = false;
  TORCH_ARG(std::string, string_option) = "0";
  TORCH_ARG(torch::Tensor, tensor_option) = torch::zeros({1});
  TORCH_ARG(ExpandingArray<2>, int_or_tuple_option) = 0;
};

struct C10_EXPORT SampleModuleImpl : public torch::nn::Cloneable<SampleModuleImpl> {
  SampleModuleImpl(bool has_submodule) : SampleModuleImpl(SampleModuleOptions(has_submodule)) {}
  explicit SampleModuleImpl(SampleModuleOptions options) : options(std::move(options)) {
    if (options.has_submodule()) {
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
  SampleModuleOptions options;
  torch::Tensor param;
  torch::Tensor buffer;
  int attr;
  std::shared_ptr<SampleModuleImpl> submodule{nullptr};
};

TORCH_MODULE(SampleModule);
}
}
"""

module_tests = [
    dict(
        module_name='SampleModule',
        desc='has_parity',
        constructor_args=(True, True),
        cpp_constructor_args='(true)',
        input_size=(3, 4),
        has_parity=True,
    ),
    dict(
        fullname='SampleModule_no_parity',
        constructor=lambda: SampleModule(False, True),
        cpp_constructor_args='(true)',
        input_size=(3, 4),
        has_parity=False,
    ),
]

torch_nn_modules.module_metadata_map['SampleModule'] = TorchNNModuleMetadata(
    cpp_default_constructor_args='(true)',
    num_attrs_recursive=20,
    cpp_sources=SAMPLE_MODULE_CPP_SOURCE,
    python_ignored_constructor_args=['has_parity'],
    python_ignored_attrs=['has_parity'],
)

torch.nn.SampleModule = SampleModule
