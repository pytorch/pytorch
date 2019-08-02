import torch

'''
`SampleModule` is used by `test_cpp_api_parity.py` to test that Python / C++ API
parity test harness works for `torch.nn.Module` subclasses.

When `SampleModule.has_parity` is true, behavior of `reset_parameters` and `forward`
is the same as the C++ equivalent.

When `SampleModule.has_parity` is false, behavior of `reset_parameters` and `forward`
is different from the C++ equivalent.
'''

class SampleModule(torch.nn.Module):
    def __init__(self, has_parity):
        super(SampleModule, self).__init__()
        self.register_parameter('param', torch.nn.Parameter(torch.Tensor(2, 3)))
        self.register_buffer('buffer', torch.Tensor(3, 4))
        self.has_parity = has_parity
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.param.zero_()
            self.buffer.zero_()
            if not self.has_parity:
                self.param.add_(1)
                self.buffer.add_(1)

    def forward(self, x):
        if not self.has_parity:
            return x + 1
        else:
            return x

SAMPLE_MODULE_CPP_SOURCE = """
struct SampleModuleImpl : public torch::nn::Cloneable<SampleModuleImpl> {
  SampleModuleImpl() {
    reset();
  }
  void reset() {
    param = register_parameter("param", torch::zeros({2, 3}));
    buffer = register_buffer("buffer", torch::zeros({3, 4}));
  }
  torch::Tensor forward(torch::Tensor x) {
    return x;
  }
  torch::Tensor param;
  torch::Tensor buffer;
};

TORCH_MODULE(SampleModule);
"""

module_tests = [
    dict(
        module_name='SampleModule',
        constructor_args=(True, ),
        cpp_constructor_args='',
        cpp_source=SAMPLE_MODULE_CPP_SOURCE,
        input_size=(3, 4),
        desc='has_parity',
        expect_error=False,
    ),
    dict(
        module_name='SampleModule',
        constructor_args=(False, ),
        cpp_constructor_args='',
        cpp_source=SAMPLE_MODULE_CPP_SOURCE,
        input_size=(3, 4),
        desc='no_parity',
        expect_error=True,
    ),
]
