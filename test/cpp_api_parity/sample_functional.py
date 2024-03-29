import torch
import torch.nn.functional as F
from torch.testing._internal.common_nn import wrap_functional

'''
`sample_functional` is used by `test_cpp_api_parity.py` to test that Python / C++ API
parity test harness works for `torch.nn.functional` functions.

When `has_parity=true` is passed to `sample_functional`, behavior of `sample_functional`
is the same as the C++ equivalent.

When `has_parity=false` is passed to `sample_functional`, behavior of `sample_functional`
is different from the C++ equivalent.
'''

def sample_functional(x, has_parity):
    if has_parity:
        return x * 2
    else:
        return x * 4

torch.nn.functional.sample_functional = sample_functional

SAMPLE_FUNCTIONAL_CPP_SOURCE = """\n
namespace torch {
namespace nn {
namespace functional {

struct C10_EXPORT SampleFunctionalFuncOptions {
  SampleFunctionalFuncOptions(bool has_parity) : has_parity_(has_parity) {}

  TORCH_ARG(bool, has_parity);
};

Tensor sample_functional(Tensor x, SampleFunctionalFuncOptions options) {
    return x * 2;
}

} // namespace functional
} // namespace nn
} // namespace torch
"""

functional_tests = [
    dict(
        constructor=wrap_functional(F.sample_functional, has_parity=True),
        cpp_options_args='F::SampleFunctionalFuncOptions(true)',
        input_size=(1, 2, 3),
        fullname='sample_functional_has_parity',
        has_parity=True,
    ),
    dict(
        constructor=wrap_functional(F.sample_functional, has_parity=False),
        cpp_options_args='F::SampleFunctionalFuncOptions(false)',
        input_size=(1, 2, 3),
        fullname='sample_functional_no_parity',
        has_parity=False,
    ),
    # This is to test that setting the `test_cpp_api_parity=False` flag skips
    # the C++ API parity test accordingly (otherwise this test would run and
    # throw a parity error).
    dict(
        constructor=wrap_functional(F.sample_functional, has_parity=False),
        cpp_options_args='F::SampleFunctionalFuncOptions(false)',
        input_size=(1, 2, 3),
        fullname='sample_functional_THIS_TEST_SHOULD_BE_SKIPPED',
        test_cpp_api_parity=False,
    ),
]
