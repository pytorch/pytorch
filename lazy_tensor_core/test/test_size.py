from collections.abc import Sequence
from functools import partial, wraps
import warnings

import torch

from torch.testing import FileCheck, make_tensor
from torch.testing._internal.common_dtype import floating_and_complex_types_and, get_all_dtypes
from torch.testing._internal.common_utils import \
    (TestCase, is_iterable_of_tensors, run_tests, IS_SANDCASTLE, clone_input_helper,
     gradcheck, gradgradcheck, IS_IN_CI, suppress_warnings)
from torch.testing._internal.common_methods_invocations import \
    (op_db, _NOTHING, UnaryUfuncInfo, ReductionOpInfo, SpectralFuncInfo)
from torch.testing._internal.common_device_type import \
    (deviceCountAtLeast, instantiate_device_type_tests, ops, onlyCUDA, onlyCPU, onlyOnCPUAndCUDA, skipCUDAIfRocm, OpDTypes)
from torch.testing._internal.common_jit import JitCommonTestCase, check_against_reference
from torch.testing._internal.jit_metaprogramming_utils import create_script_fn, create_traced_fn, \
    check_alias_annotation
from torch.testing._internal.jit_utils import disable_autodiff_subgraph_inlining
import torch.testing._internal.opinfo_helper as opinfo_helper


import lazy_tensor_core
import lazy_tensor_core.debug.metrics as metrics

lazy_tensor_core._LAZYC._ltc_init_ts_backend()

torch.manual_seed(42)


class TestLazyTensor(TestCase):

    @ops(op_db, allowed_dtypes=(torch.float,))
    @onlyCPU
    def test_size(self, device, dtype, op):

        #requires_grad = (dtype in allowed_backward_dtypes and op.supports_autograd)
        requires_grad = False
        samples = op.sample_inputs(device, dtype, requires_grad=requires_grad)
        results = []
        for sample in samples:
            result = op(sample.input, *sample.args, **sample.kwargs)
            results.append(result)

        samples = op.sample_inputs("lazy", dtype, requires_grad=requires_grad)
        for (i, sample) in enumerate(samples):
            result = op(sample.input, *sample.args, **sample.kwargs)
            self.assertEqual(results[i].size(), result.size())


instantiate_device_type_tests(TestLazyTensor, globals())

if __name__ == '__main__':
    run_tests()

