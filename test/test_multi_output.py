import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from collections import namedtuple
import copy

# Device-generic tests. Instantiated below and not run directly.
class TestMultiOutput(TestCase):

    # See https://github.com/pytorch/pytorch/issues/42364
    # This verifies that out= variants operate successfully when there are
    # multiple outputs and those outputs have unique strides
    def test_multi_output_different_strides(self, device):
        FnExecution = namedtuple('FnExecution', 'fn, args, kwargs')
        def get_ops(device):
            return [
                FnExecution(fn=torch.max, args=(torch.randn(2, 3, 4, device=device), 0), kwargs=dict()),
                FnExecution(fn=torch.min, args=(torch.randn(2, 3, 4, device=device), 0), kwargs=dict()),
                FnExecution(fn=torch.cummax, args=(torch.randn(2, 3, 4, device=device), 0), kwargs=dict()),
                FnExecution(fn=torch.cummin, args=(torch.randn(2, 3, 4, device=device), 0), kwargs=dict()),
                FnExecution(fn=torch.kthvalue, args=(torch.randn(2, 3, 4, device=device), 2, 0), kwargs=dict()),
                FnExecution(fn=torch.median, args=(torch.randn(2, 3, 4, device=device), 0), kwargs=dict()),
                FnExecution(fn=torch.mode, args=(torch.randn(2, 3, 4, device=device), 0), kwargs=dict()),
                # this enters a different code path in the CUDA kernel -- ideally we would trigger this via
                # introspection or via code coverage metrics.
                FnExecution(fn=torch.mode, args=(torch.randn(1025, 2, device=device), 0), kwargs=dict()),
            ]

        for fn, args, kwargs in get_ops(device):
            # this section verifies that the operators in the test have the properties we want, namely:
            # 1. output is a tuple of len > 1
            # 2. all outputs are tensors (NOTE: test can be expanded to support this)

            # in the final form, we'd probably turn these asserts into skips to be able to iterate over
            # all operator invocations
            self.assertFalse('out' in kwargs)  # first invoke the non-out variant
            out = fn(*args, **kwargs)
            self.assertTrue(isinstance(out, tuple))  # only want to test multi-output Tensors here
            self.assertTrue(len(out) > 1)
            self.assertTrue(all([isinstance(x, torch.Tensor) for x in out]))

            # check calling out variant gives the same result
            out_copy = copy.deepcopy(out)
            kwargs.update({'out': out_copy})
            fn(*args, **kwargs)
            self.assertEqual(out, out_copy)

            out_modified = []
            # construct 'out' tensors with different strides.
            # NOTE: this is bit brittle -- we don't guarantee determinism across strides, but our stride
            # modification algorithm maintains the relative order of strides so (at least today) we are
            # guaranteed to pick the same iteration order.
            for i in range(len(out)):
                out_modified.append(torch.empty(out_copy[i].shape + (i + 1,),
                                                dtype=out_copy[i].dtype,
                                                device=out_copy[i].device)[..., 0])
            self.assertEqual(len(set([x.stride() for x in out_modified])), len(out_modified))
            kwargs.update({'out': out_modified})
            fn(*args, **kwargs)
            self.assertEqual(out, out_modified)

instantiate_device_type_tests(TestMultiOutput, globals())

if __name__ == '__main__':
    run_tests()
