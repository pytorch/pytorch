# Owner(s): ["module: tests"]

from hypothesis import given
from hypothesis.extra.numpy import mutually_broadcastable_shapes
import numpy as np
import hypothesis
import hypothesis.extra.numpy
import hypothesis.strategies
from numpy.testing import assert_array_equal
from numpy.typing import DTypeLike

from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import TestCase, run_tests
import torch
import torch.cuda
import torch.backends.mps


@hypothesis.strategies.composite
def ufunc_inputs(draw, ufunc, input_dtype: DTypeLike = np.int32):
    shapes, result_shape = draw(mutually_broadcastable_shapes(signature=ufunc.signature))
    return [draw(hypothesis.extra.numpy.arrays(input_dtype, shape=shape)) for shape in shapes]


@hypothesis.strategies.composite
def where_inputs(draw, xy_dtype: DTypeLike = int):
    # TODO: Could we do this with a ufunc sig? `where` doesn't seem to have one
    # @given(mutually_broadcastable_shapes(signature=np.where.signature))
    shapes, result_shape = draw(mutually_broadcastable_shapes(num_shapes=3))
    cond = draw(hypothesis.extra.numpy.arrays(bool, shape=shapes[0]))
    x = draw(hypothesis.extra.numpy.arrays(xy_dtype, shape=shapes[1]))
    y = draw(hypothesis.extra.numpy.arrays(xy_dtype, shape=shapes[2]))
    return cond, x, y


class TestConsistentWithNumpy(TestCase):

    # TODO: This is very close to being able to be genericized over arbitrary ufuncs
    @given(ufunc_inputs(np.matmul, input_dtype=np.float32))
    def test_matmul_consistent(self, device, inputs):
        torch_inputs = [torch.from_numpy(arr) for arr in inputs]
        np_out = np.matmul(*inputs)
        torch_cpu_out = torch.matmul(*torch_inputs)
        # Make sure we get consistent results between numpy and CPU pytorch
        assert_array_equal(np_out, torch_cpu_out.numpy())

        torch_dev_inputs = [arr.to(device) for arr in torch_inputs]
        torch_dev_out = torch.matmul(*torch_dev_inputs)
        # Now make sure every backend also produces identical results to numpy (and thus CPU pytorch)
        assert_array_equal(np_out, torch_dev_out.cpu().numpy())

    @given(where_inputs())
    def test_where_consistent(self, device, where_inputs):
        cond, x, y = where_inputs
        t_cond, t_x, t_y = torch.from_numpy(cond), torch.from_numpy(x), torch.from_numpy(y)

        np_out = np.where(cond, x, y)
        torch_cpu_out = torch.where(t_cond, t_x, t_y)
        # Make sure we get consistent results between numpy and CPU pytorch
        assert_array_equal(np_out, torch_cpu_out.numpy())

        torch_dev_out = torch.where(t_cond.to(device), t_x.to(device), t_y.to(device))
        # Now make sure every backend also produces identical results to numpy (and thus CPU pytorch)
        assert_array_equal(np_out, torch_dev_out.cpu().numpy())


instantiate_device_type_tests(TestConsistentWithNumpy, globals(), allow_mps=True)


if __name__ == '__main__':
    run_tests()
