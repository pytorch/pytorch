from typing import List, Optional

import torch
from torch.fx.experimental.fx2trt import (
    InputTensorSpec,
)
from torch.testing._internal.common_utils import TestCase, run_tests


class TestTRTModule(TestCase):
    def _validate_spec(
        self,
        spec: InputTensorSpec,
        tensor: torch.Tensor,
        dynamic_dims: Optional[List[int]] = None
    ):
        expected_shape = list(tensor.shape)
        if dynamic_dims:
            for dim in dynamic_dims:
                expected_shape[dim] = -1
        self.assertSequenceEqual(spec.shape, expected_shape)
        self.assertEqual(spec.dtype, tensor.dtype)
        self.assertEqual(spec.device, tensor.device)
        self.assertTrue(spec.has_batch_dim)

    def test_from_tensor(self):
        tensor = torch.randn(1, 2, 3)
        spec = InputTensorSpec.from_tensor(tensor)
        self._validate_spec(spec, tensor)

    def test_from_tensors(self):
        tensors = [torch.randn(1, 2, 3), torch.randn(2, 4)]
        specs = InputTensorSpec.from_tensors(tensors)
        for spec, tensor in zip(specs, tensors):
            self._validate_spec(spec, tensor)

    def test_from_tensors_with_dynamic_batch_size(self):
        tensors = [torch.randn(1, 2, 3), torch.randn(1, 4)]
        batch_size_range = [2, 3, 4]
        specs = InputTensorSpec.from_tensors_with_dynamic_batch_size(tensors, batch_size_range)
        for spec, tensor in zip(specs, tensors):
            self._validate_spec(spec, tensor, dynamic_dims=[0])

            for batch_size, shape in zip(batch_size_range, spec.shape_ranges[0]):
                self.assertEqual(batch_size, shape[0])
                self.assertSequenceEqual(tensor.shape[1:], shape[1:])

if __name__ == '__main__':
    run_tests()
