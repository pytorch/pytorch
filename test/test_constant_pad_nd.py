import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests, onlyCPU, onlyNativeDeviceTypes
from torch.testing._internal.common_dtype import floating_types


class TestConstantPadNd(TestCase):
    """Test constant_pad_nd functionality across different devices and dtypes."""
    
    @onlyCPU
    def test_constant_pad_nd_mixed_pos_neg_allows_zero_cpu(self, device):
        x = torch.ones(5, 3, device=device)
        y = torch.ops.aten.constant_pad_nd.default(x, [-1, -2, 1, 1], 0)
        assert tuple(y.shape) == (7, 0)
        assert y.numel() == 0

    @onlyCPU
    def test_constant_pad_nd_negative_size_still_errors_cpu(self, device):
        x = torch.ones(5, 3, device=device)
        with self.assertRaises(RuntimeError):
            torch.ops.aten.constant_pad_nd.default(x, [-2, -2], 0)

    @onlyCPU
    def test_fpad_matches_constant_pad_nd_cpu(self, device):
        x = torch.ones(5, 3, device=device)
        z = F.pad(x, (-1, -2, 1, 1), mode="constant", value=0)
        assert tuple(z.shape) == (7, 0)
        assert z.numel() == 0

    @onlyNativeDeviceTypes
    @dtypes(*floating_types)
    def test_constant_pad_nd_devices(self, device, dtype):
        x = torch.ones(5, 3, device=device, dtype=dtype)
        y = torch.ops.aten.constant_pad_nd.default(x, [-1, -2, 1, 1], 0)
        assert tuple(y.shape) == (7, 0)
        assert y.numel() == 0


# Generate device-specific test classes
instantiate_device_type_tests(TestConstantPadNd, globals())


if __name__ == '__main__':
    run_tests()