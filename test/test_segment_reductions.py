import torch
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCPU,
    dtypes,
)
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)


class TestSegmentReductions(TestCase):
    @onlyCPU
    @dtypes(torch.half, torch.bfloat16, torch.float, torch.double)
    def test_max_simple_1d(self, device, dtype):
        lengths = torch.tensor([1, 2, 3], device=device)
        data = torch.tensor([1, float("nan"), 3, 4, 5, 6], device=device, dtype=dtype)
        expected_result = torch.tensor([1, float("nan"), 6], device=device, dtype=dtype)
        actual_result = torch.segment_reduce(
            data=data, reduce="max", lengths=lengths, axis=0, unsafe=False
        )
        self.assertEqual(
            expected_result, actual_result, rtol=1e-03, atol=1e-05, equal_nan=True
        )
        actual_result = torch.segment_reduce(
            data=data, reduce="max", lengths=lengths, axis=-1, unsafe=False
        )
        self.assertEqual(
            expected_result, actual_result, rtol=1e-03, atol=1e-05, equal_nan=True
        )


instantiate_device_type_tests(TestSegmentReductions, globals())

if __name__ == "__main__":
    run_tests()
