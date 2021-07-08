import numpy as np
import torch
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    dtypes,
)
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
    gradcheck,
)


reductions = ["max", "mean", "min", "sum"]


class TestSegmentReductions(TestCase):
    def _test_common(
        self,
        reduction,
        device,
        dtype,
        unsafe,
        axis,
        initial_value,
        data_arr,
        lengths_arr,
        expected_arr,
        expected_grad_arr,
        check_backward,
    ):
        lengths = torch.tensor(lengths_arr, device=device)
        data = torch.tensor(
            data_arr,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        expected_result = torch.tensor(expected_arr, device=device, dtype=dtype)
        expected_grad = torch.tensor(expected_grad_arr, device=device, dtype=dtype)
        actual_result = torch.segment_reduce(
            data=data,
            reduce=reduction,
            lengths=lengths,
            axis=axis,
            unsafe=unsafe,
            initial=initial_value,
        )
        self.assertEqual(
            expected_result, actual_result, rtol=1e-02, atol=1e-05, equal_nan=True
        )

        if not check_backward:
            return

        # Test backward
        actual_result.sum().backward()
        self.assertEqual(
            expected_grad, data.grad, rtol=1e-02, atol=1e-05, equal_nan=True
        )

        # gradcheck does not work well with bfloat16 or fp16 cpu types
        # also there is small numerical difference with fp32
        if dtype not in [torch.half, torch.bfloat16, torch.float]:
            # gradcheck does not like "nan" input, setting to random 10
            d_non_nan = np.nan_to_num(data_arr, nan=10)
            data = torch.tensor(
                # [10 if v == float("nan") else v for v in data],
                d_non_nan,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            self.assertTrue(
                gradcheck(
                    lambda x: torch.segment_reduce(
                        data=x,
                        reduce=reduction,
                        lengths=lengths,
                        axis=axis,
                        unsafe=unsafe,
                        initial=initial_value,
                    ),
                    (data,),
                )
            )

    @dtypes(torch.half, torch.bfloat16, torch.float, torch.double)
    def test_simple_1d(self, device, dtype):
        lengths = [1, 2, 3, 0]
        data = [1, float("nan"), 3, 4, 5, 5]
        check_backward = True

        for reduction in reductions:
            if reduction == "max":
                initial_value = 0
                expected_result = [1, float("nan"), 5, initial_value]
                expected_grad = [1, 1, 0, 0, 0.5, 0.5]
            elif reduction == "mean":
                initial_value = 0
                expected_result = [1, float("nan"), 4.666, initial_value]
                expected_grad = [1.0, 0.5, 0.5, 0.333, 0.333, 0.333]
            elif reduction == "min":
                initial_value = 1000  # some high number
                expected_result = [1, float("nan"), 4, initial_value]
                expected_grad = [1.0, 1.0, 0, 1, 0, 0]
            elif reduction == "sum":
                initial_value = 0
                expected_result = [1, float("nan"), 14, initial_value]
                expected_grad = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            for axis in [0, -1]:
                for unsafe in [True, False]:
                    for initial in [initial_value, None]:
                        self._test_common(
                            reduction,
                            device,
                            dtype,
                            unsafe,
                            axis,
                            initial_value,
                            data,
                            lengths,
                            expected_result,
                            expected_grad,
                            check_backward,
                        )

    @dtypes(torch.half, torch.bfloat16, torch.float, torch.double)
    def test_multi_d_simple(self, device, dtype):
        check_backward = True
        axis = 0
        lengths = [1, 2, 3, 0]
        data = [[1, 1], [float("nan"), 1], [3, float("nan")], [4, 1], [3, 2], [2, 3]]

        for reduction in reductions:
            if reduction == "max":
                initial_value = 0
                expected_result = [
                    [1, 1],
                    [float("nan"), float("nan")],
                    [4, 3],
                    [initial_value, initial_value],
                ]
                expected_grad = [
                    [1, 1],
                    [1, 0],
                    [0, 1],
                    [1, 0],
                    [0, 0],
                    [0, 1],
                ]
            elif reduction == "mean":
                initial_value = 0
                expected_result = [
                    [1, 1],
                    [float("nan"), float("nan")],
                    [3, 2],
                    [initial_value, initial_value],
                ]
                expected_grad = [
                    [1.0, 1.0],
                    [0.5, 0.5],
                    [0.5, 0.5],
                    [0.333, 0.333],
                    [0.333, 0.333],
                    [0.333, 0.333],
                ]
            elif reduction == "min":
                initial_value = 1000  # some high number
                expected_result = [
                    [1, 1],
                    [float("nan"), float("nan")],
                    [2, 1],
                    [initial_value, initial_value],
                ]
                expected_grad = [
                    [1.0, 1.0],
                    [1, 0],
                    [0, 1],
                    [0, 1],
                    [0, 0],
                    [1, 0],
                ]
            elif reduction == "sum":
                initial_value = 0
                expected_result = [
                    [1, 1],
                    [float("nan"), float("nan")],
                    [9, 6],
                    [initial_value, initial_value],
                ]
                expected_grad = [
                    [1.0, 1.0],
                    [1.0, 1.0],
                    [1.0, 1.0],
                    [1.0, 1.0],
                    [1.0, 1.0],
                    [1.0, 1.0],
                ]
            for unsafe in [True, False]:
                for initial in [initial_value, None]:
                    self._test_common(
                        reduction,
                        device,
                        dtype,
                        unsafe,
                        axis,
                        initial_value,
                        data,
                        lengths,
                        expected_result,
                        expected_grad,
                        check_backward,
                    )

    @dtypes(torch.half, torch.bfloat16, torch.float, torch.double)
    def test_multi_d(self, device, dtype):
        axis = 0
        lengths = [0, 2]
        data = np.arange(20).reshape(2, 2, 5).tolist()
        expected_grad = []

        # TODO: calculate grad and check correctness
        check_backward = False

        for reduction in reductions:
            if reduction == "max":
                initial_value = 0
                expected_result = [
                    np.full((2, 5), initial_value).tolist(),
                    np.max(data, axis=0).tolist(),
                ]
            elif reduction == "mean":
                initial_value = 0
                expected_result = [
                    np.full((2, 5), initial_value).tolist(),
                    np.mean(data, axis=0).tolist(),
                ]
            elif reduction == "min":
                initial_value = 1000  # some high number
                expected_result = [
                    np.full((2, 5), initial_value).tolist(),
                    np.min(data, axis=0).tolist(),
                ]
            elif reduction == "sum":
                initial_value = 0
                expected_result = [
                    np.full((2, 5), initial_value).tolist(),
                    np.sum(data, axis=0).tolist(),
                ]
            for unsafe in [True, False]:
                for initial in [initial_value, None]:
                    self._test_common(
                        reduction,
                        device,
                        dtype,
                        unsafe,
                        axis,
                        initial_value,
                        data,
                        lengths,
                        expected_result,
                        expected_grad,
                        check_backward,
                    )


instantiate_device_type_tests(TestSegmentReductions, globals())

if __name__ == "__main__":
    run_tests()
