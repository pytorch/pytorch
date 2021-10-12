from itertools import product

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


def get_default_value(initial_value, reduction):
    if initial_value is not None:
        return initial_value
    if reduction == "max":
        return -float("Inf")
    elif reduction == "mean":
        return float("nan")
    elif reduction == "min":
        return float("Inf")
    elif reduction == "sum":
        return 0.0


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
        lengths_dtype=torch.int,
    ):
        lengths = torch.tensor(lengths_arr, device=device, dtype=lengths_dtype)
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

    @dtypes(
        *product(
            (torch.half, torch.bfloat16, torch.float, torch.double),
            (torch.int, torch.int64),
        )
    )
    def test_simple_1d(self, device, dtypes):
        val_dtype, length_type = dtypes
        lengths = [1, 2, 3, 0]
        data = [1, float("nan"), 3, 4, 5, 5]

        for reduction in reductions:
            for initial in [0, None]:
                check_backward = True if initial is not None else False
                initial_value = initial
                default_value = get_default_value(initial_value, reduction)
                if reduction == "max":
                    expected_result = [1, float("nan"), 5, default_value]
                    expected_grad = [1, 1, 0, 0, 0.5, 0.5]
                elif reduction == "mean":
                    expected_result = [1, float("nan"), 4.666, default_value]
                    expected_grad = [1.0, 0.5, 0.5, 0.333, 0.333, 0.333]
                elif reduction == "min":
                    if initial is not None:
                        initial_value = 1000  # some high number
                        default_value = get_default_value(initial_value, reduction)
                    expected_result = [1, float("nan"), 4, default_value]
                    expected_grad = [1.0, 1.0, 0, 1, 0, 0]
                elif reduction == "sum":
                    expected_result = [1, float("nan"), 14, default_value]
                    expected_grad = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                for axis in [0, -1]:
                    for unsafe in [True, False]:
                        self._test_common(
                            reduction,
                            device,
                            val_dtype,
                            unsafe,
                            axis,
                            initial_value,
                            data,
                            lengths,
                            expected_result,
                            expected_grad,
                            check_backward,
                            length_type,
                        )

    @dtypes(
        *product(
            (torch.half, torch.bfloat16, torch.float, torch.double),
            (torch.int, torch.int64),
        )
    )
    def test_multi_d_simple(self, device, dtypes):
        val_dtype, length_type = dtypes
        axis = 0
        lengths = [1, 2, 3, 0]
        data = [[1, 1], [float("nan"), 1], [3, float("nan")], [4, 1], [3, 2], [2, 3]]

        for reduction in reductions:
            for initial in [0, None]:
                check_backward = True if initial is not None else False
                initial_value = initial
                default_value = get_default_value(initial_value, reduction)
                if reduction == "max":
                    expected_result = [
                        [1, 1],
                        [float("nan"), float("nan")],
                        [4, 3],
                        [default_value, default_value],
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
                    expected_result = [
                        [1, 1],
                        [float("nan"), float("nan")],
                        [3, 2],
                        [default_value, default_value],
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
                    if initial is not None:
                        initial_value = 1000  # some high number
                        default_value = get_default_value(initial_value, reduction)
                    expected_result = [
                        [1, 1],
                        [float("nan"), float("nan")],
                        [2, 1],
                        [default_value, default_value],
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
                    expected_result = [
                        [1, 1],
                        [float("nan"), float("nan")],
                        [9, 6],
                        [default_value, default_value],
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
                    self._test_common(
                        reduction,
                        device,
                        val_dtype,
                        unsafe,
                        axis,
                        initial_value,
                        data,
                        lengths,
                        expected_result,
                        expected_grad,
                        check_backward,
                    )

    @dtypes(
        *product(
            (torch.half, torch.bfloat16, torch.float, torch.double),
            (torch.int, torch.int64),
        )
    )
    def test_multi_d(self, device, dtypes):
        val_dtype, length_type = dtypes
        axis = 0
        lengths = [0, 2]
        data = np.arange(20).reshape(2, 2, 5).tolist()
        expected_grad = []

        # TODO: calculate grad and check correctness
        check_backward = False

        for reduction in reductions:
            initial_value = 0
            if reduction == "max":
                expected_result = [
                    np.full((2, 5), initial_value).tolist(),
                    np.max(data, axis=0).tolist(),
                ]
            elif reduction == "mean":
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
                expected_result = [
                    np.full((2, 5), initial_value).tolist(),
                    np.sum(data, axis=0).tolist(),
                ]
            for unsafe in [True, False]:
                self._test_common(
                    reduction,
                    device,
                    val_dtype,
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
