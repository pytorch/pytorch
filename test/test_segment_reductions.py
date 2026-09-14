# Owner(s): ["module: scatter & gather ops"]

from itertools import product
from functools import partial

import numpy as np
import torch
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    dtypes,
    dtypesIfMPS,
    onlyMPS,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    TestCase,
    run_tests,
    gradcheck,
    parametrize,
)


reductions = ["max", "mean", "min", "sum", "prod"]
mps_dtypes = tuple(product((torch.half, torch.bfloat16, torch.float), (torch.int, torch.int64)))


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
    elif reduction == "prod":
        return 1.0


class TestSegmentReductions(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

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
        # generate offsets from lengths
        zeros_shape = list(lengths.shape)
        zeros_shape[-1] = 1
        offsets = torch.cat((lengths.new_zeros(zeros_shape), lengths), -1).cumsum_(-1)

        data = torch.tensor(
            data_arr,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        expected_result = torch.tensor(expected_arr, device=device, dtype=dtype)
        expected_grad = torch.tensor(expected_grad_arr, device=device, dtype=dtype)
        for mode in ['lengths', 'offsets']:
            segment_reduce_kwargs = dict(
                axis=axis,
                unsafe=unsafe,
                initial=initial_value)
            if (mode == 'lengths'):
                segment_reduce_kwargs['lengths'] = lengths
            else:
                segment_reduce_kwargs['offsets'] = offsets
            actual_result = torch._segment_reduce(
                data=data,
                reduce=reduction,
                **segment_reduce_kwargs
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
            data = data.detach().clone().requires_grad_(True)

            # gradcheck does not work well with bfloat16 or fp16 cpu types
            # also there is small numerical difference with fp32
            if dtype not in [torch.half, torch.bfloat16, torch.float]:
                # gradcheck does not like "nan" input, setting to random 10
                d_non_nan = np.nan_to_num(data_arr, nan=10)
                new_data = torch.tensor(
                    # [10 if v == float("nan") else v for v in data],
                    d_non_nan,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
                self.assertTrue(
                    gradcheck(
                        lambda x: torch._segment_reduce(
                            data=x,
                            reduce=reduction,
                            **segment_reduce_kwargs
                        ),
                        (new_data,),
                    )
                )

    @dtypes(
        *product(
            (torch.half, torch.bfloat16, torch.float, torch.double),
            (torch.int, torch.int64),
        )
    )
    @dtypesIfMPS(*mps_dtypes)
    def test_simple_1d(self, device, dtypes):
        val_dtype, length_type = dtypes
        lengths = [1, 2, 3, 0]
        data = [1, float("nan"), 3, 4, 5, 5]

        for reduction in reductions:
            for initial in [0, None]:
                check_backward = initial is not None
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
                elif reduction == "prod":
                    if initial is not None:
                        initial_value = 2  # 0 initial_value will zero out everything for prod
                        default_value = get_default_value(initial_value, reduction)
                        expected_result = [2, float("nan"), 200, default_value]
                        expected_grad = [2.0, 6.0, float("nan"), 50.0, 40.0, 40.0]
                    else:
                        expected_result = [1, float("nan"), 100, default_value]
                        expected_grad = [1.0, 3.0, float("nan"), 25.0, 20.0, 20.0]
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
    @dtypesIfMPS(*mps_dtypes)
    def test_simple_zero_length(self, device, dtypes):
        val_dtype, length_type = dtypes
        lengths = [0, 0]
        data = torch.ones(0)

        for reduction in reductions:
            for initial in [0, None]:
                check_backward = initial is not None
                initial_value = initial
                default_value = get_default_value(initial_value, reduction)
                if reduction == "max":
                    expected_result = [default_value, default_value]
                    expected_grad = []
                elif reduction == "mean":
                    expected_result = [default_value, default_value]
                    expected_grad = []
                elif reduction == "min":
                    if initial is not None:
                        initial_value = 1000  # some high number
                        default_value = get_default_value(initial_value, reduction)
                    expected_result = [default_value, default_value]
                    expected_grad = []
                elif reduction == "sum":
                    expected_result = [default_value, default_value]
                    expected_grad = []
                elif reduction == "prod":
                    if initial is not None:
                        initial_value = 2  # 0 initial_value will zero out everything for prod
                        default_value = get_default_value(initial_value, reduction)
                        expected_result = [default_value, default_value]
                        expected_grad = []
                    else:
                        expected_result = [default_value, default_value]
                        expected_grad = []
                for axis in [0]:
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
    @dtypesIfMPS(*mps_dtypes)
    def test_multi_d_simple(self, device, dtypes):
        val_dtype, _ = dtypes
        axis = 0
        lengths = [1, 2, 3, 0]
        data = [[1, 1], [float("nan"), 1], [3, float("nan")], [4, 1], [3, 2], [2, 3]]

        for reduction in reductions:
            for initial in [0, None]:
                check_backward = initial is not None
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
                elif reduction == "prod":
                    if initial is not None:
                        initial_value = 2  # 0 initial_value will zero out everything for prod
                        default_value = get_default_value(initial_value, reduction)
                        expected_result = [
                            [2, 2],
                            [float("nan"), float("nan")],
                            [48, 12],
                            [default_value, default_value],
                        ]
                        expected_grad = [
                            [2.0, 2.0],
                            [6.0, float("nan")],
                            [float("nan"), 2.0],
                            [12.0, 12.0],
                            [16.0, 6.0],
                            [24.0, 4.0],
                        ]
                    else:
                        expected_result = [
                            [1, 1],
                            [float("nan"), float("nan")],
                            [24, 6],
                            [default_value, default_value],
                        ]
                        expected_grad = [
                            [1.0, 1.0],
                            [3.0, float("nan")],
                            [float("nan"), 1.0],
                            [6.0, 6.0],
                            [8.0, 3.0],
                            [12.0, 2.0],
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
    @parametrize("reduce", ['sum', 'prod', 'min', 'max', 'mean'])
    @dtypesIfMPS(*mps_dtypes)
    def test_pytorch_scatter_test_cases(self, device, dtypes, reduce):
        val_dtype, length_dtype = dtypes
        # zero-length segments are filled with reduction inits contrary to pytorch_scatter.
        tests = [
            {
                'src': [1, 2, 3, 4, 5, 6],
                'index': [0, 0, 1, 1, 1, 3],
                'indptr': [0, 2, 5, 5, 6],
                'sum': [3, 12, 0, 6],
                'prod': [2, 60, 1, 6],
                'mean': [1.5, 4, float('nan'), 6],
                'min': [1, 3, float('inf'), 6],
                'max': [2, 5, -float('inf'), 6],
            },
            {
                'src': [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]],
                'index': [0, 0, 1, 1, 1, 3],
                'indptr': [0, 2, 5, 5, 6],
                'sum': [[4, 6], [21, 24], [0, 0], [11, 12]],
                'prod': [[3, 8], [315, 480], [1, 1], [11, 12]],
                'mean': [[2, 3], [7, 8], [float('nan'), float('nan')], [11, 12]],
                'min': [[1, 2], [5, 6], [float('inf'), float('inf')], [11, 12]],
                'max': [[3, 4], [9, 10], [-float('inf'), -float('inf')], [11, 12]],
            },
            {
                'src': [[1, 3, 5, 7, 9, 11], [2, 4, 6, 8, 10, 12]],
                'index': [[0, 0, 1, 1, 1, 3], [0, 0, 0, 1, 1, 2]],
                'indptr': [[0, 2, 5, 5, 6], [0, 3, 5, 6, 6]],
                'sum': [[4, 21, 0, 11], [12, 18, 12, 0]],
                'prod': [[3, 315, 1, 11], [48, 80, 12, 1]],
                'mean': [[2, 7, float('nan'), 11], [4, 9, 12, float('nan')]],
                'min': [[1, 5, float('inf'), 11], [2, 8, 12, float('inf')]],
                'max': [[3, 9, -float('inf'), 11], [6, 10, 12, -float('inf')]],
            },
            {
                'src': [[[1, 2], [3, 4], [5, 6]], [[7, 9], [10, 11], [12, 13]]],
                'index': [[0, 0, 1], [0, 2, 2]],
                'indptr': [[0, 2, 3, 3], [0, 1, 1, 3]],
                'sum': [[[4, 6], [5, 6], [0, 0]], [[7, 9], [0, 0], [22, 24]]],
                'prod': [[[3, 8], [5, 6], [1, 1]], [[7, 9], [1, 1], [120, 143]]],
                'mean': [[[2, 3], [5, 6], [float('nan'), float('nan')]],
                         [[7, 9], [float('nan'), float('nan')], [11, 12]]],
                'min': [[[1, 2], [5, 6], [float('inf'), float('inf')]],
                        [[7, 9], [float('inf'), float('inf')], [10, 11]]],
                'max': [[[3, 4], [5, 6], [-float('inf'), -float('inf')]],
                        [[7, 9], [-float('inf'), -float('inf')], [12, 13]]],
            },
            {
                'src': [[1, 3], [2, 4]],
                'index': [[0, 0], [0, 0]],
                'indptr': [[0, 2], [0, 2]],
                'sum': [[4], [6]],
                'prod': [[3], [8]],
                'mean': [[2], [3]],
                'min': [[1], [2]],
                'max': [[3], [4]],
            },
            {
                'src': [[[1, 1], [3, 3]], [[2, 2], [4, 4]]],
                'index': [[0, 0], [0, 0]],
                'indptr': [[0, 2], [0, 2]],
                'sum': [[[4, 4]], [[6, 6]]],
                'prod': [[[3, 3]], [[8, 8]]],
                'mean': [[[2, 2]], [[3, 3]]],
                'min': [[[1, 1]], [[2, 2]]],
                'max': [[[3, 3]], [[4, 4]]],
            },
        ]
        for test in tests:
            data = torch.tensor(test['src'], dtype=val_dtype, device=device, requires_grad=True)
            indptr = torch.tensor(test['indptr'], dtype=length_dtype, device=device)
            dim = indptr.ndim - 1
            # calculate lengths from indptr
            lengths = torch.diff(indptr, dim=dim)
            expected = torch.tensor(test[reduce], dtype=val_dtype, device=device)

            actual_result = torch._segment_reduce(
                data=data,
                reduce=reduce,
                lengths=lengths,
                axis=dim,
                unsafe=True,
            )
            self.assertEqual(actual_result, expected)

            # test offsets
            actual_result = torch._segment_reduce(
                data=data,
                reduce=reduce,
                offsets=indptr,
                axis=dim,
                unsafe=True,
            )
            self.assertEqual(actual_result, expected)

            if val_dtype == torch.float64:
                def fn(x, mode='lengths'):
                    initial = 1
                    # supply initial values to prevent gradcheck from failing for 0 length segments
                    # where nan/inf are reduction identities that produce nans when calculating the numerical jacobian
                    if reduce == 'min':
                        initial = 1000
                    elif reduce == 'max':
                        initial = -1000
                    segment_reduce_args = {x, reduce}
                    segment_reduce_kwargs = dict(axis=dim, unsafe=True, initial=initial)
                    if mode == 'lengths':
                        segment_reduce_kwargs[mode] = lengths
                    elif mode == 'offsets':
                        segment_reduce_kwargs[mode] = indptr
                    return torch._segment_reduce(*segment_reduce_args, **segment_reduce_kwargs)
                self.assertTrue(gradcheck(partial(fn, mode='lengths'), (data.detach().clone().requires_grad_(True))))
                self.assertTrue(gradcheck(partial(fn, mode='offsets'), (data.detach().clone().requires_grad_(True))))


    @dtypes(
        *product(
            (torch.half, torch.bfloat16, torch.float, torch.double),
            (torch.int, torch.int64),
        )
    )
    @dtypesIfMPS(*mps_dtypes)
    def test_multi_d(self, device, dtypes):
        val_dtype, _ = dtypes
        axis = 0
        lengths = [0, 2, 3, 0]
        data = np.arange(50).reshape(5, 2, 5).tolist()
        expected_grad = []

        # TODO: calculate grad and check correctness
        check_backward = False

        for reduction in reductions:
            initial_value = 0
            if reduction == "max":
                expected_result = [
                    np.full((2, 5), initial_value).tolist(),
                    np.max(data[:2], axis=0).tolist(),
                    np.max(data[2:], axis=0).tolist(),
                    np.full((2, 5), initial_value).tolist(),
                ]
            elif reduction == "mean":
                expected_result = [
                    np.full((2, 5), initial_value).tolist(),
                    np.mean(data[:2], axis=0).tolist(),
                    np.mean(data[2:], axis=0).tolist(),
                    np.full((2, 5), initial_value).tolist(),
                ]
            elif reduction == "min":
                initial_value = 1000  # some high number
                expected_result = [
                    np.full((2, 5), initial_value).tolist(),
                    np.min(data[:2], axis=0).tolist(),
                    np.min(data[2:], axis=0).tolist(),
                    np.full((2, 5), initial_value).tolist(),
                ]
            elif reduction == "sum":
                expected_result = [
                    np.full((2, 5), initial_value).tolist(),
                    np.sum(data[:2], axis=0).tolist(),
                    np.sum(data[2:], axis=0).tolist(),
                    np.full((2, 5), initial_value).tolist(),
                ]
            elif reduction == "prod":
                initial_value = 1
                expected_result = [
                    np.full((2, 5), initial_value).tolist(),
                    np.prod(data[:2], axis=0).tolist(),
                    np.prod(data[2:], axis=0).tolist(),
                    np.full((2, 5), initial_value).tolist(),
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

    @dtypes(torch.int, torch.int64)
    def test_unsafe_flag(self, device, dtype):
        length_type = dtype
        lengths = torch.tensor([0, 2, 3, 0], device=device, dtype=length_type)
        data = torch.arange(6, dtype=torch.float, device=device)

        # test for error on 1-D lengths
        with self.assertRaisesRegex(RuntimeError, "Expected all rows of lengths along axis"):
            torch._segment_reduce(data, 'sum', lengths=lengths, axis=0, unsafe=False)
            if torch.device(device).type == "mps":
                torch.mps.synchronize()

        # test for error on multi-D lengths
        nd_lengths = torch.tensor([[0, 3, 3, 0], [2, 3, 0, 0]], dtype=length_type, device=device)
        nd_data = torch.arange(12, dtype=torch.float, device=device).reshape(2, 6)
        with self.assertRaisesRegex(RuntimeError, "Expected all rows of lengths along axis"):
            torch._segment_reduce(nd_data, 'sum', lengths=nd_lengths, axis=1, unsafe=False)
            if torch.device(device).type == "mps":
                torch.mps.synchronize()

    @dtypes(torch.float32, torch.float16, torch.bfloat16)
    @parametrize("reduce", reductions)
    @parametrize("mode", ["lengths", "offsets"])
    @parametrize("initial", [None, 0., 2.])
    @parametrize("inner", [1, 7])
    def test_noncontiguous_forward_backward(self, device, dtype, reduce, mode, initial, inner):
        source = torch.randint(1, 3, (2, inner, 12)).to(dtype)
        source[0, :, 1] = 0
        source[1, :, 3] = float("nan")
        data_cpu = source[..., 1::2].transpose(1, 2).detach().requires_grad_()
        data = source.to(device)[..., 1::2].transpose(1, 2).detach().requires_grad_()
        lengths = torch.tensor([[0, 2, 1, 3], [3, 0, 2, 1]], dtype=torch.int32)
        metadata = lengths
        if mode == "offsets":
            metadata = torch.cat((lengths.new_zeros(2, 1), lengths), -1).cumsum(-1)
        storage = torch.zeros((*metadata.shape[:-1], metadata.size(-1) * 2), dtype=metadata.dtype)
        storage[..., ::2] = metadata
        kwargs = dict(axis=-2, initial=initial)
        expected = torch.segment_reduce(data_cpu, reduce, **{mode: storage[..., ::2]}, **kwargs)
        actual = torch.segment_reduce(data, reduce, **{mode: storage.to(device)[..., ::2]}, **kwargs)
        self.assertEqual(actual, expected, equal_nan=True)
        grad = torch.randn(2, inner, 4, dtype=dtype).transpose(1, 2)
        expected.backward(grad)
        actual.backward(grad.to(device))
        self.assertEqual(data.grad, data_cpu.grad, equal_nan=True)

    @dtypes(torch.float32)
    @dtypesIfMPS(torch.float32, torch.float16, torch.bfloat16)
    @parametrize("reduce", reductions)
    @parametrize("mode", ["lengths", "offsets"])
    def test_long_segments(self, device, dtype, reduce, mode):
        data = torch.ones(4099, dtype=dtype, device=device, requires_grad=True)
        lengths = torch.tensor([2048, 0, 2051], device=device)
        metadata = lengths
        if mode == "offsets":
            metadata = torch.tensor([0, 2048, 2048, 4099], device=device)
        actual = torch.segment_reduce(data, reduce, **{mode: metadata})
        identity = get_default_value(None, reduce)
        values = [2048, identity, 2051] if reduce == "sum" else [1, identity, 1]
        self.assertEqual(actual, torch.tensor(values, dtype=dtype), equal_nan=True)
        actual.backward(torch.ones_like(actual))
        expected_grad = torch.ones_like(data)
        if reduce in ("mean", "min", "max"):
            expected_grad[:2048] /= 2048
            expected_grad[2048:] /= 2051
        self.assertEqual(data.grad, expected_grad)

    @dtypes(torch.float32, torch.float16, torch.bfloat16)
    @parametrize("reduce", reductions)
    def test_partial_offsets(self, device, dtype, reduce):
        data_cpu = torch.tensor([99., 1., 2., 0., 4., 99.], dtype=dtype, requires_grad=True)
        data = data_cpu.detach().to(device).requires_grad_()
        offsets = torch.tensor([1, 3, 3, 5], dtype=torch.int64)
        expected = torch.segment_reduce(data_cpu, reduce, offsets=offsets, initial=2.)
        actual = torch.segment_reduce(data, reduce, offsets=offsets.to(device), initial=2.)
        self.assertEqual(actual, expected)
        expected.sum().backward()
        actual.sum().backward()
        self.assertEqual(data.grad, data_cpu.grad)

    @dtypes(torch.float32, torch.float16, torch.bfloat16)
    @parametrize("reduce", reductions)
    @parametrize("initial", [None, 0., 2.])
    @parametrize("special", [0., float("nan"), float("inf"), -float("inf")])
    def test_batched_parallel_nonfinite(self, device, dtype, reduce, initial, special):
        data_cpu = torch.ones(2, 259, dtype=dtype)
        data_cpu[0, 1] = special
        data_cpu[0, 2] = special
        data_cpu[1, 2] = special
        data_cpu.requires_grad_()
        data = data_cpu.detach().to(device).requires_grad_()
        offsets = torch.tensor([[1, 65, 65, 259], [0, 64, 201, 259]], dtype=torch.int32)
        expected = torch.segment_reduce(data_cpu, reduce, offsets=offsets, axis=1, initial=initial)
        actual = torch.segment_reduce(data, reduce, offsets=offsets.to(device), axis=1, initial=initial)
        self.assertEqual(actual, expected, equal_nan=True)
        grad = torch.tensor([[1., 0., -1.], [2., 0., -2.]], dtype=dtype)
        expected.backward(grad)
        actual.backward(grad.to(device))
        self.assertEqual(data.grad, data_cpu.grad, equal_nan=True)

    @dtypes(torch.float16)
    def test_initial_overflow(self, device, dtype):
        data = torch.ones(3, dtype=dtype, device=device)
        lengths = torch.tensor([3], device=device)
        with self.assertRaisesRegex(RuntimeError, "cannot be converted"):
            torch.segment_reduce(data, "sum", lengths=lengths, initial=1e10)

    @dtypes(torch.float32)
    @dtypesIfMPS(torch.float32, torch.float16, torch.bfloat16)
    @parametrize("reduce", reductions)
    @parametrize("inner", [1, 33])
    def test_random_segments(self, device, dtype, reduce, inner):
        data_cpu = (torch.randn(2, 259, inner) * 0.025 + 1).to(dtype)
        offsets = torch.tensor([[0, 32, 36, 259], [0, 128, 129, 259]])
        data = data_cpu.to(device).requires_grad_()
        actual = torch.segment_reduce(data, reduce, offsets=offsets.to(device), axis=1, initial=.3)
        # MPS uses float32 accumulation for low-precision inputs.
        reference = data_cpu.float() if torch.device(device).type == "mps" else data_cpu
        initial = torch.tensor(.3, dtype=dtype).item()
        expected = torch.segment_reduce(reference, reduce, offsets=offsets, axis=1, initial=initial).to(dtype)
        self.assertEqual(actual, expected)
        grad = torch.randn_like(expected)
        expected_grad = torch.ops.aten._segment_reduce_backward(
            grad, actual.detach().cpu(), data_cpu, reduce, offsets=offsets, axis=1, initial=.3)
        actual.backward(grad.to(device))
        self.assertEqual(data.grad, expected_grad)

    @dtypes(torch.float32, torch.float16, torch.bfloat16)
    @parametrize("reduce", ["sum", "mean"])
    def test_empty_segment_signed_initial(self, device, dtype, reduce):
        data = torch.ones(128, dtype=dtype, device=device)
        offsets = torch.tensor([0, 0, 128], device=device)
        actual = torch.segment_reduce(data, reduce, offsets=offsets, initial=-0.)
        self.assertEqual(actual[0], 0.)
        self.assertTrue(torch.signbit(actual[0]).item())

    @dtypes(torch.float32)
    @dtypesIfMPS(torch.float32, torch.float16, torch.bfloat16)
    @parametrize("reduce", ["sum", "mean", "prod"])
    def test_long_reduction_accumulator(self, device, dtype, reduce):
        value = 1 + torch.finfo(dtype).eps if reduce == "prod" else 1.
        data = torch.full((4096, 8), value, dtype=dtype, device=device)
        lengths = torch.tensor([4096], device=device)
        actual = torch.segment_reduce(data, reduce, lengths=lengths)
        expected = getattr(data, reduce)(0, keepdim=True)
        self.assertEqual(actual, expected)

    @onlyMPS
    @parametrize("data_dtype", [
        torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
        torch.uint16, torch.uint32, torch.uint64, torch.complex64, torch.float8_e4m3fn,
    ])
    def test_unsupported_mps_data_dtype(self, device, data_dtype):
        data = torch.ones(4).to(data_dtype).to(device)
        offsets = torch.tensor([0, 2, 4], device=device)
        with self.assertRaisesRegex(RuntimeError, "supports float32, float16 and bfloat16"):
            torch.segment_reduce(data, "sum", offsets=offsets)

    @onlyMPS
    @parametrize("metadata_dtype", [
        torch.bool, torch.uint8, torch.int8, torch.int16, torch.uint16, torch.uint32,
        torch.uint64, torch.float16, torch.bfloat16, torch.float32,
    ])
    @parametrize("mode", ["lengths", "offsets"])
    def test_unsupported_mps_metadata_dtype(self, device, metadata_dtype, mode):
        data = torch.ones(4, device=device)
        values = [2, 2] if mode == "lengths" else [0, 2, 4]
        metadata = torch.tensor(values).to(metadata_dtype).to(device)
        with self.assertRaisesRegex(RuntimeError, "must have int32 or int64 dtype"):
            torch.segment_reduce(data, "sum", **{mode: metadata}, unsafe=True)

    @onlyMPS
    @parametrize("mode, values, metadata_dtype", [
        ("offsets", [0, 4, 2, 6], torch.int32),
        ("offsets", [-1, 6], torch.int64),
        ("offsets", [0, 7], torch.int32),
        ("offsets", [0, 2**40], torch.int64),
        ("lengths", [2, -1, 5], torch.int32),
        ("lengths", [2**31 - 1, 2**31 - 1, 8], torch.int32),
    ])
    @parametrize("unsafe", [False, True])
    def test_invalid_mps_boundaries(self, device, mode, values, metadata_dtype, unsafe):
        data = torch.ones(6, device=device)
        metadata = torch.tensor(values, device=device, dtype=metadata_dtype)
        with self.assertRaisesRegex(RuntimeError, "segment_reduce|negative value"):
            torch.segment_reduce(data, "sum", **{mode: metadata}, unsafe=unsafe)
            torch.mps.synchronize()

    @onlyMPS
    @parametrize("shape, lengths, axis", [
        ((2, 6, 0), [[2, 0, 4], [1, 3, 2]], 1),
        ((0, 6, 7), [], 1),
        ((0,), [], 0),
    ])
    @parametrize("mode", ["lengths", "offsets"])
    def test_empty_mps_dimensions(self, device, shape, lengths, axis, mode):
        data = torch.empty(shape, device=device, requires_grad=True)
        metadata = torch.tensor(lengths, dtype=torch.int64, device=device)
        if axis == 1 and shape[0] == 0:
            metadata = metadata.reshape(0, 3)
        segments = metadata.size(-1)
        if mode == "offsets":
            zero_shape = list(metadata.shape)
            zero_shape[-1] = 1
            metadata = torch.cat((metadata.new_zeros(zero_shape), metadata), -1).cumsum(-1)
        actual = torch.segment_reduce(data, "sum", **{mode: metadata}, axis=axis)
        expected_shape = list(shape)
        expected_shape[axis] = segments
        self.assertEqual(actual, torch.empty(expected_shape))
        actual.sum().backward()
        self.assertEqual(data.grad, torch.empty(shape))


instantiate_device_type_tests(TestSegmentReductions, globals(), allow_mps=True)

if __name__ == "__main__":
    run_tests()
