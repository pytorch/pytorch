import torch
from common_utils import _compare_mt_t, _create_random_mask
from maskedtensor import masked_tensor
from maskedtensor.binary import BINARY_NAMES
from maskedtensor.reductions import REDUCE_NAMES
from maskedtensor.unary import UNARY_NAMES
from torch._masked import _combine_input_and_mask
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
)
from torch.testing._internal.common_methods_invocations import (
    binary_ufuncs,
    reduction_ops,
    unary_ufuncs,
)
from torch.testing._internal.common_utils import run_tests, TestCase


def is_unary(op):
    return op.name in UNARY_NAMES


def is_binary(op):
    return op.name in BINARY_NAMES


def is_reduction(op):
    return op.name in REDUCE_NAMES and op.name not in {"all", "mean", "std", "var"}


mt_unary_ufuncs = [op for op in unary_ufuncs if is_unary(op)]
mt_binary_ufuncs = [op for op in binary_ufuncs if is_binary(op)]
mt_reduction_ufuncs = [op for op in reduction_ops if is_reduction(op)]

MASKEDTENSOR_FLOAT_TYPES = {
    torch.float16,
    torch.float32,
    torch.float64,
}


def _test_unary_binary_equality(device, dtype, op, layout=torch.strided):
    samples = op.sample_inputs(device, dtype, requires_grad=True)

    for sample in samples:
        input = sample.input
        sample_args, sample_kwargs = sample.args, sample.kwargs
        mask = (
            _create_random_mask(input.shape, device)
            if "mask" not in sample_kwargs
            else sample_kwargs.pop("mask")
        )

        if layout == torch.sparse_coo:
            mask = mask.to_sparse_coo().coalesce()
            input = input.sparse_mask(mask)
        elif layout == torch.sparse_csr:
            if input.ndim != 2 or mask.ndim != 2:
                continue
            mask = mask.to_sparse_csr()
            input = input.sparse_mask(mask)

        # Binary operations currently only support same size masks
        if is_binary(op):
            if input.shape != sample_args[0].shape:
                continue
            # Binary operations also don't support kwargs right now
            else:
                sample_kwargs = {}

        mt = masked_tensor(input, mask)
        mt_args = [
            masked_tensor(
                arg.sparse_mask(mask) if layout != torch.strided else arg, mask
            )
            if torch.is_tensor(arg)
            else arg
            for arg in sample_args
        ]

        mt_result = op(mt, *mt_args, **sample_kwargs)
        t_result = op(sample.input, *sample_args, **sample_kwargs)

        _compare_mt_t(mt_result, t_result)

        # If the operation is binary, check that lhs = masked, rhs = regular tensor also works
        if is_binary(op):
            mt_result2 = op(mt, *mt_args, **sample_kwargs)
            _compare_mt_t(mt_result2, t_result)


def _test_reduction_equality(device, dtype, op, layout=torch.strided):
    samples = op.sample_inputs(device, dtype, requires_grad=True)

    for sample in samples:
        input = sample.input
        sample_args, sample_kwargs = sample.args, sample.kwargs

        if input.dim() == 0 or input.numel() == 0:
            continue
        # Reduction operations don't support more advanced args/kwargs right now
        if len(sample_args) > 0:
            sample_args = ()
        if len(sample_kwargs) > 0:
            sample_kwargs = {}

        mask = (
            _create_random_mask(input.shape, device)
            if "mask" not in sample_kwargs
            else sample_kwargs.pop("mask")
        )

        if torch.count_nonzero(mask) == 0:
            continue

        tensor_input = _combine_input_and_mask(op.op, input, mask)
        if layout == torch.sparse_coo:
            mask = mask.to_sparse_coo().coalesce()
            input = input.sparse_mask(mask)
        elif layout == torch.sparse_csr:
            if input.ndim != 2 or mask.ndim != 2:
                continue
            mask = mask.to_sparse_csr()
            input = input.sparse_mask(mask)

        mt = masked_tensor(input, mask)
        mt_args = [
            masked_tensor(
                arg.sparse_mask(mask) if layout != torch.strided else arg, mask
            )
            if torch.is_tensor(arg)
            else arg
            for arg in sample_args
        ]

        mt_result = op(mt, *mt_args, **sample_kwargs)
        t_result = op(tensor_input, *sample_args, **sample_kwargs)

        _compare_mt_t(mt_result, t_result)


class TestOperators(TestCase):
    @ops(mt_unary_ufuncs, allowed_dtypes=MASKEDTENSOR_FLOAT_TYPES)  # type: ignore[arg-type]
    def test_unary_core(self, device, dtype, op):
        # Skip tests that don't have len(kwargs) == 0
        skip_variants = {
            "decimals_0",
            "decimals_3",
            "decimals_neg_3",
        }
        if op.name == "round" and op.variant_test_name in skip_variants:
            return
        _test_unary_binary_equality(device, dtype, op)

    @ops(mt_binary_ufuncs, allowed_dtypes=MASKEDTENSOR_FLOAT_TYPES)  # type: ignore[arg-type]
    def test_binary_core(self, device, dtype, op):
        _test_unary_binary_equality(device, dtype, op)

    @ops(mt_reduction_ufuncs, allowed_dtypes=MASKEDTENSOR_FLOAT_TYPES)  # type: ignore[arg-type]
    def test_reduction_all(self, device, dtype, op):
        _test_reduction_equality(device, dtype, op)

    @ops(mt_unary_ufuncs, allowed_dtypes=MASKEDTENSOR_FLOAT_TYPES)  # type: ignore[arg-type]
    def test_unary_core_sparse(self, device, dtype, op):
        # Skip tests that don't have len(kwargs) == 0
        skip_variants = {
            "decimals_0",
            "decimals_3",
            "decimals_neg_3",
        }
        if op.name == "round" and op.variant_test_name in skip_variants:
            return

        _test_unary_binary_equality(device, dtype, op, torch.sparse_coo)
        _test_unary_binary_equality(device, dtype, op, torch.sparse_csr)

    @ops(mt_binary_ufuncs, allowed_dtypes=MASKEDTENSOR_FLOAT_TYPES)  # type: ignore[arg-type]
    def test_binary_core_sparse(self, device, dtype, op):
        _test_unary_binary_equality(device, dtype, op, torch.sparse_coo)
        _test_unary_binary_equality(device, dtype, op, torch.sparse_csr)

    @ops(mt_reduction_ufuncs, allowed_dtypes=MASKEDTENSOR_FLOAT_TYPES)  # type: ignore[arg-type]
    def test_reduction_all_sparse(self, device, dtype, op):
        _test_reduction_equality(device, dtype, op, torch.sparse_coo)
        _test_reduction_equality(device, dtype, op, torch.sparse_csr)


only_for = ("cpu", "cuda")
instantiate_device_type_tests(TestOperators, globals(), only_for=only_for)

if __name__ == "__main__":
    run_tests()
