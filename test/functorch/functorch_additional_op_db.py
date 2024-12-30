import itertools
import unittest
from functools import partial

import torch
from torch.testing._internal.common_dtype import (
    all_types_and_complex_and,
    floating_types,
    floating_types_and,
)
from torch.testing._internal.common_methods_invocations import (
    DecorateInfo,
    OpInfo,
    SampleInput,
)
from torch.testing._internal.common_utils import make_tensor


# List of OpInfos that aren't in PyTorch Core yet.
# They are here because we wanted a fast way of writing OpInfos and may not be
# 100% correct (w.r.t. to dtypes and other options).
# TODO: Figure out how to upstream these, delete them when they're upstreamed

additional_op_db = []

# https://github.com/pytorch/pytorch/pull/61068


def sample_inputs_conv2d(
    has_bias, self, device, dtype, requires_grad, extra_args=(), groups=1
):
    in_ch, out_ch = 6, 4
    inp = make_tensor(
        (2, in_ch * groups, 7, 5),
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        low=-1,
        high=1,
    )
    weight = make_tensor(
        (out_ch * groups, in_ch, 3, 2),
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        low=-1,
        high=1,
    )
    bias = None
    if has_bias:
        bias = make_tensor(
            (out_ch * groups,),
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
            low=-1,
            high=1,
        )
    return [SampleInput(inp, args=((weight, bias) + extra_args))]


additional_op_db.extend(
    [
        OpInfo(
            "nn.functional.conv2d",
            aten_name="conv2d",
            variant_test_name="no_bias",
            supports_autograd=True,
            supports_forward_ad=True,
            sample_inputs_func=partial(sample_inputs_conv2d, False),
            dtypes=floating_types(),
            dtypesIfGPU=floating_types_and(torch.half, torch.bfloat16),
            supports_out=False,
        ),
        OpInfo(
            "nn.functional.conv2d",
            aten_name="conv2d",
            variant_test_name="with_bias",
            supports_autograd=True,
            supports_forward_ad=True,
            sample_inputs_func=partial(sample_inputs_conv2d, True),
            dtypesIfGPU=floating_types_and(torch.half, torch.bfloat16),
            dtypes=floating_types(),
            supports_out=False,
        ),
        OpInfo(
            "nn.functional.conv2d",
            aten_name="conv2d",
            variant_test_name="stride_with_bias",
            supports_autograd=True,
            supports_forward_ad=True,
            sample_inputs_func=partial(sample_inputs_conv2d, True, extra_args=((2, 2))),
            dtypesIfGPU=floating_types_and(torch.half, torch.bfloat16),
            dtypes=floating_types(),
            supports_out=False,
        ),
        OpInfo(
            "nn.functional.conv2d",
            aten_name="conv2d",
            variant_test_name="stride_no_bias",
            supports_autograd=True,
            supports_forward_ad=True,
            sample_inputs_func=partial(
                sample_inputs_conv2d, False, extra_args=((2, 2))
            ),
            dtypesIfGPU=floating_types_and(torch.half, torch.bfloat16),
            dtypes=floating_types(),
            supports_out=False,
        ),
        OpInfo(
            "nn.functional.conv2d",
            aten_name="conv2d",
            variant_test_name="stride_padding_with_bias",
            supports_autograd=True,
            supports_forward_ad=True,
            sample_inputs_func=partial(
                sample_inputs_conv2d, True, extra_args=((2, 2), (1, 1))
            ),
            dtypesIfGPU=floating_types_and(torch.half, torch.bfloat16),
            dtypes=floating_types(),
            supports_out=False,
        ),
        OpInfo(
            "nn.functional.conv2d",
            aten_name="conv2d",
            variant_test_name="stride_padding_no_bias",
            supports_autograd=True,
            supports_forward_ad=True,
            sample_inputs_func=partial(
                sample_inputs_conv2d, False, extra_args=((2, 2), (1, 1))
            ),
            dtypesIfGPU=floating_types_and(torch.half, torch.bfloat16),
            dtypes=floating_types(),
            supports_out=False,
        ),
        OpInfo(
            "nn.functional.conv2d",
            aten_name="conv2d",
            variant_test_name="strided_padding_dilation_with_bias",
            supports_autograd=True,
            supports_forward_ad=True,
            sample_inputs_func=partial(
                sample_inputs_conv2d, True, extra_args=((2, 2), (1, 1), (2, 2))
            ),
            dtypesIfGPU=floating_types_and(torch.half, torch.bfloat16),
            dtypes=floating_types(),
            supports_out=False,
        ),
        OpInfo(
            "nn.functional.conv2d",
            aten_name="conv2d",
            variant_test_name="strided_padding_dilation_no_bias",
            supports_autograd=True,
            supports_forward_ad=True,
            sample_inputs_func=partial(
                sample_inputs_conv2d, True, extra_args=((2, 2), (1, 1), (2, 2))
            ),
            dtypesIfGPU=floating_types_and(torch.half, torch.bfloat16),
            dtypes=floating_types(),
            supports_out=False,
        ),
        OpInfo(
            "nn.functional.conv2d",
            aten_name="conv2d",
            variant_test_name="stride_groups_with_bias",
            supports_autograd=True,
            supports_forward_ad=True,
            sample_inputs_func=partial(
                sample_inputs_conv2d, True, extra_args=((2, 3), 0, 1, 2), groups=2
            ),
            dtypesIfGPU=floating_types_and(torch.half, torch.bfloat16),
            dtypes=floating_types(),
            supports_out=False,
        ),
        OpInfo(
            "nn.functional.conv2d",
            aten_name="conv2d",
            variant_test_name="stride_depthwise_with_bias",
            supports_autograd=True,
            supports_forward_ad=True,
            sample_inputs_func=partial(
                sample_inputs_conv2d, True, extra_args=((2, 3), 0, 1, 6), groups=6
            ),
            dtypesIfGPU=floating_types_and(torch.half, torch.bfloat16),
            dtypes=floating_types(),
            supports_out=False,
        ),
    ]
)


# TODO: PyTorch core has a check for if requires_grad=True or not.
# We actually want to test more things for backward here which is why we have our own
def sample_inputs_embedding(op_info, device, dtype, requires_grad, **kwargs):
    def make_input(shape):
        return make_tensor(
            shape, device=device, dtype=dtype, requires_grad=requires_grad
        )

    def make_long_input(shape, *, low, high):
        return make_tensor(shape, device=device, dtype=torch.long, low=low, high=high)

    M = 20
    S = 5

    def generator():
        # 0-D index tensor
        idx = make_long_input((), low=0, high=M)
        yield SampleInput(
            make_input((M, S)),
            args=(idx,),
        )

        # 1-D index tensor
        idx = make_long_input((S,), low=0, high=M)
        yield SampleInput(
            make_input((M, S)),
            args=(idx,),
        )

        # 2-D index tensor
        idx = make_long_input((S, S), low=0, high=M)
        yield SampleInput(
            make_input((M, S)),
            args=(idx,),
        )

        idx = make_long_input((2, 2), low=0, high=S)
        idx[0, 0] = 2
        idx[1, 1] = 2
        yield SampleInput(
            make_input((S, S)),
            args=(idx,),
            kwargs={"padding_idx": 2},
        )

        idx = make_long_input((2, 2), low=0, high=S)
        idx[0, 0] = 4
        idx[1, 1] = 4
        yield SampleInput(
            make_input((S, S)),
            args=(idx,),
            kwargs={"padding_idx": -1},
        )

        # Scale the gradient based on the inverse frequency of a particular index.
        idx = make_long_input((2, 2), low=0, high=S)
        idx[0, 0] = 1
        idx[0, 1] = 1
        weights = make_input((S, S))
        yield SampleInput(
            weights,
            args=(idx,),
            kwargs={"scale_grad_by_freq": True},
        )

    return list(generator())


additional_op_db.append(
    OpInfo(
        "nn.functional.embedding",
        variant_test_name="functorch",
        # We use lambda to reshuffle the positional arguments.
        # This is because currently only the `input` field of SampleInput
        # is tested in gradient tests.
        op=lambda weight, idx, **kwargs: torch.nn.functional.embedding(
            idx, weight, **kwargs
        ),
        dtypes=floating_types_and(torch.bfloat16, torch.float16),
        sample_inputs_func=sample_inputs_embedding,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        supports_out=False,
    )
)


def sample_inputs_mse_loss(op_info, device, dtype, requires_grad, **kwargs):
    def make_input(shape, requires_grad=requires_grad):
        return make_tensor(
            shape, device=device, dtype=dtype, requires_grad=requires_grad
        )

    rhs_requires_grad = kwargs.get("rhs_requires_grad", requires_grad)
    S = 5

    shapes = ((S, S), (S, S, S), (S, S, S, S))
    reductions = ("none", "mean", "sum")

    for shape, reduction in itertools.product(shapes, reductions):
        yield SampleInput(
            make_input(shape),
            args=(make_input(shape, requires_grad=rhs_requires_grad),),
            kwargs={"reduction": reduction},
        )


additional_op_db.append(
    OpInfo(
        "nn.functional.mse_loss",
        variant_test_name="functorch",
        sample_inputs_func=sample_inputs_mse_loss,
        supports_out=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        dtypes=floating_types_and(torch.float16),
        backward_dtypes=floating_types(),
        dtypesIfGPU=floating_types_and(torch.bfloat16, torch.float16),
        backward_dtypesIfGPU=floating_types_and(torch.bfloat16, torch.float16),
    )
)


# TODO: upstream sample inputs to pytorch/pytorch.
# We are more comprehensive.
def sample_inputs_getitem(op_info, device, dtype, requires_grad, **kwargs):
    # Short for "advanced index"
    adv_idx = torch.LongTensor([[0, 1], [2, 3]])
    S = 5
    # self_dim, indices
    test_args = [
        (3, ([1, 2],)),
        (3, (slice(0, 3),)),
        (3, ([slice(0, 3), 1],)),
        (3, ([[0, 2, 3], [1, 3, 3], [0, 0, 2]],)),
        (3, ([[0, 0, 3], [1, 1, 3], [0, 0, 2]],)),
        (3, ([slice(None), slice(None), [0, 3]],)),
        (3, ([slice(None), [0, 3], slice(None)],)),
        (3, ([[0, 3], slice(None), slice(None)],)),
        (3, ([[0, 3], [1, 2], slice(None)],)),
        (
            3,
            (
                [
                    [0, 3],
                ],
            ),
        ),
        (3, ([[0, 3], slice(None)],)),
        (3, ([[0, 3], Ellipsis],)),
        (3, ([[0, 2, 3], [1, 3, 3], torch.LongTensor([0, 0, 2])],)),
        (4, ([slice(None), adv_idx, adv_idx, slice(None)],)),
        (4, ([slice(None), adv_idx, slice(None), adv_idx],)),
        (4, ([adv_idx, slice(None), slice(None), adv_idx],)),
        (4, ([slice(None), slice(None), adv_idx, adv_idx],)),
        (4, ([Ellipsis, adv_idx, adv_idx],)),
        (5, ([slice(None), slice(None), adv_idx, slice(None), adv_idx],)),
        (5, ([slice(None), slice(None), adv_idx, adv_idx, slice(None)],)),
        (5, ([slice(None), slice(None), adv_idx, None, adv_idx, slice(None)],)),
        (6, ([slice(None), slice(None), slice(None), adv_idx, adv_idx],)),
        (6, ([slice(None), slice(None), adv_idx, adv_idx, adv_idx],)),
        (6, ([slice(None), slice(None), None, adv_idx, adv_idx, adv_idx],)),
    ]

    def get_shape(dim):
        return tuple(S + i for i in range(dim))

    return tuple(
        SampleInput(
            make_tensor(
                get_shape(self_dim),
                device=device,
                dtype=dtype,
                low=None,
                high=None,
                requires_grad=requires_grad,
            ),
            args=args,
        )
        for self_dim, args in test_args
    )


# TODO: split PyTorch's __getitem__. The problem is we don't support indexing
# with masks with vmap.
additional_op_db.append(
    OpInfo(
        "__getitem__",
        variant_test_name="functorch",
        dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
        supports_out=False,
        supports_inplace_autograd=False,
        supports_scripting=False,
        op=torch.Tensor.__getitem__,
        assert_jit_shape_analysis=False,  # TODO: support index.Tensor()
        supports_forward_ad=True,
        sample_inputs_func=sample_inputs_getitem,
    )
)


# Turns out at::index_put is different from torch.index_put...
# TODO: figure out how to upstream this
def sample_inputs_aten_index_put(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(
        make_tensor, dtype=dtype, device=device, requires_grad=requires_grad
    )
    inputs = []
    adv_idx = torch.LongTensor([[0, 1], [2, 3]])
    # self_shape, indices
    additional = [
        ((5, 6, 7, 8), [None, adv_idx, adv_idx, None]),
        ((5, 6, 7, 8), [None, adv_idx, None, adv_idx]),
        ((5, 6, 7, 8), [adv_idx, None, None, adv_idx]),
        ((5, 6, 7, 8), [None, None, adv_idx, adv_idx]),
        ((5, 6, 7, 8, 9), [None, None, adv_idx, None, adv_idx]),
        ((5, 6, 7, 8, 9), [None, None, adv_idx, adv_idx, None]),
        ((5, 6, 7, 8, 9, 10), [None, None, None, adv_idx, adv_idx]),
        ((5, 6, 7, 8, 9, 10), [None, None, adv_idx, adv_idx, adv_idx]),
    ]
    for self_shape, indices in additional:
        for broadcast_value in [False, True]:
            inp = make_arg(self_shape)

            tmp_indices = [slice(None) if idx is None else idx for idx in indices]
            values_shape = inp[tmp_indices].shape
            if broadcast_value:
                values_shape = values_shape[3:]
            values = make_arg(values_shape)
            inputs.append(SampleInput(inp, args=(tuple(indices), values)))
    return inputs


def sample_inputs_index_put(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(
        make_tensor, dtype=dtype, device=device, requires_grad=requires_grad
    )
    make_idx = partial(
        make_tensor, dtype=torch.long, device=device, requires_grad=False
    )
    S = 5
    inputs = []
    for accumulate in [False, True]:
        # putting vectors at indexed locations
        inputs.append(
            SampleInput(
                make_arg((S, S)),
                args=((make_idx((2,), low=0, high=4),), make_arg((2, S))),
                kwargs=dict(accumulate=accumulate),
            )
        )

        # putting multi-dim tensors at indexed locations
        inputs.append(
            SampleInput(
                make_arg((S, S, 2)),
                args=((make_idx((3,), low=0, high=4),), make_arg((3, S, 2))),
                kwargs=dict(accumulate=accumulate),
            )
        )

        # value with size `0` dim
        inputs.append(
            SampleInput(
                make_arg((S, 0)),
                args=((make_idx((3,), low=0, high=4),), make_arg((3, 0))),
                kwargs=dict(accumulate=accumulate),
            )
        )

        # scalar value
        inputs.append(
            SampleInput(
                make_arg((S,)),
                args=((make_idx((), low=0, high=S),), make_arg(())),
                kwargs=dict(accumulate=accumulate),
            )
        )

        # cuda and accumulate don't work well
        # Reference: https://github.com/pytorch/pytorch/issues/72053
        if not accumulate and device == "cuda":
            # Broadcast `values`
            inputs.append(
                SampleInput(
                    make_arg((S, S)),
                    args=((make_idx((2,), low=0, high=S),), make_arg((S,))),
                    kwargs=dict(accumulate=accumulate),
                )
            )

    return inputs


additional_op_db.append(
    OpInfo(
        "index_put",
        variant_test_name="functorch",
        dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
        supports_out=False,
        sample_inputs_func=sample_inputs_index_put,
        supports_forward_ad=True,
    )
)
additional_op_db.append(
    OpInfo(
        "ops.aten.index_put",
        variant_test_name="functorch",
        dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
        supports_out=False,
        sample_inputs_func=sample_inputs_aten_index_put,
        supports_forward_ad=True,
    )
)


def sample_inputs_masked_fill(op_info, device, dtype, requires_grad, **kwargs):
    S = 3
    make_arg = partial(
        make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
    )

    yield SampleInput(make_arg((S, S)), args=(torch.randn(S, S, device=device) > 0, 10))
    yield SampleInput(make_arg((S, S)), args=(torch.randn(S, device=device) > 0, 10))
    yield SampleInput(make_arg(()), args=(torch.randn((), device=device) > 0, 10))
    yield SampleInput(make_arg((S, S)), args=(torch.randn((), device=device) > 0, 10))
    yield SampleInput(
        make_arg((S,)),
        args=(torch.randn(S, S, device=device) > 0, 10),
        broadcasts_input=True,
    )


additional_op_db.append(
    OpInfo(
        "masked_fill",
        variant_test_name="functorch_Scalar_only",
        dtypes=all_types_and_complex_and(
            torch.bool, torch.half, torch.bfloat16, torch.chalf
        ),
        sample_inputs_func=sample_inputs_masked_fill,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        check_batched_forward_grad=False,
        supports_out=False,
    )
)


def sample_inputs_new_zeros_with_same_feature_meta(
    op_info, device, dtype, requires_grad, **kwargs
):
    make_arg = partial(
        make_tensor, dtype=dtype, device=device, requires_grad=requires_grad
    )
    matrix = [
        # tangent, base, num_tangent_bdims
        ([5], [2, 3], 0),
        ([2, 3], [2, 3], 0),
        ([5], [2], 0),
        ([1, 0, 2], [1, 2], 0),
        ([], [1, 2], 0),
        ([8, 7, 5], [2, 3, 11], 1),
        ([6, 7, 5], [2, 3, 4], 2),
        ([6, 4], [3], 2),
    ]
    results = []
    for tangent_shape, base_shape, num_tangent_bdims in matrix:
        tangent = make_arg(tangent_shape)
        base = make_arg(base_shape)
        results.append(
            SampleInput(
                tangent,
                args=(base,),
                kwargs=dict(self_num_batch_dims=num_tangent_bdims),
            )
        )
    return results


additional_op_db.append(
    OpInfo(
        "ops.aten._new_zeros_with_same_feature_meta",
        variant_test_name="functorchonly",
        dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
        supports_out=False,
        supports_autograd=False,
        supports_forward_ad=False,
        sample_inputs_func=sample_inputs_new_zeros_with_same_feature_meta,
    )
)


def sample_inputs_conversion(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(
        make_tensor, dtype=dtype, device=device, requires_grad=requires_grad
    )
    shapes = ((), (2, 3))
    memory_format_options = [None, torch.contiguous_format]
    for shape, memory_format in itertools.product(shapes, memory_format_options):
        yield SampleInput(
            make_arg(shape),
            kwargs={"memory_format": memory_format} if memory_format else {},
        )


additional_op_db.extend(
    [
        OpInfo(
            "bfloat16",
            op=lambda x, *args, **kwargs: x.bfloat16(*args, **kwargs),
            dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
            supports_out=False,
            variant_test_name="functorch_no_channels_last",
            sample_inputs_func=sample_inputs_conversion,
            skips=(
                # autograd tests don't handle operators that change dtype
                DecorateInfo(unittest.expectedFailure, "TestFwdGradients"),
                DecorateInfo(unittest.expectedFailure, "TestBwdGradients"),
                DecorateInfo(
                    unittest.expectedFailure,
                    "TestNormalizeOperators",
                    "test_normalize_operator_exhaustive",
                ),
                # RuntimeError: attribute lookup is not defined on builtin
                DecorateInfo(
                    unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
                ),
                DecorateInfo(
                    unittest.skip("Skipped!"), "TestNNCOpInfo", "test_nnc_correctness"
                ),
            ),
        ),
        OpInfo(
            "bool",
            op=lambda x, *args, **kwargs: x.bool(*args, **kwargs),
            dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
            supports_out=False,
            variant_test_name="functorch_no_channels_last",
            sample_inputs_func=sample_inputs_conversion,
            supports_autograd=False,
            skips=(
                DecorateInfo(
                    unittest.expectedFailure,
                    "TestNormalizeOperators",
                    "test_normalize_operator_exhaustive",
                ),
                # RuntimeError: attribute lookup is not defined on builtin
                DecorateInfo(
                    unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
                ),
            ),
        ),
        OpInfo(
            "byte",
            op=lambda x, *args, **kwargs: x.byte(*args, **kwargs),
            dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
            supports_out=False,
            variant_test_name="functorch_no_channels_last",
            sample_inputs_func=sample_inputs_conversion,
            # The autograd test runner cannot handle functions that change dtype
            supports_autograd=False,
            skips=(
                DecorateInfo(
                    unittest.expectedFailure,
                    "TestNormalizeOperators",
                    "test_normalize_operator_exhaustive",
                ),
                # RuntimeError: attribute lookup is not defined on builtin
                DecorateInfo(
                    unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
                ),
            ),
        ),
        OpInfo(
            "char",
            op=lambda x, *args, **kwargs: x.char(*args, **kwargs),
            dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
            supports_out=False,
            variant_test_name="functorch_no_channels_last",
            sample_inputs_func=sample_inputs_conversion,
            # The autograd test runner cannot handle functions that change dtype
            supports_autograd=False,
            skips=(
                DecorateInfo(
                    unittest.expectedFailure,
                    "TestNormalizeOperators",
                    "test_normalize_operator_exhaustive",
                ),
                # RuntimeError: attribute lookup is not defined on builtin
                DecorateInfo(
                    unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
                ),
            ),
        ),
        OpInfo(
            "double",
            op=lambda x, *args, **kwargs: x.double(*args, **kwargs),
            dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
            supports_out=False,
            variant_test_name="functorch_no_channels_last",
            sample_inputs_func=sample_inputs_conversion,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
            skips=(
                DecorateInfo(
                    unittest.expectedFailure,
                    "TestNormalizeOperators",
                    "test_normalize_operator_exhaustive",
                ),
                # RuntimeError: attribute lookup is not defined on builtin
                DecorateInfo(
                    unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
                ),
            ),
        ),
        OpInfo(
            "float",
            op=lambda x, *args, **kwargs: x.float(*args, **kwargs),
            dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
            supports_out=False,
            variant_test_name="functorch_no_channels_last",
            sample_inputs_func=sample_inputs_conversion,
            skips=(
                # autograd tests don't handle operators that change dtype
                DecorateInfo(unittest.expectedFailure, "TestFwdGradients"),
                DecorateInfo(unittest.expectedFailure, "TestBwdGradients"),
                DecorateInfo(
                    unittest.expectedFailure,
                    "TestNormalizeOperators",
                    "test_normalize_operator_exhaustive",
                ),
                # RuntimeError: attribute lookup is not defined on builtin
                DecorateInfo(
                    unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
                ),
            ),
        ),
        OpInfo(
            "half",
            op=lambda x, *args, **kwargs: x.half(*args, **kwargs),
            dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
            supports_out=False,
            variant_test_name="functorch_no_channels_last",
            sample_inputs_func=sample_inputs_conversion,
            skips=(
                # autograd tests don't handle operators that change dtype
                DecorateInfo(unittest.expectedFailure, "TestFwdGradients"),
                DecorateInfo(unittest.expectedFailure, "TestBwdGradients"),
                DecorateInfo(
                    unittest.expectedFailure,
                    "TestNormalizeOperators",
                    "test_normalize_operator_exhaustive",
                ),
                # RuntimeError: attribute lookup is not defined on builtin
                DecorateInfo(
                    unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
                ),
            ),
        ),
        OpInfo(
            "int",
            op=lambda x, *args, **kwargs: x.int(*args, **kwargs),
            dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
            supports_out=False,
            variant_test_name="functorch_no_channels_last",
            sample_inputs_func=sample_inputs_conversion,
            supports_autograd=False,
            skips=(
                DecorateInfo(
                    unittest.expectedFailure,
                    "TestNormalizeOperators",
                    "test_normalize_operator_exhaustive",
                ),
                # RuntimeError: attribute lookup is not defined on builtin
                DecorateInfo(
                    unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
                ),
            ),
        ),
        OpInfo(
            "long",
            op=lambda x, *args, **kwargs: x.long(*args, **kwargs),
            dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
            supports_out=False,
            variant_test_name="functorch_no_channels_last",
            sample_inputs_func=sample_inputs_conversion,
            supports_autograd=False,
            skips=(
                DecorateInfo(
                    unittest.expectedFailure,
                    "TestNormalizeOperators",
                    "test_normalize_operator_exhaustive",
                ),
                # RuntimeError: attribute lookup is not defined on builtin
                DecorateInfo(
                    unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
                ),
            ),
        ),
        OpInfo(
            "short",
            op=lambda x, *args, **kwargs: x.short(*args, **kwargs),
            dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
            supports_out=False,
            variant_test_name="functorch_no_channels_last",
            sample_inputs_func=sample_inputs_conversion,
            supports_autograd=False,
            skips=(
                DecorateInfo(
                    unittest.expectedFailure,
                    "TestNormalizeOperators",
                    "test_normalize_operator_exhaustive",
                ),
                # RuntimeError: attribute lookup is not defined on builtin
                DecorateInfo(
                    unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
                ),
            ),
        ),
    ]
)
