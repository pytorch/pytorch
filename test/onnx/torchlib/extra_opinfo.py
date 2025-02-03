# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Test data for aten operators which don't exist in PyTorch file:
pytorch/torch/testing/_internal/common_methods_invocations.py.
"""

import functools
import itertools
from typing import Any, List

import torch
from torch import testing as torch_testing
from torch.testing._internal import (
    common_device_type,
    common_dtype,
    common_methods_invocations,
)
from torch.testing._internal.opinfo import core as opinfo_core


S = 5
M = 10


def sample_inputs_scalar_tensor(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    del kwargs
    del device
    del requires_grad
    # Not including a scalar tensor in vals because meta tests start failing due to
    # lack of meta support for _local_scalar_dense
    # torch.tensor(2, device=device)
    vals = (-5j, 0j, 1j)

    for item in vals:
        yield opinfo_core.SampleInput(item, dtype=dtype)


def sample_inputs_bernoulli_p(op_info, device, dtype, requires_grad, **kwargs):
    del op_info

    shapes = [
        [3],
        [],
        [3, 2],
        [2, 3, 2],
    ]

    for shape in shapes:
        for p in (0, 0.5, 1):
            t = torch_testing.make_tensor(
                shape,
                low=0,
                high=1,
                device=device,
                dtype=dtype,
                requires_grad=requires_grad,
                **kwargs,
            )
            yield opinfo_core.SampleInput(t, args=(p,))
            yield opinfo_core.SampleInput(t, kwargs={"p": p})


def sample_inputs_bernoulli_p_deterministic(
    op_info, device, dtype, requires_grad, **kwargs
):
    del op_info

    shapes = [
        [3],
        [],
        [3, 2],
        [2, 3, 2],
    ]

    for shape in shapes:
        for p in (0, 1):
            t = torch_testing.make_tensor(
                shape,
                low=0,
                high=1,
                device=device,
                dtype=dtype,
                requires_grad=requires_grad,
                **kwargs,
            )
            yield opinfo_core.SampleInput(t, args=(p,))
            yield opinfo_core.SampleInput(t, kwargs={"p": p})


def sample_inputs_col2im(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    # input_shape, output_size, kernal, dilation, padding, stride
    cases = (
        (
            (1, 12, 12),
            (4, 5),
            (2, 2),
            {"dilation": (1, 1), "padding": (0, 0), "stride": (1, 1)},
        ),
        (
            (1, 8, 30),
            (4, 5),
            (2, 2),
            {"dilation": (1, 1), "padding": (1, 1), "stride": (1, 1)},
        ),
        (
            (1, 8, 9),
            (4, 4),
            (2, 2),
            {"dilation": (1, 1), "padding": (0, 0), "stride": (1, 1)},
        ),
        (
            (1, 8, 25),
            (4, 4),
            (2, 2),
            {"dilation": (1, 1), "padding": (1, 1), "stride": (1, 1)},
        ),
        (
            (1, 8, 9),
            (4, 4),
            (2, 2),
            {"dilation": (1, 1), "padding": (1, 1), "stride": (2, 2)},
        ),
        (
            (1, 9, 4),
            (4, 4),
            (3, 3),
            {"dilation": (1, 1), "padding": (1, 1), "stride": (2, 2)},
        ),
        (
            (1, 18, 16),
            (2, 2),
            (1, 1),
            {"dilation": (2, 2), "padding": (3, 3), "stride": (2, 2)},
        ),
    )

    make_arg = functools.partial(
        torch_testing.make_tensor,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    for shape, output_size, kernel_size, kwargs in cases:
        tensor = make_arg(shape)
        yield opinfo_core.SampleInput(
            tensor, args=(output_size, kernel_size), kwargs=kwargs
        )


def sample_inputs_conv3d(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    make_arg = functools.partial(
        torch_testing.make_tensor,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )

    # Ordered as shapes for input, weight, bias,
    # and a dict of values of (stride, padding, dilation, groups)
    cases: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], dict[str, Any]] = (  # type: ignore[assignment]
        (
            (1, 3, 3, 224, 224),
            (32, 3, 3, 3, 3),
            None,
            {
                "stride": (2, 2, 2),
                "padding": (1, 1, 1),
                "dilation": (1, 1, 1),
                "groups": 1,
            },
        ),
        (
            (2, 4, 3, 56, 56),
            (32, 4, 3, 3, 3),
            (32,),
            {
                "stride": (3, 3, 3),
                "padding": (2, 2, 2),
                "dilation": (1, 1, 1),
                "groups": 1,
            },
        ),
    )

    for input_shape, weight, bias, kwargs in cases:  # type: ignore[assignment]
        # Batched
        yield opinfo_core.SampleInput(
            make_arg(input_shape),
            args=(make_arg(weight), make_arg(bias) if bias is not None else bias),
            kwargs=kwargs,
        )
        # Unbatched
        yield opinfo_core.SampleInput(
            make_arg(input_shape[1:]),  # type: ignore[index]
            args=(make_arg(weight), make_arg(bias) if bias is not None else bias),
            kwargs=kwargs,
        )


def sample_inputs_convolution(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    make_arg = functools.partial(
        torch_testing.make_tensor,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )

    # Ordered as shapes for input, weight, bias,
    # and a dict of values of (stride, padding, dilation, groups)
    cases: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], dict[str, Any]] = (  # type: ignore[assignment]
        (
            (1, 3, 4),
            (3, 3, 3),
            (3,),
            {
                "stride": (2,),
                "padding": (2,),
                "dilation": (1,),
                "transposed": False,
                "output_padding": (0,),
                "groups": 1,
            },
        ),
        (
            (1, 3, 4),
            (3, 3, 3),
            None,
            {
                "stride": (2,),
                "padding": (2,),
                "dilation": (1,),
                "transposed": True,
                "output_padding": (0,),
                "groups": 1,
            },
        ),
        (
            (1, 3, 224, 224),
            (32, 3, 3, 3),
            None,
            {
                "stride": (2, 2),
                "padding": (1, 1),
                "dilation": (1, 1),
                "transposed": False,
                "output_padding": (0, 0),
                "groups": 1,
            },
        ),
        (
            (1, 3, 3, 224, 224),
            (32, 3, 3, 3, 3),
            (32,),
            {
                "stride": (2, 2, 2),
                "padding": (1, 1, 1),
                "dilation": (1, 1, 1),
                "transposed": False,
                "output_padding": (0, 0, 0),
                "groups": 1,
            },
        ),
        # FIXME(jiz): Uncomment out these test data once
        # torch 2.0 is released.
        # (
        #     (1, 3, 224, 224, 224),
        #     (32, 3, 3, 3, 3),
        #     (32,),
        #     {
        #         "stride": (2, 2, 2),
        #         "padding": (1, 1, 1),
        #         "dilation": (1, 1, 1),
        #         "transposed": False,
        #         "output_padding": (0, 0, 0),
        #         "groups": 1,
        #     },
        # ),
        (
            (2, 4, 6, 6),
            (4, 1, 3, 3),
            (4,),
            {
                "stride": (3, 2),
                "padding": (1, 1),
                "dilation": (1, 1),
                "transposed": True,
                "output_padding": (0, 0),
                "groups": 4,
            },
        ),
    )

    for input_shape, weight, bias, kwargs in cases:  # type: ignore[assignment]
        yield opinfo_core.SampleInput(
            make_arg(input_shape),
            args=(make_arg(weight), make_arg(bias) if bias is not None else bias),
            kwargs=kwargs,
        )


def sample_inputs_embedding_renorm(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    del kwargs

    def make_input(shape):
        return common_methods_invocations.make_tensor(
            shape, device=device, dtype=dtype, requires_grad=requires_grad
        )

    def make_long_input(shape, *, low, high, noncontiguous=False):
        return common_methods_invocations.make_tensor(
            shape,
            device=device,
            dtype=torch.long,
            low=low,
            high=high,
            noncontiguous=noncontiguous,
        )

    for max_norm in (0.5, 1.0, 5.0):
        for norm_type in (0.8, 1.0, 2.0, 2.5):
            idx = make_long_input((6,), low=0, high=S)
            weights = make_input((S, S)) * 2
            yield common_methods_invocations.SampleInput(
                weights,
                args=(idx,),
                kwargs={"max_norm": max_norm, "norm_type": norm_type},
            )


def sample_inputs_embedding_bag(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    del kwargs

    def make_input(shape):
        return common_methods_invocations.make_tensor(
            shape, device=device, dtype=dtype, requires_grad=requires_grad
        )

    def make_long_input(shape, *, low, high, noncontiguous=False):
        return common_methods_invocations.make_tensor(
            shape,
            device=device,
            dtype=torch.long,
            low=low,
            high=high,
            noncontiguous=noncontiguous,
        )

    def make_per_sample_weight(flag, idx):
        # a tensor of float / double weights, or None
        # to indicate all weights should be taken to be 1
        if flag:
            return make_input(idx.reshape(-1).shape)
        return None

    offsets = [
        torch.tensor([0, 2, 3], device=device, dtype=torch.long),
        torch.tensor([0, 0, 2], device=device, dtype=torch.long),
        torch.tensor([0, 2, 2, 4], device=device, dtype=torch.long),
    ]
    for offset in offsets:
        for include_last_offset in (True, False):
            for generate_per_sample_weight in (True, False):
                for mode in (
                    0,
                    1,
                    2,
                ):  # ('sum', 'mean', 'max')
                    # per_sample_weights only support mode='sum'
                    if generate_per_sample_weight and mode in (
                        1,
                        2,
                    ):  # ('mean', 'max'):
                        continue

                    # 1-D index tensor
                    indices = make_long_input((S,), low=0, high=M)
                    per_sample_weights = make_per_sample_weight(
                        generate_per_sample_weight, indices
                    )
                    # 0
                    yield common_methods_invocations.SampleInput(
                        make_input((M, S)),
                        args=(indices,),
                        kwargs={
                            "offsets": offset,
                            "mode": mode,
                            "per_sample_weights": per_sample_weights,
                            "include_last_offset": include_last_offset,
                        },
                    )

                    indices = make_long_input((S,), low=0, high=M, noncontiguous=True)
                    per_sample_weights = make_per_sample_weight(
                        generate_per_sample_weight, indices
                    )
                    # 1
                    yield common_methods_invocations.SampleInput(
                        make_input((M, S)),
                        args=(indices,),
                        kwargs={
                            "offsets": offset,
                            "mode": mode,
                            "per_sample_weights": per_sample_weights,
                            "include_last_offset": include_last_offset,
                        },
                    )

                    if mode != 2:  # "max" mode in 2-D index tensor make aten func crash
                        # 2-D index tensor
                        indices = make_long_input((S, S), low=0, high=M)
                        per_sample_weights = make_per_sample_weight(
                            generate_per_sample_weight, indices
                        )
                        # 2
                        yield common_methods_invocations.SampleInput(
                            make_input((M, S)),
                            args=(indices,),
                            kwargs={
                                "offsets": offset,
                                "mode": mode,
                                "per_sample_weights": per_sample_weights,
                                "include_last_offset": include_last_offset,
                            },
                        )

                        indices = make_long_input(
                            (S, S), low=0, high=M, noncontiguous=True
                        )
                        per_sample_weights = make_per_sample_weight(
                            generate_per_sample_weight, indices
                        )
                        # 3
                        yield common_methods_invocations.SampleInput(
                            make_input((M, S)),
                            args=(indices,),
                            kwargs={
                                "offsets": offset,
                                "mode": mode,
                                "per_sample_weights": per_sample_weights,
                                "include_last_offset": include_last_offset,
                            },
                        )


def sample_inputs_embedding_bag_padding_idx(
    op_info, device, dtype, requires_grad, **kwargs
):
    del op_info
    del kwargs

    def make_input(shape):
        return common_methods_invocations.make_tensor(
            shape, device=device, dtype=dtype, requires_grad=requires_grad
        )

    def make_long_input(shape, *, low, high, noncontiguous=False):
        return common_methods_invocations.make_tensor(
            shape,
            device=device,
            dtype=torch.long,
            low=low,
            high=high,
            noncontiguous=noncontiguous,
        )

    def make_per_sample_weight(flag, idx):
        # a tensor of float / double weights, or None
        # to indicate all weights should be taken to be 1
        if flag:
            return make_input(idx.reshape(-1).shape)
        return None

    offsets = [
        torch.tensor([0, 2, 3], device=device, dtype=torch.long),
        # Below case not work for FullGraph mode, guess due to op.While() bug:
        # when the initial condition is False, it still excute the loop body once.
        # torch.tensor([0, 0, 2], device=device, dtype=torch.long),
        # torch.tensor([0, 2, 2, 4], device=device, dtype=torch.long),
    ]
    for offset in offsets:
        for include_last_offset in (True, False):
            for generate_per_sample_weight in (True, False):
                for mode in (
                    0,
                    1,
                    2,
                ):  # ('sum', 'mean', 'max')
                    # per_sample_weights only support mode='sum'
                    if generate_per_sample_weight and mode in (
                        1,
                        2,
                    ):  # ('mean', 'max'):
                        continue

                    for padding_idx in (-1, 0, 1, 2, 3):
                        # 1-D index tensor
                        indices = make_long_input((S,), low=0, high=M)
                        per_sample_weights = make_per_sample_weight(
                            generate_per_sample_weight, indices
                        )
                        # 0
                        yield common_methods_invocations.SampleInput(
                            make_input((M, S)),
                            args=(indices,),
                            kwargs={
                                "offsets": offset,
                                "scale_grad_by_freq": False,
                                "mode": mode,
                                "sparse": False,
                                "per_sample_weights": per_sample_weights,
                                "include_last_offset": include_last_offset,
                                "padding_idx": padding_idx,
                            },
                        )

                        indices = make_long_input(
                            (S,), low=0, high=M, noncontiguous=True
                        )
                        per_sample_weights = make_per_sample_weight(
                            generate_per_sample_weight, indices
                        )
                        # 1
                        yield common_methods_invocations.SampleInput(
                            make_input((M, S)),
                            args=(indices,),
                            kwargs={
                                "offsets": offset,
                                "scale_grad_by_freq": False,
                                "mode": mode,
                                "sparse": False,
                                "per_sample_weights": per_sample_weights,
                                "include_last_offset": include_last_offset,
                                "padding_idx": padding_idx,
                            },
                        )

                        # if mode != 2:  # "max" mode in 2-D index tensor make aten func crash
                        #     # 2-D index tensor
                        #     indices = make_long_input((S, S), low=0, high=M)
                        #     per_sample_weights = make_per_sample_weight(
                        #         generate_per_sample_weight, indices
                        #     )
                        #     # 2
                        #     yield common_methods_invocations.SampleInput(
                        #         make_input((M, S)),
                        #         args=(indices,),
                        #         kwargs={
                        #             "offsets": offset,
                        #             "mode": mode,
                        #             "per_sample_weights": per_sample_weights,
                        #             "include_last_offset": include_last_offset,
                        #             "padding_idx": padding_idx,
                        #         },
                        #     )

                        #     indices = make_long_input((S, S), low=0, high=M, noncontiguous=True)
                        #     per_sample_weights = make_per_sample_weight(
                        #         generate_per_sample_weight, indices
                        #     )
                        #     # 3
                        #     yield common_methods_invocations.SampleInput(
                        #         make_input((M, S)),
                        #         args=(indices,),
                        #         kwargs={
                        #             "offsets": offset,
                        #             "mode": mode,
                        #             "per_sample_weights": per_sample_weights,
                        #             "include_last_offset": include_last_offset,
                        #             "padding_idx": padding_idx,
                        #         },
                        #     )


def sample_inputs__local_scalar_dense(op_info, device, dtype, requires_grad, **kwargs):
    del op_info

    shapes = (
        (),
        (1,),
        (3,),
        (1, 1),
        (1, 2),
        (2, 1),
        (1, 1, 1),
        (2, 2, 2),
    )

    for shape in shapes:
        t = torch_testing.make_tensor(
            shape,
            low=0,
            high=1,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
            **kwargs,
        )
        yield opinfo_core.SampleInput(t)


def _prepare_data_for_fft_ops(device, dtype, requires_grad=False):
    # Adapted from https://github.com/pytorch/pytorch/blob/01069ad4be449f376cf88a56d842b8eb50f6e9b6/torch/testing/_internal/opinfo/core.py#L2448C1-L2541C79
    is_fp16_or_chalf = dtype in (torch.complex32, torch.half)
    if not is_fp16_or_chalf:
        oned_tensor = functools.partial(
            opinfo_core.make_tensor,
            (31,),
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        nd_tensor = functools.partial(
            opinfo_core.make_tensor,
            (S, S + 1, S + 2),
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )
    else:
        low = None
        high = None
        shapes = ((2, 8, 9), (33,))

        oned_tensor = functools.partial(
            opinfo_core.make_tensor,
            shapes[1],
            device=device,
            low=low,
            high=high,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        nd_tensor = functools.partial(
            opinfo_core.make_tensor,
            shapes[0],
            device=device,
            low=low,
            high=high,
            dtype=dtype,
            requires_grad=requires_grad,
        )

    return oned_tensor, nd_tensor


def sample_inputs__fft_c2c(self, device, dtype, requires_grad=False, **_):
    del self  # Unused
    oned_tensor, nd_tensor = _prepare_data_for_fft_ops(device, dtype, requires_grad)

    for normalization, forward in itertools.product((0, 1, 2), (True, False)):
        # 1-D
        yield opinfo_core.SampleInput(
            oned_tensor(), dim=(0,), normalization=normalization, forward=forward
        )
        # N-D
        for dim in [
            (0,),
            (1,),
            (2,),
            (1, 2),
            (0, 1),
            (0, 1, 2),
        ]:
            yield opinfo_core.SampleInput(
                nd_tensor(), dim=dim, normalization=normalization, forward=forward
            )


def sample_inputs__fft_r2c(self, device, dtype, requires_grad=False, **_):
    del self  # Unused
    oned_tensor, nd_tensor = _prepare_data_for_fft_ops(device, dtype, requires_grad)

    for normalization, one_sided in itertools.product((0, 1, 2), (True, True)):
        # 1-D
        yield opinfo_core.SampleInput(
            oned_tensor(), dim=(0,), normalization=normalization, onesided=one_sided
        )
        # N-D
        for dim in [
            (0,),
            (1,),
            (2,),
            (1, 2),
            (0, 1),
            (0, 1, 2),
        ]:
            yield opinfo_core.SampleInput(
                nd_tensor(), dim=dim, normalization=normalization, onesided=one_sided
            )


def sample_inputs__fft_c2r(self, device, dtype, requires_grad=False, **_):
    del self  # Unused
    oned_tensor, nd_tensor = _prepare_data_for_fft_ops(device, dtype, requires_grad)

    for normalization in (0, 1, 2):
        # 1-D
        yield opinfo_core.SampleInput(
            oned_tensor(), dim=(0,), normalization=normalization, last_dim_size=12
        )
        # N-D
        for dim in [
            (0,),
            (1,),
            (2,),
            (1, 2),
            (0, 1),
            (0, 1, 2),
        ]:
            yield opinfo_core.SampleInput(
                nd_tensor(), dim=dim, normalization=normalization, last_dim_size=6
            )


def _index_variable_bool(shape, max_indices, device):
    if not isinstance(shape, tuple):
        shape = (shape,)
    index = (
        torch.rand(*shape, dtype=torch.double, device=device)
        .mul_(max_indices)
        .floor_()
        .bool()
    )
    return index


def sample_inputs_index_bool(op_info, device, dtype, requires_grad, **kwargs):
    del op_info  # Unused
    del kwargs  # Unused
    make_arg = functools.partial(
        torch_testing.make_tensor,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    s = 5
    index_bool = _index_variable_bool(s, s, device=device)
    index_bool_2d = _index_variable_bool((s, s), s, device=device)
    index_bool_3d = _index_variable_bool((s, s, s), s, device=device)
    test_args = [
        ([index_bool],),
        ([None, index_bool],),
        ([None, None, None, index_bool],),
        ([index_bool, None],),
        ([index_bool, None, None],),
        # Extra index
        ([None, index_bool, None, index_bool],),
        ([index_bool, None, index_bool, None],),
        ([None, index_bool, index_bool, None],),
        ([index_bool_2d],),
        ([index_bool_2d, None],),
        ([index_bool_2d, None, None],),
        ([None, index_bool_2d],),
        ([None, None, index_bool_2d],),
        ([index_bool_3d],),
        ([index_bool_3d, None],),
        ([None, index_bool_3d],),
    ]

    for args in test_args:
        yield opinfo_core.SampleInput(make_arg((s, s, s, s)), args=args)


def sample_inputs_index(op_info, device, dtype, requires_grad, **kwargs):
    del op_info  # Unused
    del kwargs  # Unused
    make_arg = functools.partial(
        torch_testing.make_tensor,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    s = 5
    index_1d = common_methods_invocations.index_variable(2, s, device=device)
    index_2d = common_methods_invocations.index_variable((s + 1, 2), s, device=device)
    index_3d = common_methods_invocations.index_variable(
        (s + 2, s + 1, 2), s, device=device
    )
    test_args = [
        ([index_1d],),
        ([None, index_1d],),
        ([None, None, None, index_1d],),
        ([index_1d, None],),
        ([index_1d, None, None],),
        # Extra index
        ([None, index_1d, None, index_1d],),
        ([index_1d, None, index_1d, None],),
        ([None, index_1d, index_1d, None],),
        ([index_2d],),
        ([None, index_2d],),
        ([None, None, None, index_2d],),
        ([index_2d, None],),
        ([index_2d, None, None],),
        # Extra index
        ([None, index_2d, None, index_2d],),
        ([index_2d, None, index_2d, None],),
        ([None, index_2d, index_2d, None],),
        ([index_3d],),
        ([None, index_3d],),
        ([None, None, None, index_3d],),
        ([index_3d, None],),
        ([index_3d, None, None],),
        # Extra index
        ([None, index_3d, None, index_3d],),
        ([index_3d, None, index_3d, None],),
        ([None, index_3d, index_3d, None],),
        # Mixed indices
        ([None, index_3d, index_1d, index_2d],),
        # All indices are not None
        ([index_2d, index_3d, index_1d],),
        ([index_2d, index_3d, index_1d, index_2d],),
    ]

    for args in test_args:
        yield opinfo_core.SampleInput(make_arg((s, s, s, s)), args=args)


def sample_inputs_index_put(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    del kwargs

    data = torch_testing.make_tensor(
        (10, 3),
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    indices = [torch.arange(8, dtype=torch.int64, device=device).reshape((-1, 4))]
    values = torch_testing.make_tensor(
        (2, 4, 3),
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    yield opinfo_core.SampleInput(data, indices, values)


def sample_inputs_layer_norm(op_info, device, dtype, requires_grad, **kwargs):
    del op_info  # unused
    del kwargs
    make_arg = functools.partial(
        torch_testing.make_tensor,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )

    # Ordered as input shape, normalized_shape, eps
    cases: tuple[tuple[int], tuple[int], float] = (  # type: ignore[assignment]
        ((1, 2, 3), (1, 2, 3), 0.5),
        ((2, 2, 3), (2, 3), -0.5),
        ((1,), (1,), 1e-5),
        ((1, 2), (2,), 1e-5),
        ((0, 1), (1,), 1e-5),
    )

    for input_shape, normalized_shape, eps in cases:  # type: ignore[misc]
        # Shape of weight and bias should be the same as normalized_shape
        weight = make_arg(normalized_shape)  # type: ignore[has-type]
        bias = make_arg(normalized_shape)  # type: ignore[has-type]
        yield opinfo_core.SampleInput(
            make_arg(input_shape),  # type: ignore[has-type]
            args=(normalized_shape, weight, bias, eps),  # type: ignore[has-type]
        )
        yield opinfo_core.SampleInput(
            make_arg(input_shape),  # type: ignore[has-type]
            args=(normalized_shape, None, bias, eps),  # type: ignore[has-type]
        )
        yield opinfo_core.SampleInput(
            make_arg(input_shape),  # type: ignore[has-type]
            args=(normalized_shape, weight, None, eps),  # type: ignore[has-type]
        )
        yield opinfo_core.SampleInput(
            make_arg(input_shape),  # type: ignore[has-type]
            args=(normalized_shape, None, None, eps),  # type: ignore[has-type]
        )


def sample_inputs_like_fns(self, device, dtype, requires_grad, **kwargs):
    del self  # Unused

    inputs = [
        ((), {}),
        ((S, S), {}),
        ((0, S, 0), {}),
        ((S,), {}),
        ((S,), {"dtype": dtype}),
        # Hard-code some dtypes/devices. We want to test cases where the
        # (dtype, device) is different from the input's (dtype, device)
        ((S,), {"dtype": torch.double}),
    ]
    for shape, kwargs in inputs:
        t = torch_testing.make_tensor(
            shape,
            dtype=dtype,
            device=device,
            low=None,
            high=None,
            requires_grad=requires_grad,
        )
        yield opinfo_core.SampleInput(t, **kwargs)


def sample_inputs__log_softmax(
    op_info,
    device,
    dtype,
    requires_grad,
    **kwargs,
):
    del op_info  # Unused

    make_arg = functools.partial(
        torch_testing.make_tensor,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    cases = [
        ((S,), (0,)),
        ((S, S), (0,)),
        ((S, S), (1,)),
        ((S, S), (-1,)),
        ((S, M, S), (2,)),
        ((S, 0, 0), (-1,)),
    ]

    for (shape, dim), half_to_float in itertools.product(cases, (False,)):
        # NOTE: softmax with half to float conversion is not supported on CPU
        # So we don't test it here
        kwargs = dict(half_to_float=half_to_float)
        yield opinfo_core.SampleInput(make_arg(shape), args=dim, kwargs=kwargs)


def sample_inputs_max_pool_empty_strides(
    op_info, device, dtype, requires_grad, **kwargs
):
    make_arg = functools.partial(
        torch_testing.make_tensor, device=device, dtype=dtype, requires_grad=False
    )

    # FIXME: (RuntimeError: non-empty 3D or 4D (batch mode) tensor expected for input)

    params_generator_type_dict = {
        "ops.aten.max_pool1d": _TestParamsMaxPool1dEmptyStride,
        "ops.aten.max_pool2d": _TestParamsMaxPool2dEmptyStride,
        "ops.aten.max_pool3d": _TestParamsMaxPool3dEmptyStride,
    }

    params_generator = params_generator_type_dict[op_info.name]()
    for (shape, memory_format), kwargs in params_generator.gen_input_params():
        arg = (
            make_arg(shape)
            .to(memory_format=memory_format)
            .requires_grad_(requires_grad)
        )
        yield opinfo_core.SampleInput(arg, kwargs=kwargs)


def sample_inputs_max_pool1d_with_indices(
    op_info, device, dtype, requires_grad, **kwargs
):
    del op_info
    make_arg = functools.partial(
        torch_testing.make_tensor, device=device, dtype=dtype, requires_grad=False
    )
    params_generator = (
        common_methods_invocations._TestParamsMaxPool1d()  # pylint: disable=protected-access
    )
    for (shape, memory_format), kwargs in params_generator.gen_input_params():
        arg = (
            make_arg(shape)
            .to(memory_format=memory_format)
            .requires_grad_(requires_grad)
        )
        yield opinfo_core.SampleInput(arg, kwargs=kwargs)


def sample_inputs_max_pool2d_with_indices(
    op_info, device, dtype, requires_grad, **kwargs
):
    del op_info
    make_arg = functools.partial(
        torch_testing.make_tensor, device=device, dtype=dtype, requires_grad=False
    )
    params_generator = (
        common_methods_invocations._TestParamsMaxPool2d()  # pylint: disable=protected-access
    )
    for (shape, memory_format), kwargs in params_generator.gen_input_params():
        arg = (
            make_arg(shape)
            .to(memory_format=memory_format)
            .requires_grad_(requires_grad)
        )
        yield opinfo_core.SampleInput(arg, kwargs=kwargs)


def sample_inputs_max_pool3d_with_indices(
    op_info, device, dtype, requires_grad, **kwargs
):
    del op_info
    make_arg = functools.partial(
        torch_testing.make_tensor, device=device, dtype=dtype, requires_grad=False
    )
    params_generator = (
        common_methods_invocations._TestParamsMaxPool3d()  # pylint: disable=protected-access
    )
    for (shape, memory_format), kwargs in params_generator.gen_input_params():
        arg = (
            make_arg(shape)
            .to(memory_format=memory_format)
            .requires_grad_(requires_grad)
        )
        yield opinfo_core.SampleInput(arg, kwargs=kwargs)


def sample_inputs_native_group_norm(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    make_arg = functools.partial(
        torch_testing.make_tensor,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )

    # Ordered as input shape, C,N,HxW, and kwargs for group and eps
    cases = (
        ((1, 6, 3), (6,), (6,), 1, 6, 3, {"group": 2, "eps": 0.5}),
        ((2, 6, 3), (6,), (6,), 2, 6, 3, {"group": 3, "eps": -0.5}),
        ((5, 5, 5), (5,), (5,), 5, 5, 5, {"group": 1, "eps": 1e-5}),
        ((5, 8, 10), (8,), (8,), 5, 8, 10, {"group": 4, "eps": 1e-5}),
    )

    for input_shape, weight, bias, N, C, HxW, kwargs in cases:
        # args: running mean, running var, weight and bias should necessarily be of shape: (channels,)
        channels = input_shape[1] if len(input_shape) > 1 else 0
        weight = make_arg(channels) if channels > 0 else None
        bias = make_arg(channels) if channels > 0 else None

        yield opinfo_core.SampleInput(
            make_arg(input_shape),
            args=(
                weight,
                bias,
                N,
                C,
                HxW,
            ),
            kwargs=kwargs,
        )


def sample_inputs_native_dropout(
    op_info, device, dtype, requires_grad, *, valid_input_dim=None, **kwargs
):
    del op_info  # Unused
    del kwargs  # Unused
    make_arg = functools.partial(
        torch_testing.make_tensor,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )

    if valid_input_dim:
        cases = ((S,) * i for i in valid_input_dim)
    else:
        cases = ((S, S), (S,), ())
    # ONNX requires 0 <= p < 1
    p_vals = [0.0]

    training_vals = [True, False]

    for case, p, training in itertools.product(cases, p_vals, training_vals):
        yield opinfo_core.SampleInput(make_arg(case), p=p, train=training)


# NOTE: In `_native_batch_norm_legit` tests, it generates two kinds of args:
# 1. (input, weight, bias, running_mean, running_var, training, momentum, eps)
# 2. (input, weight, bias, training, momentum, eps)
# which requires two function signatures to take the inputs, that's why we have
# two sample_inputs functions here instead.
def sample_inputs__native_batch_norm_legit(
    op_info, device, dtype, requires_grad, **kwargs
):
    samples = common_methods_invocations.sample_inputs_batch_norm(
        op_info, device, dtype, requires_grad, **kwargs
    )
    for sample in samples:
        # torch.native_batch_norm does not support 0 numel tensors
        # IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
        if sample.input.numel() == 0:
            continue
        args = sample.args
        training = sample.kwargs.get("training", True)
        momentum = sample.kwargs.get("momentum", 0.5)
        eps = sample.kwargs.get("eps", 1e-5)
        if args[0] is not None and args[1] is not None:
            yield opinfo_core.SampleInput(
                sample.input,
                args=(args[2], args[3], args[0], args[1]),
                kwargs={"training": training, "momentum": momentum, "eps": eps},
            )


def sample_inputs__native_batch_norm_legit_no_stats(
    op_info, device, dtype, requires_grad, **kwargs
):
    samples = common_methods_invocations.sample_inputs_batch_norm(
        op_info, device, dtype, requires_grad, **kwargs
    )
    for sample in samples:
        # torch.native_batch_norm does not support 0 numel tensors
        # IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
        if sample.input.numel() == 0:
            continue
        args = sample.args
        training = sample.kwargs.get("training", True)
        momentum = sample.kwargs.get("momentum", 0.5)
        eps = sample.kwargs.get("eps", 1e-5)
        if args[0] is not None and args[1] is None:
            yield opinfo_core.SampleInput(
                sample.input,
                args=(args[2], args[3]),
                kwargs={"training": training, "momentum": momentum, "eps": eps},
            )


def sample_inputs_non_max_suppression(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    del kwargs
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [10.0, 10.0, 20.0, 20.0],
            [32.0, 32.0, 40.0, 52.0],
        ],
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    scores = torch.tensor(
        [0.8, 0.4, 0.6], device=device, dtype=dtype, requires_grad=requires_grad
    )

    for iou_threshold in (0.3, 0.5, 0.7, 0.9):
        yield opinfo_core.SampleInput(boxes, args=(scores, iou_threshold))


def sample_inputs_normal_tensor_float(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    del requires_grad
    del kwargs
    make_arg = functools.partial(
        torch_testing.make_tensor, dtype=dtype, device=device, requires_grad=False
    )
    samples = (
        ((S, S), 0.0),
        ((S, S, S), 4.2),
    )
    for mean, std in samples:
        yield opinfo_core.SampleInput(make_arg(mean), std)


def sample_inputs_normal_float_tensor(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    del requires_grad
    del kwargs
    make_arg = functools.partial(
        torch_testing.make_tensor, dtype=dtype, device=device, requires_grad=False
    )
    samples = (
        (4.2, (S, S)),
        (-2.0, (S, S, S)),
    )
    for mean, std in samples:
        yield opinfo_core.SampleInput(mean, make_arg(std, low=0.0))


def sample_inputs_normal_tensor_tensor(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    del requires_grad
    del kwargs
    make_arg = functools.partial(
        torch_testing.make_tensor, dtype=dtype, device=device, requires_grad=False
    )
    samples = (
        ((S, S), (S, S)),
        ((S, S, S), (S, S, S)),
    )
    for mean, std in samples:
        yield opinfo_core.SampleInput(make_arg(mean), make_arg(std, low=0.0))


def sample_inputs_rand(op_info, device, dtype, requires_grad, **kwargs):
    del op_info  # Unused
    del device  # Unused
    del requires_grad  # Unused
    del kwargs  # Unused

    shapes = (
        (M,),
        (S, S),
        (S, S, S),
    )

    for shape in shapes:
        yield opinfo_core.SampleInput(shape, kwargs=dict(dtype=dtype))


def sample_inputs_rand_like(op_info, device, dtype, requires_grad, **kwargs):
    del op_info  # Unused
    del kwargs  # Unused

    make_arg = functools.partial(
        torch_testing.make_tensor,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    shapes = (
        (M,),
        (S, S),
        (S, S, S),
    )

    for shape in shapes:
        yield opinfo_core.SampleInput(make_arg(shape))


def sample_inputs_randint(self, device, dtype, requires_grad, **kwargs):
    high = 10

    for sample in sample_inputs_like_fns(self, device, dtype, requires_grad, **kwargs):
        # With high
        yield opinfo_core.SampleInput(
            high, sample.input.shape, *sample.args, **sample.kwargs
        )


def sample_inputs_randint_low(self, device, dtype, requires_grad, **kwargs):
    low = 2
    high = 10

    for sample in sample_inputs_like_fns(self, device, dtype, requires_grad, **kwargs):
        # With low and high
        yield opinfo_core.SampleInput(
            low, high, sample.input.shape, *sample.args, **sample.kwargs
        )


def sample_inputs_randint_like(self, device, dtype, requires_grad, **kwargs):
    high = 10

    for sample in sample_inputs_like_fns(self, device, dtype, requires_grad, **kwargs):
        # With high
        yield opinfo_core.SampleInput(sample.input, high, *sample.args, **sample.kwargs)


def sample_inputs_randint_like_low_dtype(self, device, dtype, requires_grad, **kwargs):
    low = 2
    high = 10

    for sample in sample_inputs_like_fns(self, device, dtype, requires_grad, **kwargs):
        # With low and high
        yield opinfo_core.SampleInput(
            sample.input, low, high, *sample.args, **sample.kwargs
        )


def sample_inputs_randn(op, device, dtype, requires_grad, **kwargs):
    del op  # Unused
    del device  # Unused
    del requires_grad  # Unused
    del kwargs  # Unused

    shapes = ((M,), (S, S))

    for shape in shapes:
        yield opinfo_core.SampleInput(input=shape, kwargs=dict(dtype=dtype))


def sample_inputs_reflection_pad1d(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    del kwargs

    cases: tuple = (  # ignore
        ((2, 3), (1, 2)),
        ((4, 5), (0, 1)),
        ((6, 7), (1, 1)),
        ((8, 9), (1, 0)),
    )

    make_inp = opinfo_core.partial(
        torch.testing.make_tensor,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )

    for shape, pad in cases:
        yield opinfo_core.SampleInput(make_inp(shape), args=(pad,))


def sample_inputs_replication_pad1d(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    del kwargs

    cases: tuple = (  # ignore
        ((2, 3), (1, 2)),
        ((4, 5), (0, 1)),
        ((6, 7), (1, 1)),
        ((8, 9), (1, 0)),
    )

    make_inp = opinfo_core.partial(
        torch.testing.make_tensor,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )

    for shape, pad in cases:
        yield opinfo_core.SampleInput(make_inp(shape), args=(pad,))


def sample_inputs_slice_scatter(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    del kwargs
    make_arg = functools.partial(
        torch_testing.make_tensor,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )

    L = 20
    cases = (
        ((L, L, L), (L, L, L), (0, 0, L, 1)),
        ((L, L, L), (L // 2, L, L), (0, L // 2, L, 1)),
        ((L, L, L), (L // 4, L, L), (0, L // 2, L, 2)),
        ((L, L, L), (L, L, L), (1, 0, L, 1)),
        ((L, L, L), (L, L // 2, L), (1, L // 2, L, 1)),
        ((L, L, L), (L, L // 4, L), (1, L // 2, L, 2)),
        ((L, L, L), (L, L, L), (2, 0, L, 1)),
        ((L, L, L), (L, L, L // 2), (2, L // 2, L, 1)),
        ((L, L, L), (L, L, L // 4), (2, L // 2, L, 2)),
        ((L, L, L), (L, L // 2, L), (1, L // 2, L * 2, 1)),  # end > L
        ((L, L, L), (L, L, L), (-2, 0, L, 1)),  # negative dim
        ((L, L, L), (L, L, L // 4), (-1, L // 2, L * 2, 2)),  # end > L and negative dim
    )

    for input_shape, src_shape, args in cases:
        input_ = make_arg(input_shape)
        src = make_arg(src_shape)
        yield opinfo_core.SampleInput(input_, args=(src, *args))


def sample_inputs__scaled_dot_product_flash_attention(
    op_info, device, dtype, requires_grad, **kwargs
):
    del op_info
    del kwargs

    make = opinfo_core.partial(
        opinfo_core.make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
    )
    batch, seq_q, seq_kv, num_heads, head_dim = 4, 3, 6, 4, 8

    dim_4_q_shape = (batch, num_heads, seq_q, head_dim)
    dim_4_kv_shape = (batch, num_heads, seq_kv, head_dim)

    qkv_shapes = [(dim_4_q_shape, dim_4_kv_shape)]
    samples = []
    for qkv_shape, is_causal, dropout_p in opinfo_core.product(
        qkv_shapes, [True, False], [0.0]
    ):
        shape_q, shape_kv = qkv_shape
        samples.append(
            opinfo_core.SampleInput(
                make(shape_q),
                make(shape_kv),
                make(shape_kv),
                is_causal=is_causal,
                dropout_p=dropout_p,
            )
        )

    # Add an attn_mask
    samples.append(
        opinfo_core.SampleInput(
            make((batch, num_heads, seq_q, head_dim)),
            make((batch, num_heads, seq_kv, head_dim)),
            make((batch, num_heads, seq_kv, head_dim)),
            is_causal=False,
            dropout_p=0.0,
        )
    )

    yield from samples


def sample_inputs__scaled_dot_product_efficient_attention(
    op_info, device, dtype, requires_grad, **kwargs
):
    del op_info
    del kwargs

    make = opinfo_core.partial(
        opinfo_core.make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
    )
    batch, seq_q, seq_kv, num_heads, head_dim = 2, 3, 6, 4, 8

    dim_4_q_shape = (batch, num_heads, seq_q, head_dim)
    dim_4_kv_shape = (batch, num_heads, seq_kv, head_dim)

    qkv_shapes = [(dim_4_q_shape, dim_4_kv_shape)]

    samples = []
    for qkv_shape, is_causal, dropout_p, compute_log_sumexp in opinfo_core.product(
        qkv_shapes, [True, False], [0.0], [True, False]
    ):
        shape_q, shape_kv = qkv_shape
        samples.append(
            opinfo_core.SampleInput(
                make(shape_q),
                make(shape_kv),
                make(shape_kv),
                attn_bias=None,  # TODO: Add attn_bias
                is_causal=is_causal,
                dropout_p=dropout_p,
                compute_log_sumexp=compute_log_sumexp,
            )
        )

    yield from samples


def sample_inputs__softmax(
    op_info,
    device,
    dtype,
    requires_grad,
    **kwargs,
):
    del op_info  # Unused

    make_arg = functools.partial(
        torch_testing.make_tensor,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    cases = [
        ((S,), (0,)),
        ((S, S), (0,)),
        ((S, S), (1,)),
        ((S, S), (-1,)),
        ((S, M, S), (2,)),
        ((S, 0, 0), (-1,)),
    ]

    for (shape, dim), half_to_float in itertools.product(cases, (False,)):
        # NOTE: softmax with half to float conversion is not supported on CPU
        # So we don't test it here
        kwargs = dict(half_to_float=half_to_float)
        yield opinfo_core.SampleInput(make_arg(shape), args=dim, kwargs=kwargs)


def sample_inputs_prims_std_var(op_info, device, dtype, requires_grad, **kwargs):
    del op_info  # Unused
    del kwargs  # Unused
    tensor_nd = functools.partial(
        opinfo_core.make_tensor,
        (S, S, S),
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    tensor_1d = functools.partial(
        opinfo_core.make_tensor,
        (S,),
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )

    yield opinfo_core.SampleInput(tensor_nd(), dims=(1,), correction=0)
    yield opinfo_core.SampleInput(tensor_1d(), dims=(0,), correction=0)
    yield opinfo_core.SampleInput(tensor_1d(), dims=(0,), correction=1)

    yield opinfo_core.SampleInput(tensor_nd(), dims=(1,), correction=1)
    yield opinfo_core.SampleInput(tensor_nd(), dims=(1,), correction=S // 2)
    yield opinfo_core.SampleInput(tensor_nd(), dims=(), correction=0)
    # Negative indices are not supported


def sample_inputs_stft(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    del kwargs

    def mt(shape, **kwargs):
        return torch_testing.make_tensor(
            shape, device=device, dtype=dtype, requires_grad=requires_grad, **kwargs
        )

    yield opinfo_core.SampleInput(mt(100), n_fft=10, return_complex=True)
    yield opinfo_core.SampleInput(mt(100), n_fft=10, return_complex=False)
    if dtype.is_complex:
        yield opinfo_core.SampleInput(mt(100), n_fft=10)

    yield opinfo_core.SampleInput(mt(10), n_fft=7, return_complex=True)
    yield opinfo_core.SampleInput(
        mt((10, 100)), n_fft=16, hop_length=4, return_complex=True
    )

    window = mt(16, low=0.5, high=2.0)
    yield opinfo_core.SampleInput(
        mt((2, 100)), kwargs=dict(n_fft=16, window=window, return_complex=True)
    )
    yield opinfo_core.SampleInput(
        mt((3, 100)), kwargs=dict(n_fft=16, window=window, return_complex=True)
    )
    if not dtype.is_complex:
        yield opinfo_core.SampleInput(
            mt((10, 100)), n_fft=16, window=window, onesided=False, return_complex=True
        )


def sample_inputs_tensor_bool(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    del device
    del requires_grad
    del kwargs
    yield opinfo_core.SampleInput(True, dtype=dtype)
    yield opinfo_core.SampleInput(False, dtype=dtype)


def sample_inputs_tensor_float(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    del device
    del requires_grad
    del kwargs
    yield opinfo_core.SampleInput(3.0, dtype=dtype)
    yield opinfo_core.SampleInput(-1.0, dtype=dtype)


def sample_inputs_tensor_int(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    del device
    del requires_grad
    del kwargs
    yield opinfo_core.SampleInput(2, dtype=dtype)
    yield opinfo_core.SampleInput(-5, dtype=dtype)


def sample_inputs_unfold(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    # Case `target_end == 1`, where `target_end = (input.size(dimension) - size) // step + 1`.
    t = torch_testing.make_tensor(
        (2, 3, 4),
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        **kwargs,
    )
    for dimension, size, step in [
        (1, 2, 2),
        (-1, 2, 2),
        (-2, 2, 2),
    ]:
        yield opinfo_core.SampleInput(t, args=(dimension, size, step))


def sample_inputs_upsample_2d(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    del kwargs

    N, C = 2, 3
    D = 4
    SS = 3
    L = 5

    align_corners_options = (True, False)
    rank = 2

    def shape(size, rank, with_batch_channel=True):
        if with_batch_channel:
            return tuple([N, C] + ([size] * rank))
        return tuple([size] * rank)

    make_arg = functools.partial(
        torch_testing.make_tensor,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        low=-1,
        high=1,
    )

    yield opinfo_core.SampleInput(
        make_arg(shape(D, rank)), shape(SS, rank, False), True
    )

    for align_corners in align_corners_options:
        yield opinfo_core.SampleInput(
            make_arg(shape(D, rank)), shape(S, rank, False), align_corners
        )
        yield opinfo_core.SampleInput(
            make_arg(shape(D, rank)), shape(L, rank, False), align_corners
        )
        yield opinfo_core.SampleInput(
            make_arg(shape(D, rank)),
            args=(shape(L, rank, False), align_corners),
            kwargs=dict(scales_h=0.6, scales_w=4.2),
        )
        yield opinfo_core.SampleInput(
            make_arg(shape(D, rank)),
            args=(shape(L, rank, False), align_corners),
            kwargs=dict(scales_h=4.2, scales_w=0.6),
        )


def sample_inputs_upsample_2d_vec(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    del kwargs

    N, C = 2, 3
    D = 4
    SS = 3
    L = 5

    align_corners_options = (True, False)
    rank = 2

    def shape(size, rank, with_batch_channel=True):
        if with_batch_channel:
            return tuple([N, C] + ([size] * rank))
        return tuple([size] * rank)

    make_arg = functools.partial(
        torch_testing.make_tensor,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        low=-1,
        high=1,
    )

    yield opinfo_core.SampleInput(
        make_arg(shape(D, rank)), shape(SS, rank, False), True, None
    )

    for align_corners in align_corners_options:
        yield opinfo_core.SampleInput(
            make_arg(shape(D, rank)), shape(S, rank, False), align_corners, None
        )
        yield opinfo_core.SampleInput(
            make_arg(shape(D, rank)), shape(L, rank, False), align_corners, None
        )
        yield opinfo_core.SampleInput(
            make_arg(shape(D, rank)),
            args=(
                None,  # output_size
                align_corners,
            ),
            kwargs=dict(scale_factors=[1.7, 1.7]),
        )
        yield opinfo_core.SampleInput(
            make_arg(shape(D, rank)),
            args=(
                None,  # if this is None, the scalar must be list
                align_corners,
            ),
            kwargs=dict(scale_factors=[0.6, 0.6]),
        )
        yield opinfo_core.SampleInput(
            make_arg(shape(D, rank)),
            args=(
                None,  # if this is None, the scalar must be list
                align_corners,
            ),
            kwargs=dict(scale_factors=[0.6, 4.2]),
        )


def sample_inputs_upsample_linear1d(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    del kwargs

    N, C = 2, 3
    D = 4
    SS = 3
    L = 5

    align_corners_options = (True, False)
    rank = 1

    def shape(size, rank, with_batch_channel=True):
        if with_batch_channel:
            return tuple([N, C] + ([size] * rank))
        return tuple([size] * rank)

    make_arg = functools.partial(
        torch_testing.make_tensor,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        low=-1,
        high=1,
    )

    yield opinfo_core.SampleInput(
        make_arg(shape(D, rank)), shape(SS, rank, False), True
    )

    for align_corners in align_corners_options:
        yield opinfo_core.SampleInput(
            make_arg(shape(D, rank)), shape(S, rank, False), align_corners
        )
        yield opinfo_core.SampleInput(
            make_arg(shape(D, rank)), shape(L, rank, False), align_corners
        )
        yield opinfo_core.SampleInput(
            make_arg(shape(D, rank)), shape(L, rank, False), align_corners, scales=4.2
        )


def sample_inputs_upsample_nearest1d(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    del kwargs

    N, C = 2, 3
    D = 4
    L = 5

    rank = 1

    def shape(size, rank, with_batch_channel=True):
        if with_batch_channel:
            return tuple([N, C] + ([size] * rank))
        return tuple([size] * rank)

    make_arg = functools.partial(
        torch_testing.make_tensor,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        low=-1,
        high=1,
    )

    yield opinfo_core.SampleInput(
        make_arg(shape(D, rank)),
        shape(S, rank, False),
    )
    yield opinfo_core.SampleInput(
        make_arg(shape(D, rank)),
        shape(L, rank, False),
    )
    # yield opinfo_core.SampleInput(
    #     make_arg(shape(D, rank)),
    #     shape(S, rank, False),  # output_size
    #     [1.7],  # scaler
    # )
    # yield opinfo_core.SampleInput(
    #     make_arg(shape(D, rank)),
    #     shape(S, rank, False),  # if this is None, the scalar must be list
    #     [0.6],
    # )


def sample_inputs_upsample_nearest1d_vec(
    op_info, device, dtype, requires_grad, **kwargs
):
    del op_info
    del kwargs

    N, C = 2, 3
    D = 4
    L = 5

    rank = 1

    def shape(size, rank, with_batch_channel=True):
        if with_batch_channel:
            return tuple([N, C] + ([size] * rank))
        return tuple([size] * rank)

    make_arg = functools.partial(
        torch_testing.make_tensor,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        low=-1,
        high=1,
    )

    yield opinfo_core.SampleInput(make_arg(shape(D, rank)), shape(S, rank, False), None)
    yield opinfo_core.SampleInput(make_arg(shape(D, rank)), shape(L, rank, False), None)
    yield opinfo_core.SampleInput(
        make_arg(shape(D, rank)),
        None,  # output_size
        scale_factors=(1.7,),
    )
    yield opinfo_core.SampleInput(
        make_arg(shape(D, rank)),
        None,
        scale_factors=(0.6,),
    )


def sample_inputs_upsample_nearest2d(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    del kwargs

    N, C = 2, 3
    D = 4
    L = 5

    rank = 2

    def shape(size, rank, with_batch_channel=True):
        if with_batch_channel:
            return tuple([N, C] + ([size] * rank))
        return tuple([size] * rank)

    make_arg = functools.partial(
        torch_testing.make_tensor,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        low=-1,
        high=1,
    )

    yield opinfo_core.SampleInput(
        make_arg(shape(D, rank)),
        shape(S, rank, False),
    )
    yield opinfo_core.SampleInput(
        make_arg(shape(D, rank)),
        shape(L, rank, False),
    )
    # yield opinfo_core.SampleInput(
    #     make_arg(shape(D, rank)),
    #     shape(L, rank, False),
    #     1.7, 2.0,  # scaler
    # )
    # yield opinfo_core.SampleInput(
    #     make_arg(shape(D, rank)),
    #     shape(L, rank, False),
    #     0.6, 0.4,
    # )


def sample_inputs_upsample_nearest2d_vec(
    op_info, device, dtype, requires_grad, **kwargs
):
    del op_info
    del kwargs

    N, C = 2, 3
    D = 4
    L = 5

    rank = 2

    def shape(size, rank, with_batch_channel=True):
        if with_batch_channel:
            return tuple([N, C] + ([size] * rank))
        return tuple([size] * rank)

    make_arg = functools.partial(
        torch_testing.make_tensor,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        low=-1,
        high=1,
    )

    yield opinfo_core.SampleInput(make_arg(shape(D, rank)), shape(S, rank, False), None)
    yield opinfo_core.SampleInput(make_arg(shape(D, rank)), shape(L, rank, False), None)
    yield opinfo_core.SampleInput(
        make_arg(shape(D, rank)),
        None,
        scale_factors=(1.7, 2.0),
    )
    yield opinfo_core.SampleInput(
        make_arg(shape(D, rank)),
        None,
        scale_factors=(0.6, 0.4),
    )


def sample_inputs_upsample_nearest3d(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    del kwargs

    N, C = 2, 3
    D = 4
    L = 5

    rank = 3

    def shape(size, rank, with_batch_channel=True):
        if with_batch_channel:
            return tuple([N, C] + ([size] * rank))
        return tuple([size] * rank)

    make_arg = functools.partial(
        torch_testing.make_tensor,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        low=-1,
        high=1,
    )

    yield opinfo_core.SampleInput(
        make_arg(shape(D, rank)),
        shape(S, rank, False),
    )
    yield opinfo_core.SampleInput(
        make_arg(shape(D, rank)),
        shape(L, rank, False),
    )
    # yield opinfo_core.SampleInput(
    #     make_arg(shape(D, rank)),
    #     shape(L, rank, False),
    #     1.7, 1.5, 2.0,  # scaler
    # )
    # yield opinfo_core.SampleInput(
    #     make_arg(shape(D, rank)),
    #     shape(L, rank, False),
    #     0.6, 0.3, 0.5,
    # )


def sample_inputs_upsample_nearest3d_vec(
    op_info, device, dtype, requires_grad, **kwargs
):
    del op_info
    del kwargs

    N, C = 2, 3
    D = 4
    L = 5

    rank = 3

    def shape(size, rank, with_batch_channel=True):
        if with_batch_channel:
            return tuple([N, C] + ([size] * rank))
        return tuple([size] * rank)

    make_arg = functools.partial(
        torch_testing.make_tensor,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        low=-1,
        high=1,
    )

    yield opinfo_core.SampleInput(make_arg(shape(D, rank)), shape(S, rank, False), None)
    yield opinfo_core.SampleInput(make_arg(shape(D, rank)), shape(L, rank, False), None)
    yield opinfo_core.SampleInput(
        make_arg(shape(D, rank)),
        None,
        scale_factors=(1.7, 1.5, 2.0),  # scaler
    )
    yield opinfo_core.SampleInput(
        make_arg(shape(D, rank)),
        None,
        scale_factors=(0.6, 0.3, 0.5),
    )


def sample_inputs_upsample_trilinear3d(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    del kwargs

    N, C = 2, 3
    D = 4
    SS = 3
    L = 5

    align_corners_options = (True, False)
    rank = 3

    def shape(size, rank, with_batch_channel=True):
        if with_batch_channel:
            return tuple([N, C] + ([size] * rank))
        return tuple([size] * rank)

    make_arg = functools.partial(
        torch_testing.make_tensor,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        low=-1,
        high=1,
    )

    for align_corners in align_corners_options:
        yield opinfo_core.SampleInput(
            make_arg(shape(D, rank)), shape(SS, rank, False), align_corners
        )
        yield opinfo_core.SampleInput(
            make_arg(shape(D, rank)), shape(S, rank, False), align_corners
        )
        yield opinfo_core.SampleInput(
            make_arg(shape(D, rank)), shape(L, rank, False), align_corners
        )


def sample_inputs_upsample_trilinear3d_vec(
    op_info, device, dtype, requires_grad, **kwargs
):
    del op_info
    del kwargs

    N, C = 2, 3
    D = 4
    SS = 3
    L = 5

    align_corners_options = (True, False)
    rank = 3

    def shape(size, rank, with_batch_channel=True):
        if with_batch_channel:
            return tuple([N, C] + ([size] * rank))
        return tuple([size] * rank)

    make_arg = functools.partial(
        torch_testing.make_tensor,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        low=-1,
        high=1,
    )

    yield opinfo_core.SampleInput(
        make_arg(shape(D, rank)), shape(SS, rank, False), True, None
    )

    for align_corners in align_corners_options:
        yield opinfo_core.SampleInput(
            make_arg(shape(D, rank)), shape(S, rank, False), align_corners, None
        )
        yield opinfo_core.SampleInput(
            make_arg(shape(D, rank)), shape(L, rank, False), align_corners, None
        )
        yield opinfo_core.SampleInput(
            make_arg(shape(D, rank)),
            args=(None, align_corners),
            kwargs=dict(scale_factors=(1.7, 1.7, 1.7)),
        )
        yield opinfo_core.SampleInput(
            make_arg(shape(D, rank)),
            args=(None, align_corners),
            kwargs=dict(scale_factors=(0.6, 0.6, 0.6)),
        )
        yield opinfo_core.SampleInput(
            make_arg(shape(D, rank)),
            args=(None, align_corners),
            kwargs=dict(scale_factors=(0.6, 1.7, 4.2)),
        )


def sample_inputs_window_functions(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    del kwargs
    del device
    del requires_grad

    for window_length in [2, 3, 7, 10, 32]:
        yield opinfo_core.SampleInput(window_length, kwargs=dict(dtype=dtype))


class _TestParamsMaxPoolEmptyStrideBase:
    # Adapted from https://github.com/pytorch/pytorch/blob/d6d55f8590eab05d2536756fb4efcfb2d07eb81a/torch/testing/_internal/common_methods_invocations.py#L3203
    def __init__(self):
        self.kwargs = {
            "kernel_size": [3],
            "stride": [()],
            "ceil_mode": [True, False],
            "padding": [0, 1],
            "dilation": [1],
        }

        # fmt: off
        self.shapes = [
            [1, 2, None],  # batch
            [2],  # channels
            [3, 6]  # signal
        ]
        # fmt: on

    def _gen_shape(self):
        for shape in itertools.product(*self.shapes):
            # shape[0] is None indicates missing batch dimension
            if shape[0] is None:
                shape = shape[1:]

            yield shape, torch.contiguous_format
            # only 2d (N, C, H, W) rank 4 tensors support channels_last memory format
            if len(self.shapes) == 4 and len(shape) == 4:
                yield shape, torch.channels_last

    def _gen_kwargs(self):
        keys = self.kwargs.keys()
        for values in itertools.product(*self.kwargs.values()):
            yield dict(zip(keys, values))

    def gen_input_params(self):
        yield from itertools.product(self._gen_shape(), self._gen_kwargs())


class _TestParamsMaxPool1dEmptyStride(_TestParamsMaxPoolEmptyStrideBase):
    def __init__(self):
        super().__init__()
        self.kwargs["kernel_size"] += [(3,)]
        self.kwargs["stride"] += [(2,)]
        self.kwargs["padding"] += [(1,)]
        self.kwargs["dilation"] += [(1,)]


class _TestParamsMaxPool2dEmptyStride(_TestParamsMaxPoolEmptyStrideBase):
    def __init__(self):
        super().__init__()
        self.kwargs["kernel_size"] += [(3, 2)]
        self.kwargs["stride"] += [(2, 1)]
        self.kwargs["padding"] += [(1, 1)]
        self.kwargs["dilation"] += [(1, 2)]

        self.shapes.append([6])


class _TestParamsMaxPool3dEmptyStride(_TestParamsMaxPoolEmptyStrideBase):
    def __init__(self):
        super().__init__()
        self.kwargs["kernel_size"] += [(3, 2, 3)]
        self.kwargs["stride"] += [(2, 1, 2)]
        self.kwargs["dilation"] += [(1, 2, 1)]

        self.shapes.append([6])
        self.shapes.append([5])


# NOTE: How to create an OpInfo:
# 1. Create a function that generates sample inputs for the op.
#    This function should yield SampleInputs.
#    Use `sample_inputs_col2im` as an example.
# 2. Specify dtypes that the op supports.
# 3. Use how you would call the op in PyTorch as the name of the OpInfo.
#    For example, `torch.ops.aten.col2im` should be named "ops.aten.col2im".
#    This way OpInfo knows to use `torch.ops.aten.col2im` as the op.
#    See the docstring of OpInfo for more details.
#
#    This name is used as the unique ID to connect `TorchLibOpInfo("unique_name", ...)``
#    in ops_test_data.py and opinfo_core.OpInfo("unique_name", ...)
#    To avoid name duplication, it is possible to rename the OpInfo and specify
#    the `op` field explicitly.
OP_DB: List[opinfo_core.OpInfo] = [
    opinfo_core.OpInfo(
        "ops.aten.bernoulli.p",
        aten_name="bernoulli.p",
        # dtypes can be a tuple of (torch.float, torch.double).
        dtypes=common_dtype.all_types(),
        sample_inputs_func=sample_inputs_bernoulli_p,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        # Deterministic bernoulli sampling where p is either 0 or 1
        "ops.aten.bernoulli.p_deterministic",
        op=torch.ops.aten.bernoulli.p,
        aten_name="bernoulli.p",
        dtypes=common_dtype.all_types(),
        sample_inputs_func=sample_inputs_bernoulli_p_deterministic,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.blackman_window",
        aten_name="blackman_window",
        dtypes=common_dtype.floating_types_and(torch.bfloat16),
        sample_inputs_func=sample_inputs_window_functions,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.col2im",
        aten_name="col2im",
        dtypes=common_dtype.floating_and_complex_types_and(torch.half, torch.bfloat16),
        sample_inputs_func=sample_inputs_col2im,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.conv3d",
        aten_name="conv3d",
        dtypes=common_dtype.floating_and_complex_types_and(torch.int64, torch.bfloat16),
        sample_inputs_func=sample_inputs_conv3d,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.convolution",
        aten_name="convolution",
        dtypes=common_dtype.floating_and_complex_types_and(torch.int64, torch.bfloat16),
        sample_inputs_func=sample_inputs_convolution,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.embedding_bag",
        aten_name="embedding_bag",
        dtypes=common_dtype.floating_types_and_half(),
        sample_inputs_func=sample_inputs_embedding_bag,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.embedding_bag.padding_idx",
        aten_name="embedding_bag.padding_idx",
        dtypes=common_dtype.floating_types_and_half(),
        sample_inputs_func=sample_inputs_embedding_bag_padding_idx,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.embedding_renorm",
        aten_name="embedding_renorm",
        dtypes=common_dtype.floating_types_and_half(),
        sample_inputs_func=sample_inputs_embedding_renorm,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten._fft_c2c",
        aten_name="_fft_c2c",
        dtypes=common_dtype.complex_types(),
        sample_inputs_func=sample_inputs__fft_c2c,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten._fft_c2r",
        aten_name="_fft_c2r",
        dtypes=common_dtype.complex_types(),
        sample_inputs_func=sample_inputs__fft_c2r,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten._fft_r2c",
        aten_name="_fft_r2c",
        dtypes=common_dtype.floating_types(),
        sample_inputs_func=sample_inputs__fft_r2c,
        supports_out=False,
    ),
    opinfo_core.BinaryUfuncInfo(
        "ops.aten.floor_divide",
        aten_name="floor_divide",
        dtypes=common_dtype.floating_types_and_half(),
        rhs_make_tensor_kwargs=dict(exclude_zero=True),
    ),
    opinfo_core.BinaryUfuncInfo(
        "ops.aten.floor_divide.int",
        aten_name="floor_divide",
        op=torch.ops.aten.floor_divide,
        dtypes=common_dtype.integral_types(),
        # Create only positive inputs
        lhs_make_tensor_kwargs=dict(low=0),
        rhs_make_tensor_kwargs=dict(exclude_zero=True, low=0),
    ),
    opinfo_core.OpInfo(
        "ops.aten.hamming_window",
        aten_name="hamming_window",
        dtypes=common_dtype.floating_types_and(torch.bfloat16),
        sample_inputs_func=sample_inputs_window_functions,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.hann_window",
        aten_name="hann_window",
        dtypes=common_dtype.floating_types_and(torch.bfloat16),
        sample_inputs_func=sample_inputs_window_functions,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.index.Tensor",
        aten_name="index.Tensor",
        dtypes=common_dtype.all_types_and_complex_and(
            torch.bool, torch.float16, torch.bfloat16, torch.chalf
        ),
        sample_inputs_func=sample_inputs_index,
    ),
    opinfo_core.OpInfo(
        "ops.aten.index.Tensor.bool",
        aten_name="index.Tensor",
        dtypes=common_dtype.all_types_and_complex_and(
            torch.bool, torch.float16, torch.bfloat16, torch.chalf
        ),
        sample_inputs_func=sample_inputs_index_bool,
        op=torch.ops.aten.index.Tensor,
    ),
    opinfo_core.OpInfo(
        "ops.aten.index_put",
        aten_name="index_put",
        dtypes=common_dtype.floating_types(),
        sample_inputs_func=sample_inputs_index_put,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten._unsafe_index_put",
        aten_name="_unsafe_index_put",
        dtypes=common_dtype.floating_types(),
        sample_inputs_func=sample_inputs_index_put,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.layer_norm",
        aten_name="layer_norm",
        dtypes=common_dtype.floating_and_complex_types_and(torch.int64, torch.bfloat16),
        sample_inputs_func=sample_inputs_layer_norm,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten._local_scalar_dense",
        aten_name="_local_scalar_dense",
        dtypes=common_dtype.all_types(),
        sample_inputs_func=sample_inputs__local_scalar_dense,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten._log_softmax",
        op=torch.ops.aten._log_softmax,  # pylint: disable=protected-access
        aten_name="_log_softmax",
        dtypes=common_dtype.floating_types_and_half(),
        sample_inputs_func=sample_inputs__log_softmax,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.max_pool1d",
        variant_test_name="empty_strides",
        aten_name="max_pool1d",
        dtypes=common_dtype.floating_types_and(torch.bfloat16),
        sample_inputs_func=sample_inputs_max_pool_empty_strides,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.max_pool2d",
        variant_test_name="empty_strides",
        aten_name="max_pool2d",
        dtypes=common_dtype.floating_types_and(torch.bfloat16),
        sample_inputs_func=sample_inputs_max_pool_empty_strides,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.max_pool3d",
        variant_test_name="empty_strides",
        aten_name="max_pool3d",
        dtypes=common_dtype.floating_types_and(torch.bfloat16),
        sample_inputs_func=sample_inputs_max_pool_empty_strides,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.native_dropout",
        aten_name="native_dropout",
        dtypes=common_dtype.all_types_and_half(),
        sample_inputs_func=sample_inputs_native_dropout,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.native_group_norm",
        aten_name="native_group_norm",
        dtypes=common_dtype.floating_and_complex_types_and(torch.half, torch.bfloat16),
        sample_inputs_func=sample_inputs_native_group_norm,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten._native_batch_norm_legit",
        aten_name="_native_batch_norm_legit",
        dtypes=common_dtype.floating_types_and(torch.bfloat16),
        dtypesIfCUDA=common_dtype.floating_types_and(torch.float16, torch.bfloat16),
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        assert_jit_shape_analysis=True,
        sample_inputs_func=sample_inputs__native_batch_norm_legit,
    ),
    opinfo_core.OpInfo(
        "ops.aten._native_batch_norm_legit_functional",
        aten_name="_native_batch_norm_legit_functional",
        dtypes=common_dtype.floating_types_and(torch.bfloat16),
        dtypesIfCUDA=common_dtype.floating_types_and(torch.float16, torch.bfloat16),
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        assert_jit_shape_analysis=True,
        sample_inputs_func=sample_inputs__native_batch_norm_legit,
    ),
    opinfo_core.OpInfo(
        "ops.aten._native_batch_norm_legit.no_stats",
        aten_name="_native_batch_norm_legit.no_stats",
        dtypes=common_dtype.floating_types_and(torch.bfloat16),
        dtypesIfCUDA=common_dtype.floating_types_and(torch.float16, torch.bfloat16),
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        assert_jit_shape_analysis=True,
        sample_inputs_func=sample_inputs__native_batch_norm_legit_no_stats,
    ),
    opinfo_core.OpInfo(
        "ops.aten.normal.float_Tensor",
        aten_name="normal.Tensor_Tensor",
        dtypes=common_dtype.floating_types_and_half(),
        sample_inputs_func=sample_inputs_normal_float_tensor,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.normal.Tensor_float",
        aten_name="normal.Tensor_Tensor",
        dtypes=common_dtype.floating_types_and_half(),
        sample_inputs_func=sample_inputs_normal_tensor_float,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.normal.Tensor_Tensor",
        aten_name="normal.Tensor_Tensor",
        dtypes=common_dtype.floating_types_and_half(),
        sample_inputs_func=sample_inputs_normal_tensor_tensor,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.rand",
        aten_name="rand",
        dtypes=common_dtype.floating_types_and(torch.bfloat16),
        sample_inputs_func=sample_inputs_rand,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.rand_like",
        aten_name="rand_like",
        dtypes=common_dtype.floating_types_and(torch.bfloat16),
        sample_inputs_func=sample_inputs_rand_like,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.randint",
        aten_name="randint",
        dtypes=common_dtype.integral_types(),
        sample_inputs_func=sample_inputs_randint,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.randint.low",
        aten_name="randint.low",
        dtypes=common_dtype.integral_types(),
        sample_inputs_func=sample_inputs_randint_low,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.randint_like",
        aten_name="randint_like",
        dtypes=common_dtype.integral_types(),
        sample_inputs_func=sample_inputs_randint_like,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.randint_like.low_dtype",
        aten_name="randint_like.low_dtype",
        dtypes=common_dtype.integral_types(),
        sample_inputs_func=sample_inputs_randint_like_low_dtype,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.randn",
        aten_name="randn",
        dtypes=common_dtype.floating_types_and(torch.bfloat16),
        sample_inputs_func=sample_inputs_randn,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.randn_like",
        aten_name="randn",
        dtypes=common_dtype.floating_types_and(torch.bfloat16),
        sample_inputs_func=sample_inputs_like_fns,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.reflection_pad1d",
        aten_name="ops.aten.reflection_pad1d",
        dtypes=common_dtype.floating_and_complex_types_and(torch.int64, torch.bfloat16),
        sample_inputs_func=sample_inputs_reflection_pad1d,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.replication_pad1d",
        aten_name="ops.aten.replication_pad1d",
        dtypes=common_dtype.floating_and_complex_types_and(torch.int64, torch.bfloat16),
        sample_inputs_func=sample_inputs_replication_pad1d,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten._scaled_dot_product_flash_attention",
        aten_name="_scaled_dot_product_flash_attention",
        dtypes=common_dtype.floating_types_and(torch.bfloat16),
        # NOTE: Different from aten::scaled_dot_product_attention, this op doesn't support
        #       dim<=3 input.
        sample_inputs_func=sample_inputs__scaled_dot_product_flash_attention,
        supports_out=False,
        supports_forward_ad=False,
        supports_fwgrad_bwgrad=True,
        check_batched_forward_grad=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten._scaled_dot_product_efficient_attention",
        aten_name="_scaled_dot_product_efficient_attention",
        # only support CUDA
        dtypes=common_dtype.empty_types(),
        dtypesIfCUDA=common_dtype.floating_types_and(torch.bfloat16),
        # NOTE: Different from aten::scaled_dot_product_attention, this op doesn't support
        #       dim<=3 input.
        sample_inputs_func=sample_inputs__scaled_dot_product_efficient_attention,
        supports_out=False,
        supports_forward_ad=False,
        supports_fwgrad_bwgrad=True,
        check_batched_forward_grad=False,
        decorators=[common_device_type.onlyCUDA],
    ),
    opinfo_core.OpInfo(
        "ops.aten.slice_scatter",
        aten_name="slice_scatter",
        dtypes=common_dtype.all_types_and(torch.bfloat16, torch.half, torch.bool),
        sample_inputs_func=sample_inputs_slice_scatter,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten._softmax",
        op=torch.ops.aten._softmax,  # pylint: disable=protected-access
        aten_name="_softmax",
        dtypes=common_dtype.floating_types_and_half(),
        sample_inputs_func=sample_inputs__softmax,
        supports_out=False,
    ),
    # NOTE: torch.STFT has pre-padding and it's not supported by aten::stft
    # This custom OpInfo uses aten::stft directly.
    opinfo_core.OpInfo(
        "ops.aten.stft",
        aten_name="stft",
        dtypes=common_dtype.floating_and_complex_types_and(torch.half, torch.bfloat16),
        sample_inputs_func=sample_inputs_stft,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.tensor.bool",
        aten_name="tensor.bool",
        dtypes=common_dtype.all_types_and(torch.half, torch.bfloat16),
        sample_inputs_func=sample_inputs_tensor_bool,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.tensor.float",
        aten_name="tensor.float",
        dtypes=common_dtype.all_types_and(torch.half, torch.bfloat16),
        sample_inputs_func=sample_inputs_tensor_float,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.tensor.int",
        aten_name="tensor.int",
        dtypes=common_dtype.all_types_and(torch.half, torch.bfloat16),
        sample_inputs_func=sample_inputs_tensor_int,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.unfold",
        aten_name="unfold",
        dtypes=common_dtype.all_types(),
        sample_inputs_func=sample_inputs_unfold,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.upsample_bicubic2d.default",
        aten_name="upsample_bicubic2d",
        dtypes=common_dtype.floating_types_and(torch.bfloat16),
        sample_inputs_func=sample_inputs_upsample_2d,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.upsample_bicubic2d.vec",
        aten_name="upsample_bicubic2d.vec",
        dtypes=common_dtype.floating_types_and(torch.bfloat16),
        sample_inputs_func=sample_inputs_upsample_2d_vec,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.upsample_bilinear2d.default",
        aten_name="upsample_bilinear2d",
        dtypes=common_dtype.floating_types_and(torch.bfloat16),
        sample_inputs_func=sample_inputs_upsample_2d,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.upsample_bilinear2d.vec",
        aten_name="upsample_bilinear2d.vec",
        dtypes=common_dtype.floating_types_and(torch.bfloat16),
        sample_inputs_func=sample_inputs_upsample_2d_vec,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.upsample_linear1d",
        aten_name="upsample_linear1d",
        dtypes=common_dtype.floating_types_and(torch.bfloat16),
        sample_inputs_func=sample_inputs_upsample_linear1d,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.upsample_nearest1d",
        aten_name="upsample_nearest1d",
        dtypes=common_dtype.floating_types_and(torch.bfloat16),
        sample_inputs_func=sample_inputs_upsample_nearest1d,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.upsample_nearest1d.vec",
        aten_name="upsample_nearest1d.vec",
        dtypes=common_dtype.floating_types_and(torch.bfloat16),
        sample_inputs_func=sample_inputs_upsample_nearest1d_vec,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.upsample_nearest2d",
        aten_name="upsample_nearest2d",
        dtypes=common_dtype.floating_types_and(torch.bfloat16),
        sample_inputs_func=sample_inputs_upsample_nearest2d,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.upsample_nearest2d.vec",
        aten_name="upsample_nearest2d.vec",
        dtypes=common_dtype.floating_types_and(torch.bfloat16),
        sample_inputs_func=sample_inputs_upsample_nearest2d_vec,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.upsample_nearest3d",
        aten_name="upsample_nearest3d",
        dtypes=common_dtype.floating_types_and(torch.bfloat16),
        sample_inputs_func=sample_inputs_upsample_nearest3d,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.upsample_nearest3d.vec",
        aten_name="upsample_nearest3d.vec",
        dtypes=common_dtype.floating_types_and(torch.bfloat16),
        sample_inputs_func=sample_inputs_upsample_nearest3d_vec,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.upsample_trilinear3d.default",
        aten_name="upsample_trilinear3d",
        dtypes=common_dtype.floating_types_and(torch.bfloat16),
        sample_inputs_func=sample_inputs_upsample_trilinear3d,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.upsample_trilinear3d.vec",
        aten_name="upsample_trilinear3d.vec",
        dtypes=common_dtype.floating_types_and(torch.bfloat16),
        sample_inputs_func=sample_inputs_upsample_trilinear3d_vec,
        supports_out=False,
    ),
    opinfo_core.ReductionOpInfo(
        "ops.prims.var.default",
        nan_policy="propagate",
        supports_out=True,
        promotes_int_to_float=True,
        complex_to_real=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        check_batched_forward_grad=False,
        dtypes=common_dtype.floating_and_complex_types_and(torch.half, torch.bfloat16),
        sample_inputs_func=sample_inputs_prims_std_var,
    ),
    opinfo_core.OpInfo(
        "nn.functional.max_pool1d_with_indices",
        aten_name="max_pool1d_with_indices",
        dtypes=common_dtype.floating_types_and(torch.bfloat16),
        sample_inputs_func=sample_inputs_max_pool1d_with_indices,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "nn.functional.max_pool2d_with_indices",
        aten_name="max_pool2d_with_indices",
        dtypes=common_dtype.floating_types_and(torch.bfloat16),
        sample_inputs_func=sample_inputs_max_pool2d_with_indices,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "nn.functional.max_pool3d_with_indices",
        aten_name="max_pool3d_with_indices",
        dtypes=common_dtype.floating_types_and(torch.bfloat16),
        sample_inputs_func=sample_inputs_max_pool3d_with_indices,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.scalar_tensor",
        aten_name="scalar_tensor",
        dtypes=common_dtype.complex_types(),
        sample_inputs_func=sample_inputs_scalar_tensor,
        supports_autograd=False,
        supports_out=False,
    ),
]
