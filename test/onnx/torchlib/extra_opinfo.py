# Owner(s): ["module: onnx"]
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Additional source torchlib samples for native structural translations."""

from __future__ import annotations

import functools
import itertools

import torch
from torch.testing import make_tensor
from torch.testing._internal import common_dtype, common_methods_invocations
from torch.testing._internal.opinfo import core as opinfo_core


S = 5
M = 10


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
        make_tensor, dtype=dtype, device=device, requires_grad=requires_grad
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
        make_tensor, dtype=dtype, device=device, requires_grad=requires_grad
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

    make_arg = functools.partial(
        make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
    )

    cases = [
        # Cases: one None
        ((1, 3, 4), [None, torch.arange(2, device=device), None], (1, 2, 4)),
        ((10, 3, 4), [torch.arange(5, device=device), None, None], (5, 3, 4)),
        (
            (10, 3, 4, 6),
            [None, None, None, torch.arange(3, device=device)],
            (10, 3, 4, 3),
        ),
        # Cases: two None
        (
            (10, 3, 4),
            [None, torch.arange(3, device=device), torch.arange(3, device=device)],
            (10, 3),
        ),
        (
            (10, 3, 4, 6),
            [
                torch.arange(2, device=device),
                None,
                torch.arange(2, device=device),
                torch.arange(2, device=device),
            ],
            (2, 3),
        ),
        (
            (10, 3, 4),
            [torch.arange(2, device=device), torch.arange(2, device=device), None],
            (2, 4),
        ),
        # Cases: Single indexing
        ((10, 3, 4), [None, None, torch.tensor([0], device=device)], (10, 3, 1)),
        ((10, 3, 4), [torch.tensor([0], device=device), None, None], (1, 3, 4)),
        (
            (10, 3, 4, 6),
            [None, torch.tensor([0], device=device), None, None],
            (10, 1, 4, 6),
        ),
        # Cases: Single element
        (
            (10, 3, 4),
            [
                torch.tensor([0], device=device),
                torch.tensor([0], device=device),
                torch.tensor([0], device=device),
            ],
            (1,),
        ),
        # Cases: Multidimensional index
        (
            (10, 3),
            [torch.arange(8, dtype=torch.int64, device=device).reshape((-1, 4))],
            (2, 4, 3),
        ),
    ]

    for data_shape, indices, values_shape in cases:  # type: ignore[misc]
        data = make_arg(data_shape)
        values = make_arg(values_shape)  # type: ignore[has-type]

        yield opinfo_core.SampleInput(data, indices, values)


def sample_inputs_mean_dtype(op_info, device, dtype, requires_grad, **kwargs):
    del op_info  # Unused
    del kwargs  # Unused

    make_arg = functools.partial(
        make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
    )
    for shape in ((S, S), (S,), ()):
        yield opinfo_core.SampleInput(make_arg(shape), kwargs={"dtype": torch.float64})

    # Precision sensitive values: accumulating in float32 gives 0.0 while accumulating
    # in float64 gives 1/3, so an implementation that casts only the reduced result
    # cannot pass this sample.
    yield opinfo_core.SampleInput(
        torch.tensor(
            [[1e8, 1.0, -1e8]], dtype=dtype, device=device, requires_grad=requires_grad
        ),
        kwargs={"dtype": torch.float64},
    )


def sample_inputs_slice_scatter(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    del kwargs
    make_arg = functools.partial(
        make_tensor, dtype=dtype, device=device, requires_grad=requires_grad
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
        # start/end = None (regression for #2372)
        ((L, L, L), (L, L, L), (0, None, None, 1)),  # full replace, dim 0
        ((L, L, L), (L // 2, L, L), (0, None, L // 2, 1)),  # None start
        ((L, L, L), (L // 2, L, L), (0, L // 2, None, 1)),  # None end
        ((L, L, L), (L, L, L), (1, None, None, 1)),  # None start & end, dim 1
        ((L, L, L), (L // 2, L, L), (0, None, None, 2)),  # None start & end, step 2
        ((L, L, L), (L, L, L // 2), (-1, None, L // 2, 1)),  # None start, negative dim
    )

    for input_shape, src_shape, args in cases:
        input_ = make_arg(input_shape)
        src = make_arg(src_shape)
        yield opinfo_core.SampleInput(input_, args=(src, *args))


def sample_inputs_scatter_src(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    del kwargs
    make_arg = functools.partial(
        make_tensor, dtype=dtype, device=device, requires_grad=requires_grad
    )

    # Basic test cases for scatter.src
    cases = [
        # (self_shape, index_shape, src_shape, dim)
        ((5, 5), (2, 3), (2, 3), 0),  # 2D scatter on dim=0
        ((5, 5), (3, 2), (3, 2), 1),  # 2D scatter on dim=1
        ((3, 4, 5), (2, 2, 3), (2, 2, 3), 0),  # 3D scatter on dim=0
        ((3, 4, 5), (2, 2, 3), (2, 2, 3), 1),  # 3D scatter on dim=1
        ((3, 4, 5), (2, 2, 3), (2, 2, 3), 2),  # 3D scatter on dim=2
        ((10,), (3,), (3,), 0),  # 1D scatter
    ]

    for self_shape, index_shape, src_shape, dim in cases:
        self_tensor = make_arg(self_shape)
        # Create valid indices for the given dimension without duplication
        index_buffer_shape = list(index_shape)
        index_buffer_shape[dim] = self_shape[dim]
        index_tensor = torch.rand(index_buffer_shape, device=device).argsort(dim=dim)[
            tuple(slice(None, d, None) for d in index_shape)
        ]
        src_tensor = make_arg(src_shape)
        yield opinfo_core.SampleInput(self_tensor, args=(dim, index_tensor, src_tensor))

    # Additional test cases for scalar and single-element tensor combinations with dim=0
    # Test case: scalar index, scalar src (dim_size=5)
    dim_size = 5
    data_1d = make_arg((dim_size,))
    valid_index = torch.randint(0, dim_size, (), device=device, dtype=torch.long)
    scalar_src = make_arg(())
    yield opinfo_core.SampleInput(data_1d, args=(0, valid_index, scalar_src))

    # Test case: single-element tensor index, scalar src (dim_size=7)
    dim_size = 7
    data_1d = make_arg((dim_size,))
    valid_index_1d = torch.randint(0, dim_size, (1,), device=device, dtype=torch.long)
    scalar_src = make_arg(())
    yield opinfo_core.SampleInput(data_1d, args=(0, valid_index_1d, scalar_src))

    # Test case: scalar index, single-element tensor src (dim_size=3)
    dim_size = 3
    data_1d = make_arg((dim_size,))
    valid_index = torch.randint(0, dim_size, (), device=device, dtype=torch.long)
    src_1d = make_arg((1,))
    yield opinfo_core.SampleInput(data_1d, args=(0, valid_index, src_1d))

    # Test case: single-element tensor index, single-element tensor src (dim_size=10)
    dim_size = 10
    data_1d = make_arg((dim_size,))
    valid_index_1d = torch.randint(0, dim_size, (1,), device=device, dtype=torch.long)
    src_1d = make_arg((1,))
    yield opinfo_core.SampleInput(data_1d, args=(0, valid_index_1d, src_1d))


def sample_inputs_scatter_value(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    del kwargs
    make_arg = functools.partial(
        make_tensor, dtype=dtype, device=device, requires_grad=requires_grad
    )

    # Basic test cases for scatter.value
    cases = [
        # (self_shape, index_shape, dim, value)
        ((5, 5), (2, 3), 0, 1.0),  # 2D scatter on dim=0 with scalar value
        ((5, 5), (3, 2), 1, -2.5),  # 2D scatter on dim=1 with scalar value
        ((3, 4, 5), (2, 2, 3), 0, False),  # 3D scatter on dim=0 with scalar value
        ((3, 4, 5), (2, 2, 3), 1, 3.14),  # 3D scatter on dim=1 with scalar value
        ((3, 4, 5), (2, 2, 3), 2, -1),  # 3D scatter on dim=2 with scalar value
        ((10,), (3,), 0, 5.0),  # 1D scatter with scalar value
    ]

    for self_shape, index_shape, dim, value in cases:
        self_tensor = make_arg(self_shape)
        # Create valid indices for the given dimension without duplication
        index_buffer_shape = list(index_shape)
        index_buffer_shape[dim] = self_shape[dim]
        index_tensor = torch.rand(index_buffer_shape, device=device).argsort(dim=dim)[
            tuple(slice(None, d, None) for d in index_shape)
        ]
        yield opinfo_core.SampleInput(self_tensor, args=(dim, index_tensor, value))

    # Additional test cases for scalar and single-element tensor combinations with dim=0
    # Test case: scalar index with scalar value (dim_size=6, value_type=torch.long)
    dim_size = 6
    data_1d = make_arg((dim_size,))
    valid_index = torch.randint(0, dim_size, (), device=device, dtype=torch.long)
    random_value = torch.randint(0, 10, (), device=device, dtype=torch.long).item()
    yield opinfo_core.SampleInput(data_1d, args=(0, valid_index, random_value))

    # Test case: single-element tensor index with scalar value (dim_size=8, value_type=torch.float)
    dim_size = 8
    data_1d = make_arg((dim_size,))
    valid_index_1d = torch.randint(0, dim_size, (1,), device=device, dtype=torch.long)
    random_value = torch.rand((), device=device, dtype=torch.float).item()
    yield opinfo_core.SampleInput(data_1d, args=(0, valid_index_1d, random_value))


def sample_inputs_unfold(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    # Case `target_end == 1`, where `target_end = (input.size(dimension) - size) // step + 1`.
    t = make_tensor(
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


def sample_inputs_masked_scatter(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    del kwargs

    make_arg = functools.partial(
        make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
    )
    # (self_shape, mask_shape) with mask broadcastable to self. The same-rank
    # broadcasting cases (e.g. (1, 5, 4) / (1, 5, 1)) are the regression target for
    # pytorch/pytorch#186146 — the dynamo exporter previously left mask un-expanded.
    cases = (
        ((1, 5, 4), (1, 5, 4)),  # no broadcast
        ((1, 5, 4), (1, 5, 1)),  # same-rank broadcast over last dim
        ((1, 5, 4), (5, 4)),  # lower-rank mask
        ((2, 3), (2, 1)),  # same-rank broadcast
        ((3, 1), (3, 4)),  # self broadcast up to mask
    )
    for self_shape, mask_shape in cases:
        self_tensor = make_arg(self_shape)
        mask = torch.zeros(mask_shape, dtype=torch.bool, device=device)
        mask.view(-1)[::2] = True
        broadcast_shape = torch.broadcast_shapes(self_shape, mask_shape)
        num_selected = int(mask.expand(broadcast_shape).sum())
        source = make_arg((max(num_selected, 1),))
        yield opinfo_core.SampleInput(self_tensor, args=(mask, source))


OPS_DB = [
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
        "ops.aten.mean.dtype",
        op=torch.ops.aten.mean,
        aten_name="mean",
        dtypes=common_dtype.floating_types(),
        sample_inputs_func=sample_inputs_mean_dtype,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.slice_scatter",
        aten_name="slice_scatter",
        dtypes=common_dtype.all_types_and(torch.bfloat16, torch.half, torch.bool),
        sample_inputs_func=sample_inputs_slice_scatter,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.scatter.src",
        op=torch.ops.aten.scatter.src,
        aten_name="scatter.src",
        dtypes=common_dtype.all_types_and(torch.bfloat16, torch.half, torch.bool),
        sample_inputs_func=sample_inputs_scatter_src,
        supports_out=False,
    ),
    opinfo_core.OpInfo(
        "ops.aten.scatter.value",
        op=torch.ops.aten.scatter.value,
        aten_name="scatter.value",
        dtypes=common_dtype.all_types_and(torch.bfloat16, torch.half, torch.bool),
        sample_inputs_func=sample_inputs_scatter_value,
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
        "ops.aten.masked_scatter",
        aten_name="masked_scatter",
        op=torch.ops.aten.masked_scatter.default,
        dtypes=common_dtype.all_types(),
        sample_inputs_func=sample_inputs_masked_scatter,
        supports_out=False,
    ),
]


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
            t = make_tensor(
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
            t = make_tensor(
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
        t = make_tensor(
            shape,
            dtype=dtype,
            device=device,
            low=None,
            high=None,
            requires_grad=requires_grad,
        )
        yield opinfo_core.SampleInput(t, **kwargs)


def sample_inputs_normal_tensor_float(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    del requires_grad
    del kwargs
    make_arg = functools.partial(
        make_tensor, dtype=dtype, device=device, requires_grad=False
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
        make_tensor, dtype=dtype, device=device, requires_grad=False
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
        make_tensor, dtype=dtype, device=device, requires_grad=False
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
        make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
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


def sample_inputs_window_functions(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    del kwargs
    del device
    del requires_grad

    for window_length in [2, 3, 7, 10, 32]:
        yield opinfo_core.SampleInput(window_length, kwargs=dict(dtype=dtype))


OPS_DB.extend(
    [
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
    ]
)


def sample_inputs_grouped_mm(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    del kwargs

    make_arg = functools.partial(
        make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
    )
    # Native grouped_mm requires 16-byte-aligned strides, including on CPU.
    for shape_a, shape_b in (((2, 3, 8), (2, 8, 8)), ((1, 2, 8), (1, 8, 8))):
        yield opinfo_core.SampleInput(make_arg(shape_a), args=(make_arg(shape_b),))

    for shape_a, shape_b in (
        ((24, 8), (3, 8, 8)),
        ((3, 8, 8), (8, 24)),
        ((8, 24), (24, 8)),
    ):
        for boundaries in ([4, 12, 24], [8, 8, 24]):
            offsets = torch.tensor(boundaries, dtype=torch.int32, device=device)
            yield opinfo_core.SampleInput(
                make_arg(shape_a), args=(make_arg(shape_b), offsets)
            )


def sample_inputs_bilinear(op_info, device, dtype, requires_grad, **kwargs):
    """Sample inputs for bilinear operation."""
    del op_info
    del kwargs

    make_arg = functools.partial(
        make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
    )

    # Test cases: (batch_size, in1_features, in2_features, out_features)
    cases = [
        (2, 3, 4, 5),  # Basic case
        (1, 2, 2, 1),  # Minimal case
        (3, 5, 7, 4),  # Different dimensions
        (2, 1, 1, 3),  # Single input features
    ]

    for batch_size, in1_features, in2_features, out_features in cases:
        input1 = make_arg((batch_size, in1_features))
        input2 = make_arg((batch_size, in2_features))
        weight = make_arg((out_features, in1_features, in2_features))
        bias = make_arg((out_features,))

        # Test with bias
        yield opinfo_core.SampleInput(input1, args=(input2, weight, bias))

        # Test without bias (only for first case to avoid too many tests)
        if batch_size == 2:
            yield opinfo_core.SampleInput(input1, args=(input2, weight, None))


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
        # when the initial condition is False, it still executes the loop body once.
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


def sample_inputs_fake_quantize_per_tensor_affine(
    op_info, device, dtype, requires_grad, **kwargs
):
    del op_info, kwargs  # Unused
    make_arg = functools.partial(
        opinfo_core.make_tensor,
        device=device,
        requires_grad=requires_grad,
    )

    # Test 1D, empty and scalar tensors (like sample_inputs_elementwise_unary)
    shapes = [
        (S,),
        (1, 0, 3),
        (),
    ]

    scale_zero_point_dtypes = [
        # default (float, int)
        (None, None)
    ] + [
        # tensor_qparams (tensor, tensor)
        (t1, t2)
        for t1 in common_dtype.all_types_and()
        for t2 in common_dtype.all_types_and()
    ]

    # NOTE: (0, 127) is allowed as special case. PyTorch restricts activations to be in the range (0, 127).
    #   https://github.com/pytorch/pytorch/blob/b34b192d6b97325c9f78e5995c48c8498ede34bd/torch/ao/quantization/observer.py#L1422
    quant_vals = [(0, 255), (-128, 127), (0, 127)]

    cases = itertools.product(shapes, scale_zero_point_dtypes, quant_vals)
    for shape, (scale_dtype, zero_point_dtype), (quant_min, quant_max) in cases:
        scale = make_arg(
            (),
            dtype=scale_dtype or torch.float64,
        )
        if scale_dtype is None:
            scale = scale.item()

        zero_point = make_arg(
            (),
            dtype=zero_point_dtype or torch.int64,
            # zero_point must be between quant_min and quant_max
            low=quant_min,
            high=quant_max,
        )
        if zero_point_dtype is None:
            zero_point = zero_point.item()

        args = (scale, zero_point, quant_min, quant_max)
        yield opinfo_core.SampleInput(make_arg(shape, dtype=dtype), args=args)


def sample_inputs_fake_quantize_per_channel_affine(
    op_info, device, dtype, requires_grad, **kwargs
):
    del op_info, kwargs  # Unused
    make_arg = functools.partial(
        opinfo_core.make_tensor,
        device=device,
        requires_grad=requires_grad,
    )

    # Test 1D, 2D, 4D and empty tensors (scalar tensors not supported)
    axes_and_shapes = [
        # 1D, 2D, 4D
        (axis, (S,) * dims)
        for dims in (1, 2, 4)
        for axis in range(dims)
    ] + [
        # empty
        (0, (1, 0, 3)),
        (2, (1, 0, 3)),
        # empty channel axis causes an error due to
        # an internal zero_point.min() calculation
        # (1, (1, 0, 3)),
    ]

    # tensor_qparams
    scale_dtype = torch.float
    zero_point_dtypes = [torch.int32, torch.float, torch.half]

    # NOTE: (0, 127) is allowed as special case. PyTorch restricts activations to be in the range (0, 127).
    #   https://github.com/pytorch/pytorch/blob/b34b192d6b97325c9f78e5995c48c8498ede34bd/torch/ao/quantization/observer.py#L1422
    quant_vals = [(0, 255), (-128, 127), (0, 127)]

    cases = itertools.product(axes_and_shapes, zero_point_dtypes, quant_vals)
    for (axis, shape), zero_point_dtype, (quant_min, quant_max) in cases:
        scale = make_arg((shape[axis],), dtype=scale_dtype)

        zero_point = make_arg(
            (shape[axis],),
            dtype=zero_point_dtype or torch.int64,
            # zero_point must be between quant_min and quant_max
            low=quant_min,
            high=quant_max,
        )

        args = (scale, zero_point, axis, quant_min, quant_max)
        yield opinfo_core.SampleInput(make_arg(shape, dtype=dtype), args=args)


def sample_inputs__log_softmax(
    op_info,
    device,
    dtype,
    requires_grad,
    **kwargs,
):
    del op_info  # Unused

    make_arg = functools.partial(
        make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
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


def sample_inputs_native_dropout(
    op_info, device, dtype, requires_grad, *, valid_input_dim=None, **kwargs
):
    del op_info  # Unused
    del kwargs  # Unused
    make_arg = functools.partial(
        make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
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


def sample_inputs__softmax(
    op_info,
    device,
    dtype,
    requires_grad,
    **kwargs,
):
    del op_info  # Unused

    make_arg = functools.partial(
        make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
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


def sample_inputs_stft(op_info, device, dtype, requires_grad, **kwargs):
    del op_info
    del kwargs

    def mt(shape, **kwargs):
        return make_tensor(
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


def sample_inputs__unique(op_info, device, dtype, requires_grad, **kwargs):
    for sample in common_methods_invocations.sample_inputs_unique(
        op_info, device, dtype, requires_grad, **kwargs
    ):
        return_counts = sample.kwargs.pop("return_counts", None)
        dim = sample.kwargs.pop("dim", None)
        # take only those samples that do not ask for counts or a dim
        if not return_counts and dim is None:
            yield sample


def sample_inputs__unique2(op_info, device, dtype, requires_grad, **kwargs):
    for sample in common_methods_invocations.sample_inputs_unique(
        op_info, device, dtype, requires_grad, **kwargs
    ):
        # take only those samples that do not ask for a dim
        if sample.kwargs.pop("dim", None) is None:
            yield sample


def sample_inputs_unique_dim(op_info, device, dtype, requires_grad, **kwargs):
    for sample in common_methods_invocations.sample_inputs_unique(
        op_info, device, dtype, requires_grad, **kwargs
    ):
        # take only those samples that ask for a dim
        if sample.kwargs.get("dim") is not None:
            yield sample


OPS_DB.extend(
    [
        opinfo_core.OpInfo(
            "bilinear",
            op=torch.nn.functional.bilinear,
            dtypes=common_dtype.floating_types(),
            sample_inputs_func=sample_inputs_bilinear,
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
            "ops.aten.fake_quantize_per_tensor_affine",
            aten_name="fake_quantize_per_tensor_affine",
            op=torch.fake_quantize_per_tensor_affine,
            dtypes=common_dtype.floating_types_and(torch.half, torch.bfloat16),
            sample_inputs_func=sample_inputs_fake_quantize_per_tensor_affine,
            supports_out=False,
        ),
        opinfo_core.OpInfo(
            "ops.aten.fake_quantize_per_channel_affine",
            aten_name="fake_quantize_per_channel_affine",
            op=torch.fake_quantize_per_channel_affine,
            dtypes=common_dtype.floating_types_and(torch.half, torch.bfloat16),
            sample_inputs_func=sample_inputs_fake_quantize_per_channel_affine,
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
            "ops.aten.native_dropout",
            aten_name="native_dropout",
            dtypes=common_dtype.all_types_and_half(),
            sample_inputs_func=sample_inputs_native_dropout,
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
        opinfo_core.OpInfo(
            "ops.aten.stft",
            aten_name="stft",
            dtypes=common_dtype.floating_and_complex_types_and(
                torch.half, torch.bfloat16
            ),
            sample_inputs_func=sample_inputs_stft,
            supports_out=False,
        ),
        opinfo_core.OpInfo(
            "ops.aten._unique.default",
            aten_name="_unique.default",
            dtypes=common_dtype.floating_types_and(
                torch.float16, torch.int64, torch.int8
            ),
            sample_inputs_func=sample_inputs__unique,
            supports_out=False,
            supports_autograd=False,
        ),
        opinfo_core.OpInfo(
            "ops.aten._unique2.default",
            aten_name="_unique2.default",
            dtypes=common_dtype.floating_types_and(
                torch.float16, torch.int64, torch.int8
            ),
            sample_inputs_func=sample_inputs__unique2,
            supports_out=False,
            supports_autograd=False,
        ),
        opinfo_core.OpInfo(
            "ops.aten.unique_dim.default",
            aten_name="unique_dim.default",
            dtypes=common_dtype.floating_types_and(
                torch.float16, torch.int64, torch.int8
            ),
            sample_inputs_func=sample_inputs_unique_dim,
            supports_out=False,
            supports_autograd=False,
        ),
        opinfo_core.OpInfo(
            "ops.aten._grouped_mm",
            aten_name="_grouped_mm",
            op=getattr(torch.ops.aten, "_grouped_mm", lambda *args, **kwargs: None),
            dtypes=common_dtype.floating_types(),
            sample_inputs_func=sample_inputs_grouped_mm,
            supports_out=False,
        ),
    ]
)
