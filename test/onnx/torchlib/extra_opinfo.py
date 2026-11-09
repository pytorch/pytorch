# Owner(s): ["module: onnx"]
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Additional source torchlib samples for native structural translations."""

from __future__ import annotations

import functools

import torch
from torch.testing import make_tensor
from torch.testing._internal import common_dtype, common_methods_invocations
from torch.testing._internal.opinfo import core as opinfo_core


S = 5


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
