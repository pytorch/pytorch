# mypy: ignore-errors

from copy import copy
from functools import partial

import torch
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.opinfo.core import (
    BinaryUfuncInfo,
    ReductionOpInfo,
    SampleInput,
    UnaryUfuncInfo,
)
from torch.utils._pytree import tree_map


# random integer used for sizes
def _rnd():
    return torch.randint(3, 8, ()).item()


def _raggedness_matches(nt1, nt2):
    return (
        nt1.is_nested
        and nt2.is_nested
        and nt1._ragged_idx == nt2._ragged_idx
        and nt1.shape[nt1._ragged_idx] == nt2.shape[nt2._ragged_idx]
    )


# Generates a random NT.
# dims should be something like [5, None, 10], with None indicating that a
# random ragged structure should be used
def random_nt_from_dims(
    dims, device=None, dtype=None, layout=torch.strided, requires_grad=False
):
    sizes = [[d if d is not None else _rnd() for d in dims[1:]] for d in range(dims[0])]
    return torch.nested.nested_tensor(
        [torch.randn(*size) for size in sizes],
        device=device,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
    )


# Helper function for generating a comprehensive set of NJT sample inputs.
def _sample_njts(device, dtype, requires_grad=False, dims=None):
    if dims is None:
        dims = [2, 3, 4]
    if not isinstance(dims, (list, tuple)):
        dims = [dims]

    # contiguous NJTs
    for dim in dims:
        # with min / max seqlen cached
        shape = (_rnd(), None, *[_rnd() for _ in range(dim - 2)])
        nt = random_nt_from_dims(
            shape,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
            layout=torch.jagged,
        )
        yield nt

        # without min / max seqlen cached
        values = nt.values().clone().detach()
        offsets = nt.offsets().clone().detach()
        yield torch.nested.nested_tensor_from_jagged(values, offsets)

    # TODO: add non-contiguous NJTs


# Computes an unbind-based reference for a given OpInfo on a given SampleInput.
# This reference unbinds the input NJT and invokes the op on each of the components,
# optionally wrapping the result in an NJT.
def unbind_reference(op, sample, wrap_output_as_njt=True):
    # first NJT in the arglist determines expected ragged structure
    nt_inp = (
        sample.input
        if sample.input.is_nested
        # TODO: look in kwargs too?
        else next(a for a in sample.args if a.is_nested)
    )

    out_ref_components = []
    for i in range(nt_inp.shape[0]):

        def _slice_input(t, i=i, inp=nt_inp):
            # any NJT with the same ragged structure as the input should
            # be sliced to pass to the reference
            if isinstance(t, torch.Tensor) and _raggedness_matches(t, inp):
                return t[i]
            # allow the SampleInput to tell us how to slice it for ref calculation
            elif isinstance(t, torch.Tensor) and hasattr(t, "_batch_dim"):
                bdim = t._batch_dim  # type: ignore[attr]
                if t.shape[bdim] == 1:
                    return t[0]
                else:
                    return t.select(bdim, i)
            else:
                return t

        inp = _slice_input(sample.input)
        args = tree_map(_slice_input, sample.args)
        kwargs = tree_map(_slice_input, sample.kwargs)

        from torch._prims_common import canonicalize_dims

        # Need to adjust dim to apply on NJT component
        if "dim" in kwargs:
            kwargs["dim"] = canonicalize_dims(nt_inp.dim(), kwargs["dim"]) - 1
            assert kwargs["dim"] >= 0

        # TODO: handle this
        assert "dims" not in kwargs

        out_ref_component = op.op(inp, *args, **kwargs)

        # TODO: handle list / tuple / non-NJT outputs
        assert not isinstance(out_ref_component, (list, tuple))
        out_ref_components.append(out_ref_component)

    if wrap_output_as_njt:
        return torch.nested.as_nested_tensor(out_ref_components, layout=torch.jagged)

    return out_ref_components


# Computes the reference value for a reduction op.
def reduction_reference(op, sample):
    assert sample.input.is_nested
    dim = sample.kwargs.get("dim", None)
    assert dim != 0, "reductions over the batch dim are not supported"
    assert "dims" not in sample.kwargs
    assert sample.input._ragged_idx == 1

    if dim is None:
        # calculate reference value by running reduction on values buffer
        return op.op(sample.input.values(), *sample.args, **sample.kwargs)

    if dim == sample.input._ragged_idx:
        # calculate reference value by running an unbind reference and stacking
        out_ref_components = unbind_reference(op, sample, wrap_output_as_njt=False)
        return torch.stack(out_ref_components, dim=0)

    # unbind reference works for other reductions
    return unbind_reference(op, sample)


def sample_inputs_elementwise_njt_unary(
    op_info, device, dtype, requires_grad, op_kwargs=None, **kwargs
):
    if not op_kwargs:
        op_kwargs = {}

    for njt in _sample_njts(
        device=device, dtype=dtype, requires_grad=requires_grad, dims=[2, 3, 4]
    ):
        yield SampleInput(njt, kwargs=dict(op_kwargs))


def sample_inputs_elementwise_njt_binary(
    op_info, device, dtype, requires_grad, op_kwargs=None, **kwargs
):
    if not op_kwargs:
        op_kwargs = {}

    for njt1 in _sample_njts(
        device=device, dtype=dtype, requires_grad=requires_grad, dims=[2, 3, 4]
    ):
        # TODO: account for non-contiguous NJTs here
        njt2 = torch.randn_like(njt1)
        yield SampleInput(njt1, args=(njt2,), kwargs=dict(op_kwargs))

        # broadcasting case: (B, j0, ...) with (B, 1, ...)
        t = torch.randn(
            (njt1.shape[0], 1, *njt1.shape[2:]),
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        # used for slicing in unbind_reference()
        t._batch_dim = 0
        # (NT, T)
        yield SampleInput(njt1, args=(t,), kwargs=dict(op_kwargs))
        # (T, NT)
        yield SampleInput(t, args=(njt1,), kwargs=dict(op_kwargs))

        # broadcasting case: (B, j0, ...) with (1, 1...)
        t = torch.randn(
            [1 for _ in range(njt1.dim())],
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        # used for slicing in unbind_reference()
        t._batch_dim = 0
        # (NT, T)
        yield SampleInput(njt1, args=(t,), kwargs=dict(op_kwargs))
        # (T, NT)
        yield SampleInput(t, args=(njt1,), kwargs=dict(op_kwargs))

        # broadcasting case: (B, j0, ...) with (...)
        t = torch.randn(
            njt1.shape[2:], device=device, dtype=dtype, requires_grad=requires_grad
        )
        # (NT, T)
        yield SampleInput(njt1, args=(t,), kwargs=dict(op_kwargs))
        # (T, NT)
        yield SampleInput(t, args=(njt1,), kwargs=dict(op_kwargs))

        # broadcasting case: (B, j0, ...) with scalar
        t = torch.randn((), device=device, dtype=dtype, requires_grad=requires_grad)
        # (NT, T)
        yield SampleInput(njt1, args=(t,), kwargs=dict(op_kwargs))
        # (T, NT)
        yield SampleInput(t, args=(njt1,), kwargs=dict(op_kwargs))

    # mixed broadcasting case: (B, j0, 1) with (B, 1, D)
    B = 4
    D = 16
    njt = random_nt_from_dims(
        (B, None, 1),
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        layout=torch.jagged,
    )
    t = torch.randn(B, 1, D, device=device, dtype=dtype, requires_grad=requires_grad)
    # used for slicing in unbind_reference()
    t._batch_dim = 0

    # (NT, T)
    yield SampleInput(njt, args=(t,), kwargs=dict(op_kwargs))
    # (T, NT)
    yield SampleInput(t, args=(njt,), kwargs=dict(op_kwargs))


def sample_inputs_njt_reduction(
    op_info, device, dtype, requires_grad, op_kwargs=None, **kwargs
):
    if not op_kwargs:
        op_kwargs = {}

    for njt in _sample_njts(
        device=device, dtype=dtype, requires_grad=requires_grad, dims=[2, 3, 4]
    ):
        # dim-wise reduction; includes reduction over the ragged dim
        # NB: reduction over the batch dim is not supported!
        # TODO: Cover this in the set of error inputs
        for dim in range(1, njt.dim()):
            for keepdim in [False, True]:
                yield SampleInput(
                    njt, kwargs={**op_kwargs, "dim": dim, "keepdim": keepdim}
                )

        # full reduction
        yield SampleInput(njt, kwargs=dict(op_kwargs))


def unsupported_sample_inputs_func(op_name):
    def _f(op_info, device, dtype, requires_grad, op_name=op_name, **kwargs):
        raise RuntimeError(
            f"OpInfo for {op_name} does not support NJT. Support can be added by modifying "
            "torch/testing/_internal/opinfo/definitions/nested.py."
        )

    return _f


def unsupported_reference(op_name):
    def _f(op, sample):
        raise RuntimeError(
            f"OpInfo for {op_name} does not define a ref() function. Support can be added by "
            "modifying torch/testing/_internal/opinfo/definitions/nested.py."
        )

    return _f


# === BEGIN OP-SPECIFIC SAMPLE INPUTS FUNCS ===
def sample_inputs_clone(op_info, device, dtype, requires_grad, **kwargs):
    # non-contiguous NJTs
    for njt in _sample_njts(
        device=device, dtype=dtype, requires_grad=requires_grad, dims=[2, 3, 4]
    ):
        yield SampleInput(njt)

    for memory_format in (torch.contiguous_format, torch.preserve_format):
        # construct a "non-contiguous with holes" NJT
        values = torch.randn(
            10, 5, device=device, dtype=dtype, requires_grad=requires_grad
        )
        offsets = torch.tensor([0, 2, 4, 10], device=device, dtype=torch.int64)
        lengths = torch.tensor([2, 1, 3], device=device, dtype=torch.int64)
        njt = torch.nested.nested_tensor_from_jagged(
            values, offsets=offsets, lengths=lengths
        )

        yield SampleInput(njt, kwargs={"memory_format": memory_format})


def sample_inputs_mvl_gamma(p):
    return partial(sample_inputs_elementwise_njt_unary, op_kwargs={"p": p})


def sample_inputs_polygamma_n(n):
    return partial(sample_inputs_elementwise_njt_unary, op_kwargs={"n": n})


def sample_inputs_special_polygamma_n(n):
    return partial(sample_inputs_elementwise_njt_unary, op_kwargs={"n": n})


def sample_inputs_to(op_info, device, dtype, requires_grad, op_kwargs=None, **kwargs):
    for njt in _sample_njts(
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        dims=[2, 3, 4],
    ):
        other_dtypes = (
            d for d in (torch.float32, torch.half, torch.double) if d is not dtype
        )
        for other_dtype in other_dtypes:
            sample_name = f"{njt.dim()}D: {dtype} -> {other_dtype}"
            yield SampleInput(
                njt.clone().detach(), kwargs={"dtype": dtype}, name=sample_name
            )

        # only include device transfer for CUDA inputs
        if "cuda" in device:
            other_device = "cpu"
            sample_name = f"{njt.dim()}D: {device} -> {other_device}"
            yield SampleInput(
                njt.clone().detach(), kwargs={"device": other_device}, name=sample_name
            )


def sample_inputs_bmm(op_info, device, dtype, requires_grad, op_kwargs=None, **kwargs):
    for njt_3d in _sample_njts(
        device=device, dtype=dtype, requires_grad=requires_grad, dims=[3]
    ):
        # (B, j1, D) x (B, D, E) => (B, j1, E)
        B, D = njt_3d.shape[0], njt_3d.shape[-1]
        E = D + 2
        other = torch.randn(B, D, E, device=device, dtype=dtype)
        # used for slicing in unbind_reference()
        other._batch_dim = 0
        yield SampleInput(njt_3d.clone().detach(), kwargs={"mat2": other})

        # TODO (need factory functions):
        # (B, D, j1) x (B, j1, E) => (B, D, E)


def reference_bmm(op, sample):
    # unbind reduces a dim and bmm requires 3D, so use matmul as the reference
    matmul_op = copy(op)
    matmul_op.op = torch.matmul
    # change arg name from mat2 -> other
    modified_sample = copy(sample)
    other = modified_sample.kwargs["mat2"]
    del modified_sample.kwargs["mat2"]
    modified_sample.kwargs["other"] = other
    return unbind_reference(matmul_op, modified_sample)


def sample_inputs_matmul(
    op_info, device, dtype, requires_grad, op_kwargs=None, **kwargs
):
    # also run bmm samples through
    for sample_input in sample_inputs_bmm(op_info, device, dtype, requires_grad):
        # change arg name from mat2 -> other
        other = sample_input.kwargs["mat2"]
        del sample_input.kwargs["mat2"]
        sample_input.kwargs["other"] = other
        yield sample_input

    # 3D cases not covered by bmm
    for njt_3d in _sample_njts(
        device=device, dtype=dtype, requires_grad=requires_grad, dims=[3]
    ):
        # (B, j1, D) x (D, E) => (B, j1, E)
        D = njt_3d.shape[-1]
        E = D + 2
        yield SampleInput(
            njt_3d.clone().detach(),
            kwargs={"other": torch.randn(D, E, device=device, dtype=dtype)},
        )

    # 4D cases
    for njt_4d in _sample_njts(
        device=device, dtype=dtype, requires_grad=requires_grad, dims=[4]
    ):
        # (B, j1, D, E) x (E, F) => (B, j1, D, F)
        E = njt_4d.shape[-1]
        F = E + 2
        yield SampleInput(
            njt_4d.clone().detach(),
            kwargs={"other": torch.randn(E, F, device=device, dtype=dtype)},
        )

        # TODO (need factory functions):
        # (B, j1, D, E) x (B, j1, E, F) => (B, j1, D, F)


def sample_inputs_masked_select(
    op_info, device, dtype, requires_grad, op_kwargs=None, **kwargs
):
    for njt in _sample_njts(
        device=device, dtype=dtype, requires_grad=requires_grad, dims=[2]
    ):
        yield SampleInput(
            njt, kwargs={"mask": (torch.randn_like(njt, requires_grad=False) < 0.0)}
        )


def sample_inputs_nn_functional_embedding_bag(
    op_info, device, dtype, requires_grad, **kwargs
):
    for generate_per_sample_weight in (True, False):
        for mode in ("sum", "mean", "max"):
            # per_sample_weights is only supported for mode='sum'
            if mode != "sum" and generate_per_sample_weight:
                continue

            NUM_EMBEDDINGS = 10
            EMBEDDING_DIM = 32
            weight = torch.randn(
                NUM_EMBEDDINGS, EMBEDDING_DIM, dtype=dtype, device=device
            )

            njt = torch.nested.nested_tensor(
                [
                    torch.randint(0, NUM_EMBEDDINGS, size=(2,)),
                    torch.randint(0, NUM_EMBEDDINGS, size=(3,)),
                    torch.randint(0, NUM_EMBEDDINGS, size=(4,)),
                ],
                layout=torch.jagged,
                dtype=torch.int64,
                device=device,
            )

            per_sample_weights = None
            if generate_per_sample_weight:
                per_sample_weights = torch.randn_like(njt, dtype=dtype)

            # NB: the OpInfo entry for embedding_bag expects weight first so the gradients
            # can be checked
            yield SampleInput(
                weight,
                args=(njt,),
                kwargs={
                    "mode": mode,
                    "per_sample_weights": per_sample_weights,
                },
            )


def reference_nn_functional_embedding_bag(op, sample):
    # run reference on a single bag at a time
    new_kwargs = dict(sample.kwargs)
    new_kwargs.update(
        {"offsets": torch.tensor([0], dtype=torch.int64, device=sample.input.device)}
    )
    # flip input / weight back to what unbind_reference() expects
    sample = SampleInput(sample.args[0], args=(sample.input,), kwargs=new_kwargs)
    old_op = op.op
    op.op = torch.nn.functional.embedding_bag
    output = unbind_reference(op, sample, wrap_output_as_njt=False)
    op.op = old_op
    # concat bag outputs to get final output
    return torch.cat(output, dim=0)


def sample_inputs_nn_functional_linear(op_info, device, dtype, requires_grad, **kwargs):
    for njt in _sample_njts(
        device=device, dtype=dtype, requires_grad=requires_grad, dims=[3, 4, 5]
    ):
        # with bias
        NUM_OUTPUT = 10
        weight = torch.randn(
            NUM_OUTPUT,
            njt.size(-1),
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        bias = torch.randn(
            NUM_OUTPUT, device=device, dtype=dtype, requires_grad=requires_grad
        )
        yield SampleInput(
            njt,
            kwargs={
                "weight": weight,
                "bias": bias,
            },
        )

        # without bias
        yield SampleInput(
            njt,
            kwargs={
                "weight": weight,
            },
        )


def sample_inputs_nn_functional_rms_norm(
    op_info, device, dtype, requires_grad, **kwargs
):
    for njt in _sample_njts(
        device=device, dtype=dtype, requires_grad=requires_grad, dims=[3, 4]
    ):
        # normalize over non-ragged dims
        for start_dim in range(2, njt.dim()):
            normalized_shape = njt.shape[start_dim:]
            weight = torch.randn(
                normalized_shape,
                device=device,
                dtype=dtype,
                requires_grad=requires_grad,
            )

            yield SampleInput(
                njt,
                kwargs={
                    "normalized_shape": normalized_shape,
                    "weight": weight,
                },
            )


sample_inputs_nn_functional_threshold = partial(
    sample_inputs_elementwise_njt_unary,
    op_kwargs={"threshold": float.fromhex("0x1.3ap-3"), "value": -9},
)
# === END OP-SPECIFIC SAMPLE INPUTS FUNCS ===


# Mapping of OpInfo full names -> sample_inputs_funcs, which define the set of sample inputs
# (involving NJTs) to pass to the op. Full name consists of the OpInfo's name and variant name
# separated by a period (e.g. special.polygamma.special_polygamma_n_0). These are necessary
# to specify if they cannot be auto-generated for some reason. Try to keep these sorted
# in alphabetical order!
njt_sample_inputs = {
    "bmm": sample_inputs_bmm,
    "clone": sample_inputs_clone,
    **{f"mvlgamma.mvlgamma_p_{p}": sample_inputs_mvl_gamma(p=1) for p in (1, 3, 5)},
    "nn.functional.embedding_bag": sample_inputs_nn_functional_embedding_bag,
    "nn.functional.linear": sample_inputs_nn_functional_linear,
    "nn.functional.rms_norm": sample_inputs_nn_functional_rms_norm,
    "nn.functional.threshold": sample_inputs_nn_functional_threshold,
    **{f"polygamma.polygamma_n_{n}": sample_inputs_polygamma_n(n=n) for n in range(5)},
    "special.polygamma.special_polygamma_n_0": sample_inputs_special_polygamma_n(n=0),
    "to": sample_inputs_to,
    "matmul": sample_inputs_matmul,
    "masked_select": sample_inputs_masked_select,
}

njt_references = {
    "bmm": reference_bmm,
    "nn.functional.embedding_bag": reference_nn_functional_embedding_bag,
}


# Translates an OpInfo entry to one that operates on NJTs.
def translate_opinfo(op):
    new_op = copy(op)
    new_op.supports_njt = True

    if op.full_name in njt_sample_inputs:
        new_op.sample_inputs_func = njt_sample_inputs[op.full_name]
        new_op.ref = njt_references.get(op.full_name, unbind_reference)
    elif isinstance(op, UnaryUfuncInfo):
        new_op.sample_inputs_func = partial(
            sample_inputs_elementwise_njt_unary, op_kwargs=None
        )
        new_op.ref = unbind_reference
    elif isinstance(op, BinaryUfuncInfo):
        new_op.sample_inputs_func = partial(
            sample_inputs_elementwise_njt_binary, op_kwargs=None
        )
        new_op.ref = unbind_reference
    elif isinstance(op, ReductionOpInfo):
        new_op.sample_inputs_func = partial(sample_inputs_njt_reduction, op_kwargs=None)
        new_op.ref = reduction_reference
    # TODO: Translate the rest of the OpInfos
    else:
        new_op.sample_inputs_func = unsupported_sample_inputs_func(op.full_name)
        new_op.ref = unsupported_reference(op.full_name)
        new_op.supports_njt = False

    return new_op


njt_op_db = [translate_opinfo(op) for op in op_db]
