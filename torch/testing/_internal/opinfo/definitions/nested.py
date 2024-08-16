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
    assert sample.input.is_nested
    out_ref_components = []
    for i, component in enumerate(sample.input.unbind(dim=0)):

        def _slice_njts(t, i=i, inp=sample.input):
            # any NJT with the same ragged structure as the input should
            # also be sliced to pass to the reference
            if isinstance(t, torch.Tensor) and _raggedness_matches(t, inp):
                return t[i]
            else:
                return t

        args = tree_map(_slice_njts, sample.args)
        kwargs = tree_map(_slice_njts, sample.kwargs)

        from torch._prims_common import canonicalize_dims

        # Need to adjust dim to apply on NJT component
        if "dim" in kwargs:
            kwargs["dim"] = canonicalize_dims(sample.input.dim(), kwargs["dim"]) - 1
            assert kwargs["dim"] >= 0

        # TODO: handle this
        assert "dims" not in kwargs

        out_ref_component = op.op(component, *args, **kwargs)

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
    keepdim = sample.kwargs.get("keepdim", False)
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
        # TODO: provide sample inputs for broadcasting cases and mixed (NT, T), (T, NT) inputs
        njt2 = torch.randn_like(njt1)
        yield SampleInput(njt1, args=(njt2,), kwargs=dict(op_kwargs))


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


def sample_inputs_masked_select(
    op_info, device, dtype, requires_grad, op_kwargs=None, **kwargs
):
    for njt in _sample_njts(
        device=device, dtype=dtype, requires_grad=requires_grad, dims=[1]
    ):
        yield SampleInput(
            njt, kwargs={"mask": (torch.randn_like(njt, requires_grad=False) < 0.0)}
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
    "clone": sample_inputs_clone,
    **{f"mvlgamma.mvlgamma_p_{p}": sample_inputs_mvl_gamma(p=1) for p in (1, 3, 5)},
    "nn.functional.threshold": sample_inputs_nn_functional_threshold,
    **{f"polygamma.polygamma_n_{n}": sample_inputs_polygamma_n(n=n) for n in range(5)},
    "special.polygamma.special_polygamma_n_0": sample_inputs_special_polygamma_n(n=0),
    "masked_select": sample_inputs_masked_select,
}


# Translates an OpInfo entry to one that operates on NJTs.
def translate_opinfo(op):
    new_op = copy(op)
    new_op.supports_njt = True

    if op.full_name in njt_sample_inputs:
        new_op.sample_inputs_func = njt_sample_inputs[op.full_name]
        # TODO: make the reference customizeable
        new_op.ref = unbind_reference
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
