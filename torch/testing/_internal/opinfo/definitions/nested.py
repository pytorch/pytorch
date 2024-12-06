# mypy: ignore-errors

import math
from copy import copy
from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Tuple

import torch
from torch.fx.experimental.symbolic_shapes import is_nested_int
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.opinfo.core import (
    BinaryUfuncInfo,
    ReductionOpInfo,
    SampleInput,
    UnaryUfuncInfo,
)
from torch.utils._pytree import tree_flatten, tree_map


@dataclass
class ExtraOpData:
    """
    Contains info on top of the typical OpInfo data that is useful for NJT test generation.

    The process that converts the standard op_db -> an NJT-compatible op_db will attach this
    data onto each associated OpInfo entry.
    """

    # Indicates whether the associated op is a view op
    is_view: bool = False

    # Specifies the names of any dim-related args that the op takes in. This is useful
    # for NJT tests because there is often asymmetry across the supported set of dims for
    # an op; it may make sense to operate over the batch dim but not the ragged dim, for
    # example. The length of this list should match the number of relevant overloads.
    # Each list item of the outer list should specify dim argnames. Ellipses should be used
    # to indicate multi-dim support for a given overload.
    #
    # For example, squeeze() has both a dim and multi-dim overload, where the argname for
    # each is simply "dim". Its entry should be: [["dim"], ["dim..."]].
    #
    # If no overload of the op accepts dim-related args, this should be None.
    dim_args: List[List[str]] = None

    # Helper function to extract names of dim-related args.
    # Returns: tuple of (single dim argname if available, dim list argname if available)
    # If the op doesn't support dim-related args at all OR this op only has overloads
    # with multiple dim args (e.g. transpose()), then this returns (None, None).
    def get_dim_argnames(self) -> Tuple[Optional[str], Optional[str]]:
        if self.dim_args is None:
            return (None, None)

        # name for the dim arg that supports a single dim
        single_dim_argname = None
        # name for the dim arg that supports a list of dims
        dimlist_argname = None
        for overload in self.dim_args:
            # only consider overloads with a single dim-related arg
            if len(overload) != 1:
                continue
            if overload[0].endswith("..."):
                dimlist_argname = overload[0].replace("...", "")
                if single_dim_argname is None:
                    single_dim_argname = dimlist_argname
            else:
                single_dim_argname = overload[0]
        return (single_dim_argname, dimlist_argname)


# Mapping of OpInfo full names -> extra data to tack onto the OpInfo entry for use
# in test generation.
extra_op_data = {
    "_segment_reduce.lengths": ExtraOpData(dim_args=[["axis0"]]),
    "_segment_reduce.offsets": ExtraOpData(dim_args=[["axis0"]]),
    "all": ExtraOpData(dim_args=[["dim"], ["dim..."]]),
    "argmax": ExtraOpData(dim_args=[["dim"]]),
    "argmin": ExtraOpData(dim_args=[["dim"]]),
    "amax": ExtraOpData(dim_args=[["dim..."]]),
    "amin": ExtraOpData(dim_args=[["dim..."]]),
    "any": ExtraOpData(dim_args=[["dim"], ["dim..."]]),
    "argsort": ExtraOpData(dim_args=[["dim"]]),
    "broadcast_to": ExtraOpData(is_view=True),
    "cat": ExtraOpData(dim_args=[["dim"]]),
    "chunk": ExtraOpData(is_view=True, dim_args=[["dim"]]),
    "conj": ExtraOpData(is_view=True),
    "contiguous": ExtraOpData(is_view=True),
    "count_nonzero": ExtraOpData(dim_args=[["dim"], ["dim..."]]),
    "cummax": ExtraOpData(dim_args=[["dim"]]),
    "cummin": ExtraOpData(dim_args=[["dim"]]),
    "cumprod": ExtraOpData(dim_args=[["dim"]]),
    "cumsum": ExtraOpData(dim_args=[["dim"]]),
    "cumulative_trapezoid": ExtraOpData(dim_args=[["dim"]]),
    "diag_embed": ExtraOpData(dim_args=[["dim1", "dim2"]]),
    "diagonal": ExtraOpData(is_view=True, dim_args=[["dim1", "dim2"]]),
    "diagonal_copy": ExtraOpData(dim_args=[["dim1", "dim2"]]),
    "diagonal_scatter": ExtraOpData(dim_args=[["dim1", "dim2"]]),
    "diff": ExtraOpData(dim_args=[["dim"]]),
    "expand": ExtraOpData(is_view=True),
    "expand_as": ExtraOpData(is_view=True),
    "fft.fft": ExtraOpData(dim_args=[["dim"]]),
    "fft.hfft": ExtraOpData(dim_args=[["dim"]]),
    "fft.ifft": ExtraOpData(dim_args=[["dim"]]),
    "fft.ihfft": ExtraOpData(dim_args=[["dim"]]),
    "fft.irfft": ExtraOpData(dim_args=[["dim"]]),
    "fft.rfft": ExtraOpData(dim_args=[["dim"]]),
    "flatten": ExtraOpData(is_view=True, dim_args=[["start_dim", "end_dim"]]),
    "flip": ExtraOpData(dim_args=[["dims..."]]),
    "gather": ExtraOpData(dim_args=[["dim"]]),
    "imag": ExtraOpData(is_view=True),
    "index_add": ExtraOpData(dim_args=[["dim"]]),
    "index_copy": ExtraOpData(dim_args=[["dim"]]),
    "index_fill": ExtraOpData(dim_args=[["dim"]]),
    "index_reduce.amax": ExtraOpData(dim_args=[["dim"]]),
    "index_reduce.amin": ExtraOpData(dim_args=[["dim"]]),
    "index_reduce.mean": ExtraOpData(dim_args=[["dim"]]),
    "index_reduce.prod": ExtraOpData(dim_args=[["dim"]]),
    "index_select": ExtraOpData(dim_args=[["dim"]]),
    "kthvalue": ExtraOpData(dim_args=[["dim"]]),
    "linalg.cross": ExtraOpData(dim_args=[["dim"]]),
    "linalg.diagonal": ExtraOpData(is_view=True, dim_args=[["dim1", "dim2"]]),
    "linalg.tensorsolve": ExtraOpData(dim_args=[["dims..."]]),
    "linalg.vecdot": ExtraOpData(dim_args=[["dim"]]),
    "linalg.vector_norm": ExtraOpData(dim_args=[["dim..."]]),
    "log_softmax": ExtraOpData(dim_args=[["dim"]]),
    "logcumsumexp": ExtraOpData(dim_args=[["dim"]]),
    "masked.amax": ExtraOpData(dim_args=[["dim"]]),
    "masked.amin": ExtraOpData(dim_args=[["dim"]]),
    "masked.argmax": ExtraOpData(dim_args=[["dim"]]),
    "masked.argmin": ExtraOpData(dim_args=[["dim"]]),
    "masked.logsumexp": ExtraOpData(dim_args=[["dim"]]),
    "masked.mean": ExtraOpData(dim_args=[["dim"]]),
    "masked.norm": ExtraOpData(dim_args=[["dim"]]),
    "masked.prod": ExtraOpData(dim_args=[["dim"]]),
    "masked.std": ExtraOpData(dim_args=[["dim"]]),
    "masked.sum": ExtraOpData(dim_args=[["dim"]]),
    "masked.var": ExtraOpData(dim_args=[["dim"]]),
    "max.reduction_with_dim": ExtraOpData(dim_args=[["dim"]]),
    "median": ExtraOpData(dim_args=[["dim"]]),
    "mean": ExtraOpData(dim_args=[["dim..."]]),
    "min.reduction_with_dim": ExtraOpData(dim_args=[["dim"]]),
    "mode": ExtraOpData(dim_args=[["dim"]]),
    "movedim": ExtraOpData(
        dim_args=[["source", "destination"], ["source...", "destination..."]]
    ),
    "nanmean": ExtraOpData(dim_args=[["dim..."]]),
    "nanmedian": ExtraOpData(dim_args=[["dim"]]),
    "nansum": ExtraOpData(dim_args=[["dim..."]]),
    "narrow": ExtraOpData(is_view=True, dim_args=[["dim"]]),
    "narrow_copy": ExtraOpData(dim_args=[["dim"]]),
    "nn.functional.cosine_similarity": ExtraOpData(dim_args=[["dim"]]),
    "nn.functional.glu": ExtraOpData(dim_args=[["dim"]]),
    "permute": ExtraOpData(is_view=True, dim_args=[["dims..."]]),
    "positive": ExtraOpData(is_view=True),
    "prod": ExtraOpData(dim_args=[["dim"]]),
    "ravel": ExtraOpData(is_view=True),
    "real": ExtraOpData(is_view=True),
    "renorm": ExtraOpData(dim_args=[["dim"]]),
    "reshape": ExtraOpData(is_view=True),
    "reshape_as": ExtraOpData(is_view=True),
    "roll": ExtraOpData(dim_args=[["dims..."]]),
    "rot90": ExtraOpData(dim_args=[["dims..."]]),
    "scatter": ExtraOpData(dim_args=[["dim"]]),
    "scatter_add": ExtraOpData(dim_args=[["dim"]]),
    "scatter_reduce.amax": ExtraOpData(dim_args=[["dim"]]),
    "scatter_reduce.amin": ExtraOpData(dim_args=[["dim"]]),
    "scatter_reduce.mean": ExtraOpData(dim_args=[["dim"]]),
    "scatter_reduce.prod": ExtraOpData(dim_args=[["dim"]]),
    "scatter_reduce.sum": ExtraOpData(dim_args=[["dim"]]),
    "select": ExtraOpData(is_view=True, dim_args=[["dim"]]),
    "select_scatter": ExtraOpData(dim_args=[["dim"]]),
    "slice": ExtraOpData(is_view=True, dim_args=[["dim"]]),
    "slice_scatter": ExtraOpData(dim_args=[["dim"]]),
    "softmax": ExtraOpData(dim_args=[["dim"]]),
    "sort": ExtraOpData(dim_args=[["dim"]]),
    "split": ExtraOpData(is_view=True, dim_args=[["dim"]]),
    "split_with_sizes": ExtraOpData(is_view=True, dim_args=[["dim"]]),
    "split_with_sizes_copy": ExtraOpData(dim_args=[["dim"]]),
    "squeeze": ExtraOpData(is_view=True, dim_args=[["dim"], ["dim..."]]),
    "squeeze_copy": ExtraOpData(dim_args=[["dim"], ["dim..."]]),
    "stack": ExtraOpData(dim_args=[["dim"]]),
    "std": ExtraOpData(dim_args=[["dim..."]]),
    "std.unbiased": ExtraOpData(dim_args=[["dim..."]]),
    "sum": ExtraOpData(dim_args=[["dim..."]]),
    "t": ExtraOpData(is_view=True),
    "tensor_split": ExtraOpData(is_view=True, dim_args=[["dim"]]),
    "tensordot": ExtraOpData(dim_args=[["dims..."]]),
    "tile": ExtraOpData(dim_args=[["dims..."]]),
    "topk": ExtraOpData(dim_args=[["dim"]]),
    "transpose": ExtraOpData(is_view=True, dim_args=[["dim0", "dim1"]]),
    "transpose_copy": ExtraOpData(dim_args=[["dim0", "dim1"]]),
    "trapezoid": ExtraOpData(dim_args=[["dim"]]),
    "trapz": ExtraOpData(dim_args=[["dim"]]),
    "unbind": ExtraOpData(is_view=True, dim_args=[["dim"]]),
    "unflatten": ExtraOpData(is_view=True, dim_args=[["dim"]]),
    "unfold": ExtraOpData(is_view=True, dim_args=[["dimension"]]),
    "unfold_copy": ExtraOpData(dim_args=[["dimension"]]),
    "unsafe_chunk": ExtraOpData(dim_args=[["dim"]]),
    "unsafe_split": ExtraOpData(dim_args=[["dim"]]),
    "unsqueeze": ExtraOpData(is_view=True, dim_args=[["dim"]]),
    "unsqueeze_copy": ExtraOpData(dim_args=[["dim"]]),
    "var": ExtraOpData(dim_args=[["dim..."]]),
    "var.unbiased": ExtraOpData(dim_args=[["dim..."]]),
    "view": ExtraOpData(is_view=True),
    "view_as": ExtraOpData(is_view=True),
    "view_as_complex": ExtraOpData(is_view=True),
    "view_as_real": ExtraOpData(is_view=True),
}


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


# Helper function to update a sample with new kwargs / name
def _update_sample(sample, new_kwargs):
    all_kwargs = dict(sample.kwargs)
    all_kwargs.update(new_kwargs)
    full_name = ", ".join([sample.name, *(f"{k}={v}" for (k, v) in new_kwargs.items())])
    return SampleInput(
        sample.input.clone().detach(),
        args=sample.args,
        kwargs=all_kwargs,
        name=full_name,
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


# Helper function to get a reasonable string representation of an NJT for use in
# SampleInput names.
def _describe_njt(njt) -> str:
    contig_type = "_contig" if njt.is_contiguous() else "_noncontig"
    if njt._lengths is not None and njt._offsets is not None:
        contig_type += "_holes"
    elif njt._ragged_idx != 1:
        contig_type += "_transposed"

    cached_data = "_without_seqlen_cache"
    if njt._max_seqlen_tensor is not None:
        cached_data = "_with_seqlen_cache"

    return f"{njt.dim()}D{contig_type}{cached_data}"


# Helper function to get a reasonable string representation of a given dim wrt an NJT.
def _describe_dim(njt, dim):
    if dim == 0:
        return "batch_dim"
    elif dim == njt._ragged_idx:
        return "ragged_dim"
    return "normal_dim"


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
        values = nt.values().detach().clone()
        offsets = nt.offsets().detach().clone()
        yield torch.nested.nested_tensor_from_jagged(values, offsets)

        # non-contiguous transposed NJT (not possible for 2D)
        if dim > 2:
            yield nt.transpose(-1, nt._ragged_idx)

        # non-contiguous with holes NJT
        values = nt.values().clone().detach()
        offsets = nt.offsets().clone().detach()
        # subtract 1 to cause holes
        lengths = (offsets.diff() - 1).clone().detach()
        yield torch.nested.nested_tensor_from_jagged(
            values=values,
            offsets=offsets,
            lengths=lengths,
        )


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

        # Handle indices in index_put
        if "index_put" in op.full_name and "indices" in kwargs:
            if len(kwargs["indices"]) > 1:
                # If after unrolling we still have indices left, use them
                kwargs["indices"] = [t[i] for t in kwargs["indices"][1:]]
            else:
                # If no indices are left, create them so they match the NJT implementation
                sequence_put = kwargs["indices"][0].tolist()
                if i in sequence_put:
                    kwargs["indices"] = [
                        torch.tensor(
                            list(range(inp.shape[0])),
                            dtype=torch.int32,
                            device=kwargs["indices"][0].device,
                        )
                    ]
                else:
                    kwargs["indices"] = [
                        torch.tensor(
                            [], dtype=torch.int32, device=kwargs["indices"][0].device
                        )
                    ]

        from torch.nested._internal.ops import _outer_to_inner_dim

        # Need to adjust dims to apply on NJT component
        if op._extra_op_data.dim_args is not None:
            # get all possible dim-related argnames that could be encountered for this op
            argnames = tree_map(
                lambda a: a.replace("...", ""),
                tree_flatten(op._extra_op_data.dim_args)[0],
            )
            # for all dim-related args present, convert from outer -> inner dim space
            for argname in {a for a in argnames if a in kwargs}:
                # allow the SampleInput to tell us how to canonicalize the dim kwargs
                ndim = nt_inp._ndim if hasattr(nt_inp, "_ndim") else nt_inp.dim()
                kwargs[argname] = _outer_to_inner_dim(
                    ndim, kwargs[argname], nt_inp._ragged_idx, canonicalize=True
                )

        out_ref_component = op.op(inp, *args, **kwargs)
        out_ref_components.append(out_ref_component)

    if wrap_output_as_njt:
        # handle list / tuple of outputs
        if len(out_ref_components) > 0 and isinstance(
            out_ref_components[0], (list, tuple)
        ):
            num_returns = len(out_ref_components[0])
            # ensure we get the same number of returns for each invocation
            assert all(len(o) == num_returns for o in out_ref_components)
            # construct NJTs from same index returns from each invocation
            njt_returns = [
                torch.nested.as_nested_tensor(
                    [o[r] for o in out_ref_components], layout=torch.jagged
                )
                for r in range(num_returns)
            ]
            return type(out_ref_components[0])(njt_returns)
        return torch.nested.as_nested_tensor(out_ref_components, layout=torch.jagged)

    return out_ref_components


# Computes the reference value for a non-reduction unary op with dim-wise application.
def unary_dimwise_reference(op, sample, batchwise_reference=None):
    # extract info about the dim args this op supports
    assert op._extra_op_data.dim_args is not None
    single_dim_argname, dimlist_argname = op._extra_op_data.get_dim_argnames()
    # only support a single non-list dim arg for now
    assert dimlist_argname is None
    assert single_dim_argname is not None
    if sample.kwargs[single_dim_argname] == 0:
        # unbind reference won't work for batch-wise operation; handle this case here
        assert batchwise_reference is not None
        return batchwise_reference(op, sample)
    return unbind_reference(op, sample)


# Computes the reference value for a reduction op.
def reduction_reference(op, sample):
    assert sample.input.is_nested

    # extract info about the dim args this op supports
    assert op._extra_op_data.dim_args is not None
    single_dim_argname, dimlist_argname = op._extra_op_data.get_dim_argnames()
    assert single_dim_argname is not None
    supports_dimlist = dimlist_argname is not None

    dim = sample.kwargs.get(
        dimlist_argname, sample.kwargs.get(single_dim_argname, None)
    )
    keepdim = sample.kwargs.get("keepdim", False)
    assert dim != 0, "reductions over just the batch dim are not supported"
    if isinstance(dim, (tuple, list)):
        reduce_on_ragged = sample.input._ragged_idx in dim
        reduce_on_batch = 0 in dim
    else:
        reduce_on_ragged = sample.input._ragged_idx == dim
        reduce_on_batch = dim == 0

    if dim is None:
        # calculate reference value by running reduction on values buffer
        return op.op(sample.input.values(), *sample.args, **sample.kwargs)

    if reduce_on_ragged and reduce_on_batch:
        # run reference directly on buffer with dims converted to inner space
        from torch.nested._internal.ops import _outer_to_inner_dim

        ref_kwargs = dict(sample.kwargs)
        assert dimlist_argname is not None
        ref_kwargs[dimlist_argname] = _outer_to_inner_dim(
            sample.input.dim(), dim, sample.input._ragged_idx, canonicalize=True
        )
        out = op.op(sample.input.values(), *sample.args, **ref_kwargs)
        if keepdim:
            if isinstance(out, (tuple, list)):
                # some ops return multiple things; unsqueeze all of them
                out = type(out)(o.unsqueeze(0) for o in out)
            else:
                out = out.unsqueeze(0)
        return out

    if reduce_on_ragged and not reduce_on_batch:
        # calculate reference value by running an unbind reference and stacking
        out_ref_components = unbind_reference(op, sample, wrap_output_as_njt=False)
        if len(out_ref_components) > 0 and isinstance(
            out_ref_components[0], (tuple, list)
        ):
            # some ops return multiple things; stack all of them
            num_returns = len(out_ref_components[0])
            # ensure we get the same number of returns for each invocation
            assert all(len(o) == num_returns for o in out_ref_components)
            # stack same index returns from each invocation
            stacked_returns = [
                torch.stack([o[r] for o in out_ref_components], dim=0)
                for r in range(num_returns)
            ]
            return type(out_ref_components[0])(stacked_returns)
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
        yield SampleInput(njt, kwargs=dict(op_kwargs), name=_describe_njt(njt))


def sample_inputs_elementwise_njt_binary(
    op_info, device, dtype, requires_grad, op_kwargs=None, **kwargs
):
    if not op_kwargs:
        op_kwargs = {}

    for njt1 in _sample_njts(
        device=device, dtype=dtype, requires_grad=requires_grad, dims=[2, 3, 4]
    ):
        njt_desc = _describe_njt(njt1)
        njt2 = torch.randn_like(njt1)
        yield SampleInput(
            njt1.clone().detach(),
            args=(njt2,),
            kwargs=dict(op_kwargs),
            name=f"{njt_desc}: (NT, NT)",
        )

        # broadcasting case: (B, j0, ...) with (B, 1, ...)
        dense_shape = list(njt1.shape)
        dense_shape[njt1._ragged_idx] = 1
        t = torch.randn(
            dense_shape,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        t2 = t.clone().detach()
        # used for slicing in unbind_reference()
        t._batch_dim = 0
        t2._batch_dim = 0
        # (NT, T)
        yield SampleInput(
            njt1.clone().detach(),
            args=(t,),
            kwargs=dict(op_kwargs),
            name=f"{njt_desc}: (NT, T) broadcasting 1 over ragged",
        )
        # (T, NT)
        yield SampleInput(
            t2,
            args=(njt1.clone().detach(),),
            kwargs=dict(op_kwargs),
            name=f"{njt_desc}: (T, NT) broadcasting 1 over ragged",
        )

        # broadcasting case: (B, j0, ...) with (1, 1...)
        t = torch.randn(
            [1 for _ in range(njt1.dim())],
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        t2 = t.clone().detach()
        # used for slicing in unbind_reference()
        t._batch_dim = 0
        t2._batch_dim = 0
        # (NT, T)
        yield SampleInput(
            njt1.clone().detach(),
            args=(t,),
            kwargs=dict(op_kwargs),
            name=f"{njt_desc}: (NT, T) broadcasting all 1s",
        )
        # (T, NT)
        yield SampleInput(
            t2,
            args=(njt1.clone().detach(),),
            kwargs=dict(op_kwargs),
            name=f"{njt_desc}: (T, NT) broadcasting all 1s",
        )

        # broadcasting case: (B, j0, ...) with (...)
        if njt1.dim() > njt1._ragged_idx + 1:
            t = torch.randn(
                njt1.shape[njt1._ragged_idx + 1 :],
                device=device,
                dtype=dtype,
                requires_grad=requires_grad,
            )
            # (NT, T)
            yield SampleInput(
                njt1.clone().detach(),
                args=(t.clone().detach(),),
                kwargs=dict(op_kwargs),
                name=f"{njt_desc}: (NT, T) broadcasting normal dims",
            )
            # (T, NT)
            yield SampleInput(
                t.clone().detach(),
                args=(njt1.clone().detach(),),
                kwargs=dict(op_kwargs),
                name=f"{njt_desc}: (T, NT) broadcasting normal dims",
            )

        # broadcasting case: (B, j0, ...) with scalar
        t = torch.randn((), device=device, dtype=dtype, requires_grad=requires_grad)
        # (NT, T)
        yield SampleInput(
            njt1.clone().detach(),
            args=(t.clone().detach(),),
            kwargs=dict(op_kwargs),
            name=f"{njt_desc}: (NT, T) broadcasting with scalar",
        )
        # (T, NT)
        yield SampleInput(
            t.clone().detach(),
            args=(njt1.clone().detach(),),
            kwargs=dict(op_kwargs),
            name=f"{njt_desc}: (T, NT) broadcasting with scalar",
        )

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
    njt_desc = _describe_njt(njt)
    t = torch.randn(B, 1, D, device=device, dtype=dtype, requires_grad=requires_grad)
    t2 = t.clone().detach()
    # used for slicing in unbind_reference()
    t._batch_dim = 0
    t2._batch_dim = 0

    # (NT, T)
    yield SampleInput(
        njt.clone().detach(),
        args=(t,),
        kwargs=dict(op_kwargs),
        name=f"{njt_desc}: (NT, T) mixed broadcasting",
    )
    # (T, NT)
    yield SampleInput(
        t2,
        args=(njt.clone().detach(),),
        kwargs=dict(op_kwargs),
        name=f"{njt_desc}: (T, NT) mixed broadcasting",
    )


def sample_inputs_njt_reduction(
    op_info,
    device,
    dtype,
    requires_grad,
    supports_keepdim=True,
    op_kwargs=None,
    **kwargs,
):
    if not op_kwargs:
        op_kwargs = {}

    # extract info about the dim args this op supports
    assert op_info._extra_op_data.dim_args is not None
    (
        single_dim_argname,
        dimlist_argname,
    ) = op_info._extra_op_data.get_dim_argnames()
    assert single_dim_argname is not None
    supports_dimlist = dimlist_argname is not None

    for njt in _sample_njts(
        device=device, dtype=dtype, requires_grad=requires_grad, dims=[2, 3, 4]
    ):
        njt_desc = _describe_njt(njt)
        keepdim_values = [False, True] if supports_keepdim else [None]
        for keepdim in keepdim_values:
            keepdim_suffix = f" with keepdim={keepdim}" if supports_keepdim else ""
            # single dim-wise reduction; includes reduction over the ragged dim
            # NB: reduction over the batch dim is not supported!
            # TODO: Cover this in the set of error inputs
            for dim in range(1, njt.dim()):
                dim_desc = "normal" if dim != njt._ragged_idx else "ragged"
                yield SampleInput(
                    njt.detach().clone(),
                    kwargs={
                        **op_kwargs,
                        single_dim_argname: dim,
                        **({"keepdim": keepdim} if supports_keepdim else {}),
                    },
                    name=f"{njt_desc}: {dim_desc} dim reduction{keepdim_suffix}",
                )

            if supports_dimlist:
                # reduce on both batch and ragged dims
                yield SampleInput(
                    njt.detach().clone(),
                    kwargs={
                        **op_kwargs,
                        dimlist_argname: [0, njt._ragged_idx],
                        **({"keepdim": keepdim} if supports_keepdim else {}),
                    },
                    name=f"{njt_desc}: batch+ragged reduction{keepdim_suffix}",
                )

                # reduce on batch, ragged, and other dims
                for other_dim in range(njt._ragged_idx + 1, njt.dim()):
                    yield SampleInput(
                        njt.detach().clone(),
                        kwargs={
                            **op_kwargs,
                            dimlist_argname: [0, njt._ragged_idx, other_dim],
                            **({"keepdim": keepdim} if supports_keepdim else {}),
                        },
                        name=(
                            f"{njt_desc}: batch+ragged+dim={other_dim} "
                            f"reduction{keepdim_suffix}"
                        ),
                    )

                # reduce on two non-ragged, non-batch dims
                if njt.dim() > 3 and njt._ragged_idx == 1:
                    yield SampleInput(
                        njt.detach().clone(),
                        kwargs={
                            **op_kwargs,
                            dimlist_argname: [njt.dim() - 2, njt.dim() - 1],
                            **({"keepdim": keepdim} if supports_keepdim else {}),
                        },
                        name=f"{njt_desc}: two normal dim reduction{keepdim_suffix}",
                    )

                # full reduction by specifying all dims
                yield SampleInput(
                    njt.detach().clone(),
                    kwargs={
                        **op_kwargs,
                        dimlist_argname: list(range(njt.dim())),
                        **({"keepdim": keepdim} if supports_keepdim else {}),
                    },
                    name=f"{njt_desc}: all dim reduction{keepdim_suffix}",
                )

                # TODO: Reducing on ragged dim and non-batch dim is not supported;
                # cover this in the set of error inputs.

        # full reduction
        yield SampleInput(
            njt.detach().clone(),
            kwargs=dict(op_kwargs),
            name=f"{njt_desc}: full reduction with keepdim={keepdim}",
        )


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


# === BEGIN OP-SPECIFIC SAMPLE INPUTS FUNCS / REFERENCES ===
def sample_inputs_unary_dimwise(
    op_info, device, dtype, requires_grad, op_kwargs=None, **kwargs
):
    if op_kwargs is None:
        op_kwargs = {}

    # only support a single non-list dim arg for now
    assert op_info._extra_op_data is not None
    single_dim_argname, dimlist_argname = op_info._extra_op_data.get_dim_argnames()
    assert single_dim_argname is not None
    assert dimlist_argname is None

    for njt in _sample_njts(
        device=device, dtype=dtype, requires_grad=requires_grad, dims=[2, 3, 4]
    ):
        for dim in range(njt.dim()):
            kwargs = {single_dim_argname: dim}
            kwargs.update(op_kwargs)
            yield SampleInput(
                njt.clone().detach(),
                kwargs=kwargs,
                name=f"{_describe_njt(njt)}: {_describe_dim(njt, dim)}",
            )


def batchwise_reference_chunk(op, sample):
    # reference for chunk() over dim=0
    kwargs = sample.kwargs
    B = sample.input.size(0)
    num_chunks = sample.kwargs["chunks"]
    chunk_size = math.ceil(B / num_chunks)
    num_full_chunks = B // chunk_size
    chunk_sizes = [chunk_size for _ in range(num_full_chunks)]
    if B % chunk_size != 0:
        # final chunk contains the leftovers
        chunk_sizes.append(B % chunk_size)

    # split unbound components into chunks according to calculated sizes
    components = list(sample.input.unbind())
    start = 0
    chunks = []
    for chunk_size in chunk_sizes:
        chunks.append(components[start : start + chunk_size])
        start += chunk_size

    # rejoin into NJT outputs
    return [torch.nested.nested_tensor(lst, layout=torch.jagged) for lst in chunks]


def batchwise_reference_narrow(op, sample):
    # TODO: write this!
    raise NotImplementedError


def batchwise_reference_select(op, sample):
    # reference for select() over dim=0
    return sample.input.unbind()[sample.kwargs["index"]]


def batchwise_reference_split(op, sample):
    # TODO: write this!
    raise NotImplementedError


def batchwise_reference_split_with_sizes(op, sample):
    # TODO: write this!
    raise NotImplementedError


def batchwise_reference_unflatten(op, sample):
    # TODO: write this!
    raise NotImplementedError


def batchwise_reference_unsqueeze(op, sample):
    raise ValueError("unsqueeze() is not intended to operate on the batch dim")


def sample_inputs_clone(op_info, device, dtype, requires_grad, **kwargs):
    # non-contiguous NJTs
    for njt in _sample_njts(
        device=device, dtype=dtype, requires_grad=requires_grad, dims=[2, 3, 4]
    ):
        yield SampleInput(njt, name=_describe_njt(njt))

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

        njt_desc = _describe_njt(njt)
        yield SampleInput(
            njt,
            kwargs={"memory_format": memory_format},
            name=f"{njt_desc}: {memory_format})",
        )


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
                njt.detach().clone(), kwargs={"dtype": dtype}, name=sample_name
            )

        # only include device transfer for CUDA inputs
        if "cuda" in device:
            other_device = "cpu"
            sample_name = f"{njt.dim()}D: {device} -> {other_device}"
            yield SampleInput(
                njt.detach().clone(), kwargs={"device": other_device}, name=sample_name
            )


def sample_inputs_bmm(op_info, device, dtype, requires_grad, op_kwargs=None, **kwargs):
    for njt_3d in _sample_njts(
        device=device, dtype=dtype, requires_grad=requires_grad, dims=[3]
    ):
        # (B, j1, D) x (B, D, E) => (B, j1, E)
        if njt_3d._ragged_idx == 1:
            B, D = njt_3d.shape[0], njt_3d.shape[-1]
            E = D + 2
            other = torch.randn(B, D, E, device=device, dtype=dtype)
            # used for slicing in unbind_reference()
            other._batch_dim = 0
            njt_desc = _describe_njt(njt_3d)
            yield SampleInput(
                njt_3d.detach().clone(),
                kwargs={"mat2": other},
                name=f"{njt_desc}: (B, j, D) x (B, D, E)",
            )

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


def sample_inputs_chunk(op_info, device, dtype, requires_grad, **kwargs):
    for sample_input in sample_inputs_unary_dimwise(
        op_info, device, dtype, requires_grad, **kwargs
    ):
        # ragged dim chunking: test a single chunks value
        if sample_input.kwargs["dim"] == sample_input.input._ragged_idx:
            yield _update_sample(sample_input, {"chunks": 3})
        # other dim chunking: test different chunks values
        else:
            D = sample_input.input.size(sample_input.kwargs["dim"])
            for chunks in [1, D // 2, D - 1, D]:
                yield _update_sample(sample_input, {"chunks": chunks})


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
        if njt_3d._ragged_idx == 1:
            D = njt_3d.shape[-1]
            E = D + 2
            njt_desc = _describe_njt(njt_3d)
            yield SampleInput(
                njt_3d.detach().clone(),
                kwargs={"other": torch.randn(D, E, device=device, dtype=dtype)},
                name=f"{njt_desc}: (B, j, D) x (D, E)",
            )

    # 4D cases
    for njt_4d in _sample_njts(
        device=device, dtype=dtype, requires_grad=requires_grad, dims=[4]
    ):
        # (B, j1, D, E) x (E, F) => (B, j1, D, F)
        if njt_4d._ragged_idx == 1:
            E = njt_4d.shape[-1]
            F = E + 2
            njt_desc = _describe_njt(njt_4d)
            yield SampleInput(
                njt_4d.detach().clone(),
                kwargs={"other": torch.randn(E, F, device=device, dtype=dtype)},
                name=f"{njt_desc}: (B, j, D, E) x (E, F)",
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
            njt,
            kwargs={"mask": (torch.randn_like(njt, requires_grad=False) < 0.0)},
            name=_describe_njt(njt),
        )


def sample_inputs_narrow(op_info, device, dtype, requires_grad, **kwargs):
    for sample_input in sample_inputs_unary_dimwise(
        op_info, device, dtype, requires_grad, **kwargs
    ):
        # ragged dim narrowing: test a single start, length value
        if sample_input.kwargs["dim"] == sample_input.input._ragged_idx:
            yield _update_sample(sample_input, {"start": 1, "length": 2})
        # other dim narrowing: test different start, length values
        else:
            D = sample_input.input.size(sample_input.kwargs["dim"])
            for start, length in [(0, D), (0, D - 1), (1, D - 1), (D - 1, 1)]:
                yield _update_sample(sample_input, {"start": start, "length": length})


def sample_inputs_nn_functional_embedding(
    op_info, device, dtype, requires_grad, **kwargs
):
    indices = torch.nested.nested_tensor(
        [
            torch.tensor([0, 2, 1, 3]),
            torch.tensor([4, 2, 1]),
            torch.tensor([6, 7, 5, 2, 4]),
        ],
        layout=torch.jagged,
        dtype=torch.int64,
        device=device,
    )

    NUM_EMBEDDINGS = 20
    EMBEDDING_DIM = 32
    weight = torch.randn(NUM_EMBEDDINGS, EMBEDDING_DIM, device=device, dtype=dtype)

    # NB: the OpInfo entry for embedding_bag expects weight first so the gradients
    # can be checked
    yield SampleInput(
        weight.detach().clone().requires_grad_(),
        args=(indices,),
    )

    yield SampleInput(
        weight.detach().clone().requires_grad_(),
        args=(indices,),
        kwargs={"padding_idx": 1},
    )


def sample_inputs_index_put(
    op_info, device, dtype, requires_grad, op_kwargs=None, **kwargs
):
    for njt in _sample_njts(
        device=device, dtype=dtype, requires_grad=requires_grad, dims=[2, 3, 4]
    ):
        for dim in range(njt.dim()):
            indices = [
                torch.tensor(list(range(njt.size(0))), device=njt.device),
                *[
                    torch.tensor([0] * njt.size(0), device=njt.device)
                    for _ in range(dim - 1)
                ],
            ]
            njt_desc = _describe_njt(njt)
            yield SampleInput(
                njt.detach().clone(),
                kwargs={
                    "indices": indices,
                    "values": torch.tensor(1.0, device=njt.device),
                },
                name=f"{njt_desc}: up to dim {dim - 1}",
            )

    # Non-cont NJT for completeness
    offsets = torch.tensor([0, 2, 5, 7], device=device)
    lengths = torch.tensor([2, 2, 2], device=device)
    indices = [
        torch.tensor([0, 1, 2], device=device),
        torch.tensor([0, 1, 1], device=device),
        torch.tensor([0, 0, 0], device=device),
    ]
    a = torch.nested.nested_tensor_from_jagged(
        torch.zeros(7, 3, device=device), offsets, lengths
    )

    njt_desc = _describe_njt(a)
    yield SampleInput(
        a.detach().clone(),
        kwargs={"indices": indices, "values": torch.tensor(1.0, device=a.device)},
        name=f"{njt_desc}: all dims",
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
        # projection over a ragged dim is not currently supported
        if is_nested_int(njt.size(-1)):
            continue

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
        for start_dim in range(njt.dim()):
            if start_dim <= njt._ragged_idx:
                continue

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


def sample_inputs_select(op_info, device, dtype, requires_grad, **kwargs):
    for sample_input in sample_inputs_unary_dimwise(
        op_info, device, dtype, requires_grad, **kwargs
    ):
        # ragged dim chunking: test a single index
        if sample_input.kwargs["dim"] == sample_input.input._ragged_idx:
            yield _update_sample(sample_input, {"index": 0})
        # other dim chunking: test different indices
        else:
            D = sample_input.input.size(sample_input.kwargs["dim"])
            for index in [0, D // 2, D - 1]:
                yield _update_sample(sample_input, {"index": index})


def sample_inputs_split(op_info, device, dtype, requires_grad, **kwargs):
    for sample_input in sample_inputs_unary_dimwise(
        op_info, device, dtype, requires_grad, **kwargs
    ):
        # ragged dim chunking: test a single split size
        if sample_input.kwargs["dim"] == sample_input.input._ragged_idx:
            yield _update_sample(sample_input, {"split_size_or_sections": 3})
        # other dim chunking: test different split sizes
        else:
            D = sample_input.input.size(sample_input.kwargs["dim"])
            for split_size in [1, D // 2, D - 1, D]:
                yield _update_sample(
                    sample_input, {"split_size_or_sections": split_size}
                )


def sample_inputs_split_with_sizes(op_info, device, dtype, requires_grad, **kwargs):
    for sample_input in sample_inputs_unary_dimwise(
        op_info, device, dtype, requires_grad, **kwargs
    ):
        # It will never make sense to operate on the ragged dim.
        # TODO: Handle this with error_inputs
        if sample_input.kwargs["dim"] == sample_input.input._ragged_idx:
            continue

        D = sample_input.input.size(sample_input.kwargs["dim"])
        # splits should add up to D
        split1 = torch.randint(0, D - 1, size=()).item()
        split2 = D - split1
        yield _update_sample(sample_input, {"split_sizes": [split1, split2]})


def sample_inputs_squeeze(op_info, device, dtype, requires_grad, **kwargs):
    # squeeze-specific NJT generator (need to ensure there are some 1s in the shape)
    def _get_njts():
        njt = random_nt_from_dims(
            (4, None, 1, 3, 1),
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
            layout=torch.jagged,
        )
        yield njt
        # without min / max seqlen cached
        values = njt.values().detach().clone()
        offsets = njt.offsets().detach().clone()
        yield torch.nested.nested_tensor_from_jagged(values, offsets)
        # non-contiguous transposed
        yield njt.transpose(1, 3)
        # non-contiguous with holes
        values = njt.values().clone().detach()
        offsets = njt.offsets().clone().detach()
        # subtract 1 to cause holes
        lengths = (offsets.diff() - 1).clone().detach()
        yield torch.nested.nested_tensor_from_jagged(
            values=values,
            offsets=offsets,
            lengths=lengths,
        )

    for njt in _get_njts():
        # single dim operation
        for dim in range(njt.dim()):
            # Operation on batch / ragged dim is never expected to work.
            # TODO: Handle these via error_inputs.
            if dim == 0 or dim == njt._ragged_idx:
                continue

            yield SampleInput(
                njt.clone().detach(),
                kwargs={"dim": dim},
                name=f"{_describe_njt(njt)}: {_describe_dim(njt, dim)}",
            )

        # multiple dim operation (pass no args)
        yield SampleInput(
            njt.clone().detach(),
            kwargs={"dim": dim},
            name=f"{_describe_njt(njt)}: multiple dims",
        )


def sample_inputs_unflatten(op_info, device, dtype, requires_grad, **kwargs):
    for sample_input in sample_inputs_unary_dimwise(
        op_info, device, dtype, requires_grad, **kwargs
    ):
        # It will never make sense to operate on the ragged dim.
        # TODO: Handle this with error_inputs
        if sample_input.kwargs["dim"] == sample_input.input._ragged_idx:
            continue

        D = sample_input.input.size(sample_input.kwargs["dim"])
        # sizes should multiply to be D
        yield _update_sample(sample_input, {"sizes": [D, 1]})
        yield _update_sample(sample_input, {"sizes": [1, D]})
        if D % 2 == 0:
            yield _update_sample(sample_input, {"sizes": [D // 2, 2]})
            yield _update_sample(sample_input, {"sizes": [2, D // 2]})


def sample_inputs_unsqueeze(op_info, device, dtype, requires_grad, **kwargs):
    for sample_input in sample_inputs_unary_dimwise(
        op_info, device, dtype, requires_grad, **kwargs
    ):
        yield sample_input

        last_dim_sample = _update_sample(sample_input, {"dim": -1})
        last_dim_sample.name = (
            f"{_describe_njt(last_dim_sample.input)}: add dim to the end"
        )
        # Tell the unbind reference how to canonicalize the dim kwargs
        # This is necessary because unsqueeze() allows for a dim after
        # the last dim to indicate an unsqueeze at the end.
        last_dim_sample.input._ndim = last_dim_sample.input.dim() + 1
        yield last_dim_sample


def sample_inputs_where(op_info, device, dtype, requires_grad, **kwargs):
    for sample in sample_inputs_elementwise_njt_binary(
        op_info, device, dtype, requires_grad, **kwargs
    ):
        other = sample.args[0]
        sample.args = ()
        sample.kwargs["other"] = other
        sample.kwargs["condition"] = sample.input > 0.0
        sample.name = sample.name.replace("(", "(NT, ")
        yield sample


# === END OP-SPECIFIC SAMPLE INPUTS FUNCS / REFERENCES ===


# Mapping of OpInfo full names -> sample_inputs_funcs, which define the set of sample inputs
# (involving NJTs) to pass to the op. Full name consists of the OpInfo's name and variant name
# separated by a period (e.g. special.polygamma.special_polygamma_n_0). These are necessary
# to specify if they cannot be auto-generated for some reason. Try to keep these sorted
# in alphabetical order!
njt_sample_inputs = {
    "bmm": sample_inputs_bmm,
    "chunk": sample_inputs_chunk,
    "clone": sample_inputs_clone,
    "count_nonzero": partial(sample_inputs_njt_reduction, supports_keepdim=False),
    **{f"mvlgamma.mvlgamma_p_{p}": sample_inputs_mvl_gamma(p=1) for p in (1, 3, 5)},
    "nn.functional.embedding": sample_inputs_nn_functional_embedding,
    "nn.functional.embedding_bag": sample_inputs_nn_functional_embedding_bag,
    "nn.functional.linear": sample_inputs_nn_functional_linear,
    "nn.functional.rms_norm": sample_inputs_nn_functional_rms_norm,
    "nn.functional.threshold": sample_inputs_nn_functional_threshold,
    **{f"polygamma.polygamma_n_{n}": sample_inputs_polygamma_n(n=n) for n in range(5)},
    "special.polygamma.special_polygamma_n_0": sample_inputs_special_polygamma_n(n=0),
    "to": sample_inputs_to,
    "matmul": sample_inputs_matmul,
    "masked_select": sample_inputs_masked_select,
    "narrow": sample_inputs_narrow,
    "index_put": sample_inputs_index_put,
    # these two don't have ReductionOpInfo entries
    "max.reduction_with_dim": sample_inputs_njt_reduction,
    "min.reduction_with_dim": sample_inputs_njt_reduction,
    "select": sample_inputs_select,
    "split": sample_inputs_split,
    "split_with_sizes": sample_inputs_split_with_sizes,
    "squeeze": sample_inputs_squeeze,
    "unflatten": sample_inputs_unflatten,
    "unsqueeze": sample_inputs_unsqueeze,
    "where": sample_inputs_where,
}

njt_references = {
    "bmm": reference_bmm,
    "chunk": partial(
        unary_dimwise_reference, batchwise_reference=batchwise_reference_chunk
    ),
    "count_nonzero": reduction_reference,
    # these two don't have ReductionOpInfo entries
    "max.reduction_with_dim": reduction_reference,
    "min.reduction_with_dim": reduction_reference,
    "narrow": partial(
        unary_dimwise_reference, batchwise_reference=batchwise_reference_narrow
    ),
    "select": partial(
        unary_dimwise_reference, batchwise_reference=batchwise_reference_select
    ),
    "split": partial(
        unary_dimwise_reference, batchwise_reference=batchwise_reference_split
    ),
    "split_with_sizes": partial(
        unary_dimwise_reference,
        batchwise_reference=batchwise_reference_split_with_sizes,
    ),
    "squeeze": unbind_reference,
    "nn.functional.embedding_bag": reference_nn_functional_embedding_bag,
    "unflatten": partial(
        unary_dimwise_reference, batchwise_reference=batchwise_reference_unflatten
    ),
    "unsqueeze": partial(
        unary_dimwise_reference, batchwise_reference=batchwise_reference_unsqueeze
    ),
}


# Translates an OpInfo entry to one that operates on NJTs.
def translate_opinfo(op):
    new_op = copy(op)
    new_op.supports_njt = True
    # add some extra info for use in generating tests on the right subset of ops
    new_op._extra_op_data = extra_op_data.get(op.full_name, ExtraOpData())

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
