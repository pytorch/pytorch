# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
from typing import List, Optional, Tuple, Union

import sympy

import torch

from .ir import Pointwise, TensorBox
from .lowering import fallback_handler, is_integer_type, register_lowering
from .virtualized import ops


# pyre-ignore[2,3]
def dense_idx_to_jagged_idx(batch_idx, seq_idx, offsets_loader, jagged_len):
    # jagged_len + 1 is used as the upper bound,
    # because the last sequence length may be zero
    begin_idx = ops.indirect_indexing(
        offsets_loader([batch_idx]),
        jagged_len + 1,
    )
    end_idx = offsets_loader([batch_idx + 1])
    jagged_idx = begin_idx + seq_idx
    return jagged_idx, end_idx


def get_inverse_offsets(
    offsets: TensorBox,
    jagged_len: Union[int, sympy.Expr],
    realize: bool = True,
) -> TensorBox:
    """
    Returns "inverse_offsets" - the inverse of the offsets array.
    offsets maps batch index (dense) to jagged index (i.e. offset into jagged tensor).
    inverse_offsets maps jagged index to batch index.

    e.g. for offsets [0, 3, 4, 9, 10] this will return
    inverse_offsets = [0, 0, 0, 1, 2, 2, 2, 2, 2, 3]

    For the given offsets, the computed inverse_offsets are cached
    on the first call and reused in the further calls.
    """

    if hasattr(offsets, "inverse_offsets"):
        # inverse_offsets are already computed
        # for these offsets: can reuse
        return offsets.inverse_offsets

    # ops.bucketize takes offsets.get_name() which doesn't exist on Pointwise
    # kernels, i.e. we need to realize it before using. In other words, we need
    # offsets to be in global memory so that we can binary search over the
    # entire tensor
    offsets.realize()
    device: torch.device = offsets.get_device()
    dtype: torch.dtype = offsets.get_dtype()

    # pyre-ignore[2,3]
    def inner_fn(index):
        idx = index[0]
        bucket = ops.bucketize(
            values=ops.index_expr(idx, dtype),
            offsets_name=offsets.get_name(),
            offsets_size=offsets.get_size()[0],
            indexing_dtype=dtype,
            right=True,
        )
        # ops.bucketize above returns 1-based bucket indices,
        # but we need 0-based, hence we subtract 1 from batch
        return bucket - 1

    inverse_offsets = Pointwise.create(
        device=device,
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=[jagged_len],
    )

    if realize:
        # "freeze" the node so that it doesn't get inlined downstream.
        inverse_offsets.realize()

    # cache inverse_offsets for further reuse
    offsets.inverse_offsets = inverse_offsets  # type: ignore[attr-defined]

    return inverse_offsets


def jagged_idx_to_dense_idx(
    jagged_idx,  # pyre-ignore[2]
    inverse_offsets_loader,  # pyre-ignore[2]
    offsets_loader,  # pyre-ignore[2]
    batch_size: Union[int, sympy.Expr],
    max_seq_len: Union[int, sympy.Expr],
    offsets_dtype: torch.dtype,
) -> Tuple[sympy.Expr, sympy.Expr]:
    batch_idx = ops.indirect_indexing(
        inverse_offsets_loader([jagged_idx]),
        batch_size + 1,
    )
    batch_start = offsets_loader([batch_idx])
    seq = ops.index_expr(jagged_idx, offsets_dtype) - batch_start
    # check=False because there may be sequences longer than max_seq_len
    seq_idx = ops.indirect_indexing(seq, max_seq_len, check=False)
    return batch_idx, seq_idx


def register_jagged_ops():
    # pyre-ignore[56]
    @register_lowering(torch.ops.aten._jagged_to_padded_dense_forward.default)
    def _jagged_to_padded_dense_forward(
        jagged_values: TensorBox,
        jagged_offsets: List[TensorBox],
        max_lengths: List[int],  # list of ints/SymInts
        padding_value: float = 0.0,
    ) -> TensorBox:
        device = jagged_values.get_device()
        dtype = jagged_values.get_dtype()

        jagged_values_size = jagged_values.get_size()

        # only handle the common case of a single jagged dimension
        if (
            len(jagged_offsets) != 1
            or device.type != "cuda"
            or device != jagged_offsets[0].get_device()
            or len(jagged_values_size) != 2
            or len(jagged_offsets[0].get_size()) != 1
            or len(max_lengths) != len(jagged_offsets)
            or not is_integer_type(jagged_offsets[0])
        ):
            return fallback_handler(
                torch.ops.aten._jagged_to_padded_dense_forward.default,
                add_to_fallback_set=False,
            )(
                jagged_values,
                jagged_offsets,
                max_lengths,
                padding_value,
            )

        offsets: TensorBox = jagged_offsets[0]
        offsets_len = offsets.get_size()[0]
        offsets_dtype = offsets.get_dtype()
        batch_size = offsets_len - 1
        max_seq_len = max_lengths[0]
        embedding_len = jagged_values_size[1]
        jagged_len = jagged_values_size[0]

        output_size = [batch_size, max_seq_len, embedding_len]

        values_loader = jagged_values.make_loader()
        offsets_loader = offsets.make_loader()

        # pyre-ignore[2,3,53]
        def inner_fn(index):
            # dense tensor size: [B, N, D]
            batch_idx, seq_idx, emb_idx = index
            jagged_idx, end_idx = dense_idx_to_jagged_idx(
                batch_idx=batch_idx,
                seq_idx=seq_idx,
                offsets_loader=offsets_loader,
                jagged_len=jagged_len,
            )
            return ops.masked(
                ops.lt(
                    ops.index_expr(jagged_idx, offsets_dtype),
                    end_idx,
                ),
                lambda: values_loader([jagged_idx, emb_idx]),
                padding_value,
            )

        return Pointwise.create(
            device=device,
            dtype=dtype,
            inner_fn=inner_fn,
            ranges=output_size,
        )

    def _dense_to_jagged_forward_impl(
        fallback_op,  # pyre-ignore[2]
        dense: TensorBox,
        jagged_offsets: List[TensorBox],
        jagged_len: Optional[int] = None,
    ) -> TensorBox:
        device = dense.get_device()
        dtype = dense.get_dtype()

        dense_size = dense.get_size()

        # only handle the common case of a single jagged dimension
        if (
            len(jagged_offsets) != 1
            or device.type != "cuda"
            or device != jagged_offsets[0].get_device()
            or len(jagged_offsets[0].get_size()) != 1
            or len(dense_size) != 3
            or jagged_len is None
            or not is_integer_type(jagged_offsets[0])
        ):
            return fallback_handler(fallback_op, add_to_fallback_set=False)(
                dense,
                jagged_offsets,
                jagged_len,
            )

        offsets: TensorBox = jagged_offsets[0]
        offsets_dtype = offsets.get_dtype()
        batch_size = dense_size[0]
        max_seq_len = dense_size[1]
        embedding_len = dense_size[-1]

        output_size = [jagged_len, embedding_len]

        dense_loader = dense.make_loader()
        offsets_loader = offsets.make_loader()

        inverse_offsets = get_inverse_offsets(
            offsets=offsets,
            jagged_len=jagged_len,
        )
        inverse_offsets_loader = inverse_offsets.make_loader()

        # pyre-ignore[2,3,53]
        def inner_fn(index):
            # jagged tensor size: [sum_B(N_B), D]
            jagged_idx, emb_idx = index
            batch_idx, seq_idx = jagged_idx_to_dense_idx(
                jagged_idx=jagged_idx,
                offsets_loader=offsets_loader,
                inverse_offsets_loader=inverse_offsets_loader,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                offsets_dtype=offsets_dtype,
            )
            return ops.masked(
                ops.lt(
                    ops.index_expr(seq_idx, offsets_dtype),
                    ops.index_expr(max_seq_len, offsets_dtype),
                ),
                lambda: dense_loader([batch_idx, seq_idx, emb_idx]),
                0.0,  # jagged sequence longer than max_seq_len
            )

        return Pointwise.create(
            device=device,
            dtype=dtype,
            inner_fn=inner_fn,
            ranges=output_size,
        )

    # pyre-ignore[56]
    @register_lowering(torch.ops.aten._padded_dense_to_jagged_forward)
    def _dense_to_jagged_forward(
        dense: TensorBox,
        jagged_offsets: List[TensorBox],
        jagged_len: Optional[int] = None,
    ) -> TensorBox:
        return _dense_to_jagged_forward_impl(
            fallback_op=torch.ops.aten._padded_dense_to_jagged_forward.default,
            dense=dense,
            jagged_offsets=jagged_offsets,
            jagged_len=jagged_len,
        )
