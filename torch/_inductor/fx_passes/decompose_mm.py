import sympy

import torch


def _fn(a: torch.Tensor, b: torch.Tensor, num_chunks: int, orig_out: torch.Tensor):
    a_flat = a.flatten(0, -2)
    a_flat_chunks = a_flat.chunk(num_chunks)
    out = torch.empty_strided(
        size=orig_out.shape,
        stride=orig_out.stride(),
        dtype=orig_out.dtype,
        device=orig_out.device,
    )
    out_flat = out.flatten(0, -2)
    out_flat_chunks = out_flat.chunk(num_chunks)
    for i in range(num_chunks):
        torch.ops.aten.mm.out(a_flat_chunks[i], b, out=out_flat_chunks[i])
    return (out,)


def split_mms(gm: torch.fx.GraphModule, min_m_size: int, num_chunks: int = 2):
    g = gm.graph

    def _size_hint(s: sympy.Expr) -> int:
        from torch.fx.experimental.symbolic_shapes import size_hint

        hint = size_hint(s)
        if hint is not None:
            return hint
        return 0

    def _is_contiguous(t) -> bool:
        return t.is_contiguous(memory_format=torch.contiguous_format)

    for n in g.nodes:
        if n.target != torch.ops.aten.mm.default:
            continue

        mm_n = n
        arg_a = mm_n.args[0]
        arg_b = mm_n.args[1]

        a_t = arg_a.meta["val"]

        M = 1
        for s in a_t.shape[:-1]:
            M *= _size_hint(s)

        if M < min_m_size:
            continue

        arg_a_t = arg_a.meta["val"]
        arg_b_t = arg_b.meta["val"]
        mm_out_t = mm_n.meta["val"]

        # Decompose only to contiguous chunks
        if not (_is_contiguous(arg_a_t) and _is_contiguous(mm_out_t)):
            continue

        from torch._inductor.fx_passes.bucketing import _insert_fn_trace_before_node

        trace_args = (arg_a_t, arg_b_t, num_chunks, mm_out_t)
        _insert_fn_trace_before_node(
            g,
            _fn,
            trace_args,
            mm_n,  # insert before
            [arg_a, arg_b],
            [mm_n],
        )

        g.erase_node(mm_n)
