"""
[not for land] Low-level demo: drive the CODA backward epilogue-fusion machinery
by hand, using the private plumbing directly instead of the front door
(FusibleFunction.apply / apply_epilogue_fusion).

This is exactly the mechanism the public API wraps, unrolled step by step so the
moving parts are visible:

  _StagingCtx        collects each node's saved tensors (and the main node's
                     intermediate metadata) during a single no_grad forward
  _MainNode          the "matmul" autograd node; its forward returns a PHANTOM
                     output shaped like the intermediate (GEMM output), carrying
                     the graph edge to the inputs and the saved set
  _EpilogueNode      the "pointwise" autograd node; attaches the real output to
                     the graph and, in backward, either runs the epilogue or the
                     fused kernel
  DeferredGradTensor a metadata-only placeholder grad that rides the existing
                     backward edge from the deferring main node to the epilogue,
                     carrying the producer's grad and main saved set (a plain tuple,
                     captured in the producer's backward) to the fused kernel

Run:
    python coda_backward_epilogue_fusion_lowlevel_demo.py
"""

import os
import sys


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import coda_backward_epilogue_fusion as coda  # noqa: F401
from coda_backward_epilogue_fusion import (  # noqa: F401
    _EpilogueNode,
    _is_epilogue,
    _is_main,
    _MainNode,
    _StagingCtx,
    DeferredGradTensor,  # imported for narration/reference
    FusibleFunction,
    LOG,
)

import torch


# ---------------------------------------------------------------------------
# User code: the op (a self-contained forward + its main/epilogue backwards) and
# the fused backward kernel. This is everything the user writes; everything
# imported above is framework plumbing. The framework only needs cls.forward /
# cls.main_backward / cls.epilogue_backward to exist.
# ---------------------------------------------------------------------------
class MMRelu(FusibleFunction):
    @staticmethod
    def forward(main_ctx, epilogue_ctx, x, w):
        a = x @ w
        main_ctx.save_for_backward(x, w)
        main_ctx.set_output_meta(a)  # REQUIRED: declares the intermediate metadata
        epilogue_ctx.save_for_backward(a)
        return torch.relu(a)

    @staticmethod
    def main_backward(main_ctx, grad_main_out):
        x, w = main_ctx.saved_tensors
        need_x, need_w = main_ctx.needs_input_grad[0], main_ctx.needs_input_grad[1]
        # dW is always local; dx (input 0) is guarded by needs_input_grad and
        # skipped when this op's activation grad is deferred into the fused kernel.
        grad_w = x.transpose(-1, -2) @ grad_main_out if need_w else None
        grad_x = grad_main_out @ w.transpose(-1, -2) if need_x else None
        return grad_x, grad_w

    @staticmethod
    def epilogue_backward(epilogue_ctx, grad_out):
        (a,) = epilogue_ctx.saved_tensors
        return grad_out * (a > 0).to(a.dtype)


def mm_relu_fused_backward(grad_producer_out, main_saved_tensors, consumer_ctx):
    # The producer main node's deferred grad_input GEMM with the consumer relu
    # epilogue fused on: dL/dh = (dL/da @ w^T) then * relu'(a).
    _x_p, w_p = main_saved_tensors
    (a_c,) = consumer_ctx.saved_tensors
    grad_main_input = grad_producer_out @ w_p.transpose(-1, -2)
    return grad_main_input * (a_c > 0).to(a_c.dtype)


def _edge_names(node):
    return [("None" if f is None else type(f).__name__) for f, _ in node.next_functions]


def _eager_relu_chain_grads(values):
    # Reference: grads of relu(...relu(x @ w1)... @ wn).sum() over fresh leaves.
    leaves = [v.detach().clone().requires_grad_(True) for v in values]
    acc = leaves[0]
    for w in leaves[1:]:
        acc = torch.relu(acc @ w)
    acc.sum().backward()
    return [leaf.grad for leaf in leaves]


def section_1_build_one_op_by_hand():
    print("=" * 72)
    print("SECTION 1: build one op (y = relu(x @ w)) by hand -- no fusion")
    print("=" * 72)
    torch.manual_seed(0)
    x = torch.randn(4, 6, dtype=torch.double, requires_grad=True)
    w = torch.randn(6, 5, dtype=torch.double, requires_grad=True)

    # STEP 1 -- run the op's forward ONCE under no_grad against two staging ctxs.
    # The forward saves into each (main_ctx gets x, w + the intermediate meta;
    # epilogue_ctx gets the preactivation a). Nothing is on the autograd graph yet.
    main_staging = _StagingCtx()
    epilogue_staging = _StagingCtx()
    with torch.no_grad():
        out = MMRelu.forward(main_staging, epilogue_staging, x, w)
    msaved = tuple(t.shape for t in main_staging.saved)
    esaved = tuple(t.shape for t in epilogue_staging.saved)
    meta_shape = tuple(main_staging.output_meta[0])
    print("\n[step 1] no_grad forward populated the staging ctxs:")
    print("  main_staging.saved    :", msaved, "(x, w)")
    print("  main_staging.meta     :", meta_shape, "(a = x @ w)")
    print("  epilogue_staging.saved:", esaved, "(a)")

    # STEP 2 -- the MAIN node. Pass cls, the declared intermediate meta, and the
    # main saved set explicitly (these are non-tensors / a nested tuple, so they
    # add no graph edges). Its forward returns a PHANTOM tensor shaped like the
    # intermediate: its value is unused, it exists only to own the grad_fn whose
    # next edges are (x, w).
    main_out = _MainNode.apply(
        MMRelu, main_staging.output_meta, main_staging.saved, x, w
    )
    edges = _edge_names(main_out.grad_fn)
    print("\n[step 2] _MainNode.apply -> phantom intermediate carrier:")
    print("  grad_fn   :", type(main_out.grad_fn).__name__)
    print("  next edges:", edges, "(only the tensor inputs x, w)")

    # STEP 3 -- the EPILOGUE node. main_out (phantom) requires grad, so a fresh
    # view of the no_grad `out` picks up THIS node's grad_fn and becomes the real
    # output. `out` rides inside a 1-tuple so it is not an autograd input.
    y = _EpilogueNode.apply(MMRelu, epilogue_staging.saved, (out,), main_out)
    print("\n[step 3] _EpilogueNode.apply -> real output attached to graph:")
    print("  grad_fn      :", type(y.grad_fn).__name__)
    print("  value matches:", torch.allclose(y, torch.relu(x @ w)))

    # BACKWARD with nothing armed: the epilogue node sees a REAL grad (not a
    # placeholder) and runs MMRelu.epilogue_backward; the main node runs the full
    # MMRelu.main_backward (both dx and dw).
    LOG.reset()
    y.sum().backward()
    ref_x, ref_w = _eager_relu_chain_grads([x, w])
    matches = torch.allclose(x.grad, ref_x) and torch.allclose(w.grad, ref_w)
    print("\n[backward] unfused path; grads match eager autograd:", matches)
    print("  kernel paths:", LOG, "(main_full + epilogue_unfused, no fusion)")


def section_2_two_op_chain_fused_by_hand():
    print("\n" + "=" * 72)
    print("SECTION 2: chain y = relu(relu(x @ w1) @ w2), arm ONE fusion by hand")
    print("=" * 72)
    torch.manual_seed(0)
    x = torch.randn(4, 6, dtype=torch.double, requires_grad=True)
    w1 = torch.randn(6, 6, dtype=torch.double, requires_grad=True)
    w2 = torch.randn(6, 6, dtype=torch.double, requires_grad=True)

    def build(cls, *inputs):
        # The three private steps of FusibleFunction.apply, inlined. Returns the
        # real output and the phantom main output (so we can reach the main node).
        ms, es = _StagingCtx(), _StagingCtx()
        with torch.no_grad():
            out = cls.forward(ms, es, *inputs)
        main_out = _MainNode.apply(cls, ms.output_meta, ms.saved, *inputs)
        final = _EpilogueNode.apply(cls, es.saved, (out,), main_out)
        return final, main_out

    h, main1_out = build(MMRelu, x, w1)  # h = relu(x @ w1)
    y, main2_out = build(MMRelu, h, w2)  # y = relu(h @ w2)
    print("\n[built] two ops on one graph:")
    print("  y.grad_fn:", type(y.grad_fn).__name__, "(epilogue of op2)")
    print("  h.grad_fn:", type(h.grad_fn).__name__, "(epilogue of op1)")

    # The pair to fuse. PRODUCER = op2's main node: it will defer the grad of its
    # activation input. CONSUMER = op1's epilogue node: it produced that
    # activation (h) and will run the fused kernel. They are adjacent because op2's
    # main node consumes h, whose grad_fn IS op1's epilogue node.
    producer_main = main2_out.grad_fn  # _MainNodeBackward (op2)
    consumer_epi = h.grad_fn  # _EpilogueNodeBackward (op1)
    assert _is_main(producer_main) and _is_epilogue(consumer_epi)
    edge0 = producer_main.next_functions[0][0]
    print(
        "\n[adjacency] producer_main input 0 ->",
        type(edge0).__name__,
        "is consumer_epi:",
        edge0 is consumer_epi,
    )

    # ARM the fusion -- exactly what apply_epilogue_fusion does for this one pair:
    #  (a) defer input 0 (the activation h) on the producer main node: instead of
    #      computing h's grad GEMM it will emit a DeferredGradTensor on that edge.
    #  (b) stamp the fused kernel onto the consumer epilogue node.
    # The producer's saved set is NOT snapshotted here -- the producer's backward
    # captures it (still alive there) and rides it on the DeferredGradTensor to the
    # consumer, since by the time the consumer's fused kernel runs the producer's
    # real saved_tensors are already freed.
    producer_main.defer_input_idx = 0
    consumer_epi.fused_impl = mm_relu_fused_backward
    print("\n[armed] defer_input_idx=0 on producer; fused_impl on consumer")

    # BACKWARD. The narrated flow:
    #  * op2 epilogue receives a REAL grad (dL/dy) -> epilogue_backward (unfused)
    #  * op2 main is armed -> returns dW2 locally and a DeferredGradTensor for h
    #  * op1 epilogue receives that placeholder -> runs the fused kernel:
    #        dL/dh = (dL/da2 @ w2^T)  then  * relu'(a1)   -- one fused pass
    #  * op1 main receives the resulting real grad -> full main_backward
    LOG.reset()
    y.sum().backward()
    print("\n[backward] done. kernel paths:", LOG)

    rx, rw1, rw2 = _eager_relu_chain_grads([x, w1, w2])
    ok = all(
        [
            torch.allclose(x.grad, rx),
            torch.allclose(w1.grad, rw1),
            torch.allclose(w2.grad, rw2),
        ]
    )
    print("  grads match eager autograd:", ok)
    assert ok
    assert LOG.c["fused_impl"] == 1 and LOG.c["epilogue_unfused"] == 1
    assert LOG.c["main_params_only"] == 1 and LOG.c["main_full"] == 1
    print("  PASS: one epilogue fused by hand through the private machinery.")


if __name__ == "__main__":
    section_1_build_one_op_by_hand()
    section_2_two_op_chain_fused_by_hand()
    print("\nDEMO COMPLETE")
