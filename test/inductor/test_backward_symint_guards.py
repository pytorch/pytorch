# Owner(s): ["module: inductor"]
import torch
import torch._functorch.config as functorch_config
import torch._inductor.config as inductor_config
from torch import Tensor
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import clear_caches, fresh_cache


# A minimal opaque op pair modeling fbgemm.dense_to_jagged (fwd) /
# jagged_to_padded_dense (bwd). The backward reconstructs the dense shape
# *inside* the op from the (B_RO + 1)-sized offsets tensor, so the batch symbol
# B_RO only ever reaches the backward graph inside the compound expression
# `B_RO + 1` -- it is never a standalone SymInt input. This is the structure
# that triggers the bug; see test below.
@torch.library.custom_op("test_bro::d2j", mutates_args=())
def _d2j(dense: Tensor, offsets: Tensor, nnz: int) -> Tensor:
    b, length, d = dense.shape
    return dense.reshape(b * length, d)[:nnz].contiguous()


@_d2j.register_fake
def _(dense, offsets, nnz):
    return dense.new_empty((nnz, dense.shape[2]))


@torch.library.custom_op("test_bro::j2d", mutates_args=())
def _j2d(grad: Tensor, offsets: Tensor, length: int, nnz: int) -> Tensor:
    b = offsets.shape[0] - 1
    out = grad.new_zeros(b * length, grad.shape[1])
    out[:nnz] = grad
    return out.reshape(b, length, grad.shape[1])


@_j2d.register_fake
def _(grad, offsets, length, nnz):
    b = offsets.shape[0] - 1
    return grad.new_empty((b, length, grad.shape[1]))


def _d2j_setup(ctx, inputs, output):
    dense, offsets, nnz = inputs
    ctx.save_for_backward(offsets)
    ctx.length = dense.shape[1]
    ctx.nnz = nnz


def _d2j_backward(ctx, grad):
    (offsets,) = ctx.saved_tensors
    return torch.ops.test_bro.j2d(grad, offsets, ctx.length, ctx.nnz), None, None


torch.library.register_autograd(
    "test_bro::d2j", _d2j_backward, setup_context=_d2j_setup
)


class TestBackwardSymIntGuardBinding(TestCase):
    @inductor_config.patch(
        {
            "fx_graph_cache": True,
            "fx_graph_remote_cache": False,
            "compile_threads": 1,
        }
    )
    @functorch_config.patch({"enable_autograd_cache": False})
    def test_backward_compound_symint_guard_is_bindable(self):
        # Regression test for the partitioner passing a raw (compound) SymInt
        # binding to the backward: a dynamic scalar int `B_RO` (Dynamo source
        # L['B_RO']) is used as pos.expand(B_RO, -1, -1) and its only backward
        # relevance is the saved offsets tensor of size B_RO + 1. The backward
        # then references B_RO only through `B_RO + 1`, so FxGraphCache's stored
        # guards_expr renders it via the var_to_sources fallback as L['B_RO'].
        # On a later cache lookup, find_guarded_entry ->
        # evaluate_guards_expression evaluates that expression with args named
        # t0, t1, ... and raises KeyError: 'B_RO'.
        length, dim = 64, 8

        def model(pos, b_ro, offsets, nnz):
            x = pos.view(1, length, dim).expand(b_ro, -1, -1)
            return torch.ops.test_bro.d2j(x, offsets, nnz).sum()

        def make(b_ro):
            pos = torch.randn(length * dim, requires_grad=True)
            offsets = torch.arange(b_ro + 1, dtype=torch.int64) * 3
            return pos, b_ro, offsets, b_ro * 3

        def run(fn, b_ro):
            pos, b, offsets, nnz = make(b_ro)
            fn(pos, b, offsets, nnz).sum().backward()
            return pos.grad

        with fresh_cache():
            compiled = torch.compile(model)
            # First value specializes B_RO; the second promotes it to dynamic
            # (automatic dynamic) and stores the dynamic backward entry.
            run(compiled, 7)
            run(compiled, 5)

            # Drop in-memory caches so the next compile re-looks-up the stored
            # dynamic backward via find_guarded_entry (mirrors a second rank
            # reading the shared cache).
            clear_caches()
            torch._dynamo.reset()

            # This lookup must not raise KeyError, and must match eager grads.
            run(compiled, 7)
            self.assertEqual(run(compiled, 5), run(model, 5))


class TestPositionalGuardExpressionIsEvaluable(TestCase):
    def test_symbol_without_placeholder_is_specialized_not_sourced(self):
        # produce_guards_expression names its args t0..tN and evaluate_guards_expression
        # binds only those, so everything the former emits must be resolvable against
        # them. get_pruned_guards keeps a guard whose UNSIMPLIFIED symbols are all
        # placeholders, but rendering applies ShapeEnv replacements and can surface a
        # symbol that has no placeholder; emitting that symbol's original Dynamo source
        # produced an expression no later lookup could evaluate. Here `extra` stands in
        # for such a symbol -- in production it was a KJT's _offset_per_key[1].
        from torch._dynamo.source import LocalSource
        from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv

        env = ShapeEnv()

        def backed(val, name):
            return env.create_symintnode(
                env.create_symbol(val, LocalSource(name), DimDynamic.DYNAMIC),
                hint=val,
            )

        bound = backed(7, "bound")
        extra = backed(11, "extra")
        total = backed(11 + 5 * 7, "total")

        # Record a guard that ties a placeholder to an expression over `extra`.
        self.assertTrue(bool(total == extra + 5 * bound))

        # `extra` is deliberately NOT a placeholder.
        expr = env.produce_guards_expression([total, bound], guards=env.guards)
        self.assertIsNotNone(expr)
        self.assertNotIn("extra", expr)

        # The whole point: the emitted expression must be evaluable.
        self.assertTrue(env.evaluate_guards_expression(expr, [11 + 5 * 7, 7]))
        # And it must still reject values that violate the recorded relation.
        self.assertFalse(env.evaluate_guards_expression(expr, [11 + 5 * 7 + 1, 7]))


if __name__ == "__main__":
    run_tests()
