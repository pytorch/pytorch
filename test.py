import torch
import torch._dynamo.test_case
import torch._inductor.test_case
from torch._dynamo.testing import AotEagerAndRecordGraphs
from torch.testing._internal.opinfo.core import SampleInput


DEBUG = True


class ViewAndMutationMetaFromDynamo(torch._dynamo.test_case.TestCase):
    # Note: These 4 tests are to evalaute the feasability for the four
    # fields  in metadata analysis that were identifed as diffifuclt to import pdb; pdb.set_trace()
    # extract from dynamo

    # We want to run each one with different backend to check the graph
    # To view the created artifact and verify we have sufficient info
    def test_output_alias_info_functional_tensor(self):
        def f(x):
            return x[1].view(-1)

        x = torch.randn(4, 4, requires_grad=True)
        # backend = EagerAndRecordGraphs()
        backend = AotEagerAndRecordGraphs()
        compiled_f = torch.compile(f, backend=backend, fullgraph=True)
        out = compiled_f(x)
        assert len(backend.graphs) == 1
        gm = backend.graphs[0]

    def test_input_alias_info_mutations_hidden_from_autograd(self):
        # From https://github.com/pytorch/pytorch/blob/1258aac1c28f2e66f54ecacaf798a0e7a24206ef/test/functorch/test_aotdispatch.py#L1457
        def f(a):
            a_alias = a.view(-1)
            with torch.no_grad():
                a_alias.mul_(2)
            return a + 1

        x = torch.randn(4, 4, requires_grad=True)
        # backend = EagerAndRecordGraphs()
        backend = AotEagerAndRecordGraphs()
        compiled_f = torch.compile(f, backend=backend, fullgraph=True)
        out = compiled_f(x)
        assert len(backend.graphs) == 1
        gm = backend.graphs[0]

    # Currently fails because second collection pass doesn't pass GM
    # this is okay, as first pass has parity - just need to clena up code
    def test_traced_tangents(self):
        # From ttps://github.com/pytorch/pytorch/blob/1258aac1c28f2e66f54ecacaf798a0e7a24206ef/test/functorch/test_aotdispatch.py#L6541
        def fn(x):
            return x.clone()

        a = torch.randn(2, 3, requires_grad=True, dtype=torch.float64)
        b = torch.randn(3, 3, requires_grad=True, dtype=torch.float64)
        c = torch.randn(4, 3, requires_grad=True, dtype=torch.float64)
        nt = torch.nested.as_nested_tensor([a, b, c], layout=torch.jagged)
        # backend = EagerAndRecordGraphs()
        backend = AotEagerAndRecordGraphs()
        out = torch.compile(fn, backend=backend, fullgraph=True)(nt)
        out_buffer = out.values()
        ga, gb, gc = torch.autograd.grad(out_buffer.sum(), (a, b, c))

        assert len(backend.graphs) == 1
        gm = backend.graphs[0]

    def test_tokens(self):
        # Map of effect type (ex. _EffectType.ORDERED) to token
        # FunctionalTensorMode would have populated this, so we need to validate
        # that we can populate this from dynamo - should be similar to HOPs and Triton
        # kernels
        # From https://github.com/pytorch/pytorch/blob/1258aac1c28f2e66f54ecacaf798a0e7a24206ef/test/higher_order_ops/test_with_effects.py#L89
        def f(x):
            torch.ops.aten._print("moo")
            res = x + x
            torch.ops.aten._print("moo")
            return (res,)

        inputs = (torch.randn(3),)
        # backend = EagerAndRecordGraphs()
        backend = AotEagerAndRecordGraphs()
        out = torch.compile(f, backend=backend, fullgraph=True)(inputs)
        assert len(backend.graphs) == 1
        gm = backend.graphs[0]

    def test_tp_transform_with_uncovered_op(self):
        class DummyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = torch.nn.Linear(3, 5)
                self.bn = torch.nn.BatchNorm1d(5)

            def forward(self, x):
                return self.bn(self.fc(x))

        inputs = (torch.randn(7, 3, requires_grad=False),)
        model = DummyModel()
        res = model(*inputs)
        exported_program = torch.export.export(
            model, inputs, strict=True
        ).run_decompositions()
        tp_res = exported_program.module()(*inputs)
        self.assertEqual(res, tp_res)
        # Expect all_gather to be inserted to distributed sharded fc resutls

    def test_chebyshev_polynomial(self):
        out = torch.zeros((2,))

        def f(samp: SampleInput):
            torch.special.chebyshev_polynomial_u(*samp.args, samp.input, out=out)
            return out

        n = -5.115719318389893
        inp = (torch.tensor(1.0961),)
        samp = SampleInput(
            n,
            args=inp,
            kwargs={},
        )
        # backend = EagerAndRecordGraphs()
        backend = AotEagerAndRecordGraphs()
        compiled_f = torch.compile(f, backend=backend, fullgraph=True)
        out = compiled_f(samp)
        plain = f(samp)
        print(f"Out: {out} vs Plain: {plain}")
        self.assertEqual(out, plain)

    def test_baddbmm(self):
        aten = torch.ops.aten

        def fn(a, b, c, beta):
            return aten.baddbmm(a, b, c, beta=beta)

        compiled_fn = torch.compile(fn, dynamic=True)
        a = torch.randn(6, 1, 100)
        b = torch.randn(6, 128, 64)
        c = torch.randn(6, 64, 100)
        self.assertEqual(compiled_fn(a, b, c, 0.0), fn(a, b, c, 0.0))
        self.assertEqual(compiled_fn(a, b, c, 1.0), fn(a, b, c, 1.0))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
