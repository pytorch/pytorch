import inspect
import torch
from torch._dynamo.test_case import TestCase


def compile_fullgraph(fn):
    return torch.compile(fn, fullgraph=True, backend="eager")


class TestInspectSignature(TestCase):
    def test_nested_fn_signature(self):
        @compile_fullgraph
        def f(x):
            def g(a, b, *, c=None, **kw):
                return a

            sig = inspect.signature(g)
            positional_count = 0
            for parameter in sig.parameters.values():
                if parameter.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                ):
                    positional_count += 1
            return x * positional_count

        x = torch.ones(3, 4)
        y = f(x)
        self.assertEqual(y, x * 2)

    def test_parameters_and_inspect_empty(self):
        @compile_fullgraph
        def f(x):
            def g(a, b, *, c=None, **kw):
                return a

            sig = inspect.signature(g)
            names = []
            kinds = []
            empty_count = 0
            for parameter in sig.parameters.values():
                names.append(parameter.name)
                kinds.append(parameter.kind)
                if parameter.default is inspect._empty:
                    empty_count += 1

            ok = (
                (names == ["a", "b", "c", "kw"])
                and (
                    kinds[0]
                    in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    )
                )
                and (
                    kinds[1]
                    in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    )
                )
                and (kinds[2] == inspect.Parameter.KEYWORD_ONLY)
                and (kinds[3] == inspect.Parameter.VAR_KEYWORD)
                and (empty_count == 3)
            )
            scale = 1 if ok else 0
            return x * scale

        x = torch.randn(2, 2)
        y = f(x)
        self.assertTrue(y.allclose(x))

    def test_signature(self):
        @compile_fullgraph
        def f(x):
            def g(a, b, *, c=None, **kw):
                return a

            sig = g.__signature__
            return x + torch.tensor(len(sig.parameters))

        x = torch.zeros(())
        y = f(x)
        self.assertEqual(y.item(), 4)  # a, b, c, kw - as in g()

    def test_parameters_mapping(self):
        @compile_fullgraph
        def f(x):
            def g(a, b, *, c=None, **kw):
                return a

            sig = inspect.signature(g)
            c = sig.parameters["c"]
            is_kwonly = c.kind == inspect.Parameter.KEYWORD_ONLY
            has_none_default = c.default is None
            ok = is_kwonly and has_none_default
            return x + (1 if ok else 0)

        y = f(torch.tensor(0))
        self.assertEqual(y.item(), 1)

    def test_constant_function_signature(self):
        def top(a, /, b):
            return a + b

        @compile_fullgraph
        def f(x):
            sig = inspect.signature(top)
            positional_only = [
                p
                for p in sig.parameters.values()
                if p.kind == inspect.Parameter.POSITIONAL_ONLY
            ]
            return x + torch.tensor(len(positional_only))

        out = f(torch.tensor(0))
        self.assertEqual(out.item(), 1)

    def test_id_on_nested_function_in_graph(self):
        @torch.compile(fullgraph=True, backend="eager")
        def f(x):
            def g(a):
                return a

            # force id() on the nested function, like
            # create_block_mask->_get_mod_type->inspect.signature->...->id
            # https://github.com/pytorch/pytorch/issues/164247

            i = id(g)
            return x + (1 if isinstance(i, int) and i != 0 else 0)

        out = f(torch.tensor(0))
        assert out.item() == 1


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
