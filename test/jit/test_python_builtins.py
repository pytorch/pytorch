# Owner(s): ["oncall: jit"]

import os
import random
import sys
import tempfile
from textwrap import dedent

import torch
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import execWrapper, JitTestCase


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)


def get_fn(file_name, script_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(file_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = module.fn
    return fn


class TestPythonBuiltinOP(JitTestCase):
    def test_add(self):
        def func(a, b):
            c = a + b
            c += a
            return c

        a = torch.rand(1, requires_grad=True)
        b = torch.rand(1, requires_grad=True)
        self.checkScript(func, (a, b), optimize=True)

    def test_mul(self):
        def func(a, b):
            return a * b

        a = torch.rand(1, requires_grad=True)
        b = torch.rand(1, requires_grad=True)
        self.checkScript(func, (a, b), optimize=True)

    def test_matmul_py3(self):
        code = dedent(
            """
        def fn(a, b):
            return a @ b
        """
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            script_path = os.path.join(tmp_dir, "script.py")
            with open(script_path, "w") as f:
                f.write(code)
            fn = get_fn("test_matmul_py3", script_path)

            a = torch.rand(4, 3, requires_grad=True)
            b = torch.rand(3, 2, requires_grad=True)
            self.checkScript(fn, (a, b), optimize=True)

    def test_pow(self):
        def func(a, b):
            return a**b

        def func2(a, b, c, d):
            return c + a**b**d

        def func3(a, b):
            # type: (int, float) -> float
            return a**b

        def func4():
            # type: () -> float
            return 2**-2

        def func5(x, y):
            return x.item() ** y.item()

        a = torch.rand(1, requires_grad=True)
        b = torch.rand(1, requires_grad=True)
        c = torch.rand(1, requires_grad=True)
        d = torch.rand(1, requires_grad=True)
        self.checkScript(func, (a, b), optimize=True)
        self.checkScript(func2, (a, b, c, d), optimize=True)
        self.checkScript(func3, (4, -0.5), optimize=True)
        self.checkScript(func4, ())

        inputs = [
            torch.tensor(2),
            torch.tensor(-2),
            torch.tensor(0.5),
            torch.tensor(0.2),
        ]
        for x in inputs:
            for y in inputs:
                if x < 0:
                    continue
                else:
                    self.checkScript(func5, (x, y))

    def test_triple(self):
        def func(x):
            return 3.0 * x

        x = torch.rand(1, dtype=torch.float, requires_grad=True)
        self.checkScript(func, [x], optimize=True)

    def test_slice(self):
        def func(x):
            return x[:5]

        x = torch.rand(10, dtype=torch.float, requires_grad=True)
        self.checkScript(func, [x], optimize=True)

        def func2(x):
            return x[5:]

        self.checkScript(func2, [x], optimize=True)

        def func3(x):
            return x[:8:2]

        self.checkScript(func3, [x], optimize=True)

        def func4(x):
            return x[1::4]

        self.checkScript(func4, [x], optimize=True)

    def test_gather(self):
        def func(x):
            return x[0]

        x = torch.rand(10, dtype=torch.float, requires_grad=True)
        self.checkScript(func, [x], optimize=True)

    def test_random(self):
        @torch.jit.script
        def f(mean, std):
            return torch.normal(mean, std)

        mean, std = torch.zeros(5, 5), torch.ones(5, 5)
        with torch.random.fork_rng(devices=[]):
            output = torch.normal(mean, std)
        with torch.random.fork_rng(devices=[]):
            script_output = f(mean, std)
        self.assertEqual(output, script_output)

    def _check_code(self, code_str, fn_name, inputs):
        scope = {}
        exec(code_str, globals(), scope)
        cu = torch.jit.CompilationUnit(code_str)
        self.assertEqual(cu.func(*inputs), scope[fn_name](*inputs))

    def test_stepped_tuple_slicing(self):
        def check_slicing_tuple(slicing, tuple_type, tuple):
            template = dedent(
                """
            def func(x):
                # type: ({}) -> Any
                return x{}
            """
            )
            self._check_code(template.format(tuple_type, slicing), "func", [tuple])

        check_slicing_tuple("[-3:3:2]", "Tuple[int, int, int]", (0, 1, 2))
        check_slicing_tuple("[::55]", "Tuple[int, int, int, int, int]", (0, 1, 2, 3, 4))
        check_slicing_tuple("[:4:4]", "Tuple[int, int, int, int, int]", (0, 1, 2, 3, 4))
        check_slicing_tuple(
            "[::-1]", "Tuple[int, int, int, int, int, int, int]", (0, 1, 2, 3, 4, 5, 6)
        )
        check_slicing_tuple(
            "[7:5:2]", "Tuple[int, int, int, int, int, int, int]", (0, 1, 2, 3, 4, 5, 6)
        )
        check_slicing_tuple(
            "[5:7:-2]",
            "Tuple[int, int, int, int, int, int, int]",
            (0, 1, 2, 3, 4, 5, 6),
        )
        check_slicing_tuple("[::-2]", "Tuple[int, int, int, int, int]", (0, 1, 2, 3, 4))
        check_slicing_tuple(
            "[:4:-3]", "Tuple[int, int, int, int, int, int]", (0, 1, 2, 3, 4, 5)
        )
        check_slicing_tuple(
            "[3::-2]", "Tuple[int, int, int, int, int]", (0, 1, 2, 3, 4)
        )

    def test_index(self):
        def consec(size, start=0):
            numel = torch.tensor(size).prod().item()
            return torch.arange(numel).view(size)

        def check_indexing(indexing, tensor):
            template = dedent(
                """
            def func(x):
                return x{}
            """
            )

            self._check_code(template.format(indexing), "func", [tensor])

        def check_dynamic_indexing(indexing, tensor, value1, value2):
            value1 = torch.tensor(value1)
            value2 = torch.tensor(value2)

            template = dedent(
                """
            def func(x, value1, value2):
                i = int(value1)
                j = int(value2)
                return x{}
            """
            )

            self._check_code(
                template.format(indexing), "func", [tensor, value1, value2]
            )

        # basic slices
        check_indexing("[0]", consec((3, 3)))
        check_indexing("[1]", consec((3, 3), 10))
        check_indexing("[2]", consec((3, 3), 19))
        check_indexing("[2]", consec((3,)))
        check_indexing("[-1]", consec((3, 3), 19))
        check_indexing("[0:2]", consec((3, 3, 3)))
        check_indexing("[1:-1]", consec((3, 3, 3)))
        check_indexing("[-3:-1]", consec((6, 3)))
        check_indexing("[1:]", consec((3, 3)))
        check_indexing("[:1]", consec((3, 3)))
        check_indexing("[:]", consec((3, 2)))

        # multi-dim: indexes
        check_indexing("[0, 1]", consec((3, 3)))
        check_indexing("[0, 1]", consec((3, 3, 2)))
        check_indexing("[1, 0, 2]", consec((3, 3, 3)))
        check_indexing("[2, -1]", consec((3, 3)))

        # multi-dim: mixed slicing and indexing
        check_indexing("[0, 1:2]", consec((3, 3)))
        check_indexing("[0, :1]", consec((3, 3, 2)))
        check_indexing("[1, 2:]", consec((3, 3, 3)))
        check_indexing("[-1, 1:, 0]", consec((3, 3, 3, 3)))
        check_indexing("[1:, -1, 0]", consec((3, 3, 3, 3)))
        check_indexing("[-1, 2:, 1:2]", consec((3, 3, 3, 3)))
        check_indexing("[-1, 1:, 0]", consec((3, 3, 3, 3)))
        check_indexing("[-1, :, 0, 2]", consec((3, 3, 3, 3)))

        # zero-sized slices
        check_indexing("[0:0]", consec((2, 2)))
        check_indexing("[0:0, 1]", consec((3, 3)))

        # trivial expression usage
        check_indexing("[1+1]", consec((3, 3)))
        check_indexing("[1:(0 + 2)]", consec((3, 3, 3)))

        # None for new dimensions
        check_indexing("[None, 0]", consec((3, 3)))
        check_indexing("[1, None]", consec((3, 3), 10))
        check_indexing("[None, None, 2]", consec((3, 3), 19))
        check_indexing("[None, 2, None]", consec((3,)))
        check_indexing("[0:2, None]", consec((3, 3, 3)))
        check_indexing("[None, 1:-1]", consec((3, 3, 3)))
        check_indexing("[None, -3:-1, None]", consec((6, 3)))
        check_indexing("[-1, None, 2:, None, 1:2]", consec((3, 3, 3, 3)))
        check_indexing("[None, -1, None, 2:, None, 1:2, None]", consec((3, 3, 3, 3)))

        # dynamic expression usage
        check_dynamic_indexing("[i + j]", consec((3, 3)), 0, 1)
        check_dynamic_indexing("[i:j, i]", consec((3, 3, 2)), 0, 2)

    def test_advancedindex(self):
        def consec(size, start=0):
            numel = torch.tensor(size).prod().item()
            return torch.arange(numel).view(size)

        def check_indexing(indexing, tensor, **kwargs):
            indices_dict = kwargs

            template = dedent(
                """
            def func(x{formals}):
                return x{expr}
            """
            )

            formals = []
            values = []
            for formal, value in indices_dict.items():
                formals.append(formal)
                values.append(value)

            formals = "".join(map(", {}".format, formals))
            inputs = [tensor] + values
            self._check_code(
                template.format(formals=formals, expr=indexing), "func", inputs
            )

        # Indexing with tensor (basic)
        check_indexing("[i]", consec((3, 3)), i=torch.tensor([0]))
        check_indexing("[i]", consec((3, 3)), i=torch.tensor(1))
        check_indexing("[i]", consec((3, 3)), i=torch.tensor([-2]))
        check_indexing("[i]", consec((3, 3), 2), i=torch.tensor([0, 0]))
        check_indexing("[i]", consec((3, 3, 2, 2)), i=torch.tensor([0, -2, 1]))

        # NB: indexing with tensors and indexing with sequences can be implemented
        # in a very similar way (sequences are converted to tensors), so only one
        # case needs to be tested extensively.
        # XXX: When we can index with sequences, replace these cases with
        # sequence indexing expressions; those are much easier to read.

        # Misc sequence advanced indexing
        inp = consec((4, 8, 5))
        to_check = [
            # [[0, 1, 3]]
            ["[i]", {"i": [0, 1, 3]}],
            # [[0, 2], [1, 3]]
            ["[i, j]", {"i": [0, 2], "j": [1, 3]}],
            # [[[0, 1], [0, 1]], [[0, 1], [0, 1]]]
            ["[i, j]", {"i": [[0, 1], [0, 1]], "j": [[0, 1], [0, 1]]}],
            # [[0, 2], [1, 3], [1, 1]]
            ["[i, j, k]", {"i": [0, 2], "j": [1, 3], "k": [1, 1]}],
            # [[0, 2], 1, [1, 1]]
            ["[i, j, k]", {"i": [0, 2], "j": 1, "k": [1, 1]}],
            # [:, :, [0, 3, 4]]
            ["[:, :, i]", {"i": [0, 3, 4]}],
            # [:, [2, 4, 5, 7], 2:4]
            ["[:, i, 2:4]", {"i": [0, 2, 3]}],
            # [[2, 3], :, :]
            ["[i, :, :]", {"i": [2, 3]}],
            # [:, [0, 2, 3], [1, 3, 4]]
            ["[:, i, j]", {"i": [0, 2, 3], "j": [1, 3, 4]}],
            # [:, [0], [1, 2, 4]]
            ["[:, i, j]", {"i": [0], "j": [1, 2, 4]}],
            # [:, [0, 1, 3], [4]]
            ["[:, i, j]", {"i": [0, 1, 3], "j": [4]}],
            # [:, [[0, 1], [1, 0]], [[2, 3]]]
            ["[:, i, j]", {"i": [[0, 1], [1, 0]], "j": [[2, 3]]}],
            # [:, [[0, 1], [2, 3]], [[0]]]
            ["[:, i, j]", {"i": [[0, 1], [2, 3]], "j": [[0]]}],
            # [:, [[5, 6]], [[0, 3], [4, 4]]]
            ["[:, i, j]", {"i": [[5, 6]], "j": [[0, 3], [4, 4]]}],
            # [[0, 2, 3], [1, 3, 4], :]
            ["[i, j, :]", {"i": [0, 2, 3], "j": [1, 3, 4]}],
            # [0, [1, 2, 4], :]
            ["[i, j, :]", {"i": 0, "j": [1, 2, 4]}],
            # [[0, 1, 3], 4, :]
            ["[i, j, :]", {"i": [0, 1, 3], "j": 4}],
            # [[[0, 1], [1, 0]], [[2, 1], [3, 5]], :]
            ["[i, j, :]", {"i": [[0, 1], [1, 0]], "j": [[2, 1], [3, 5]]}],
            # [[[0, 1], [1, 0]], [[2, 3]], :]
            ["[i, j, :]", {"i": [[0, 1], [1, 0]], "j": [[2, 3]]}],
            # [[[0, 1], [2, 3]], [[0]], :]
            ["[i, j, :]", {"i": [[0, 1], [2, 3]], "j": [[0]]}],
            # [[[2, 1]], [[0, 3], [4, 4]], :]
            ["[i, j, :]", {"i": [[2, 1]], "j": [[0, 3], [4, 4]]}],
            # [[[2]], [[0, 3], [4, 1]], 0:2]
            ["[i, j, 0:2]", {"i": [[2]], "j": [[0, 3], [4, 1]]}],
        ]

        for expr, argdict in to_check:
            tensordict = {k: torch.tensor(v) for (k, v) in argdict.items()}
            check_indexing(expr, inp, **tensordict)

    def test_adv_indexing_list(self):
        # indexing with list is equivalent to indexing with tensor
        def func1(x):
            return x[[0, 1, 5]]

        def func2(x):
            return x[[0, 1], [0, 1]]

        def func3(x):
            return x[[[0, 1], [0, 1]], [[0, 1], [0, 1]]]

        def func4(x):
            ls = [0]
            ls.append(1)
            ls.append(2)
            return x[ls]

        def func5(x):
            ls = [0.1, 1.2, 2.3]
            return x[ls]

        input = torch.rand((6, 2))
        self.checkScript(func1, (input,))
        self.checkScript(func2, (input,))
        self.checkScript(func3, (input,))
        self.checkScript(func4, (input,))
        self.checkScript(func5, (input,))

    def test_index_ellipses(self):
        vals = [":", 1, None]
        for _ in range(100):
            indices = [random.choice(vals) for _ in range(4)]
            indices[random.randint(0, len(indices) - 1)] = "..."
            test_str = dedent(
                """
            def f():
                x = torch.ones(10, 9, 8, 7, 6)
                return x{indices}.shape
            """.format(indices=indices)
            )
            test_str = test_str.replace(r"'", r"")
            scope = {}
            execWrapper(test_str, globals(), scope)
            cu = torch.jit.CompilationUnit(test_str)
            res1 = cu.f()
            res2 = scope["f"]()
            self.assertEqual(res1, res2)

    def test_inf(self):
        @torch.jit.script
        def foo(a):
            return a < float("inf")

        s = torch.rand(1)
        self.assertTrue(foo(s))

        @torch.jit.script
        def bar(a):
            return a > float("-inf")

        s = torch.rand(1)
        self.assertTrue(foo(s))

        # test re-assignment on imported source
        str = """
        def foo(x):
            # type: (bool)
            a = float("-inf")
            if not x:
                a = float(torch.tensor([5]))
            return a < 4
        """
        cu = torch.jit.CompilationUnit(str)
        self.assertTrue(cu.foo(True))
        self.assertFalse(cu.foo(False))

    def test_str_to_float(self):
        @torch.jit.script
        def foo(a):
            return 0.5 == float("0.5 hello")

        s = torch.rand(1)
        with self.assertRaisesRegex(RuntimeError, "could not convert string to float"):
            self.assertTrue(foo(s))

        @torch.jit.script
        def foo(a):
            return 0.5 == float("0.5")

        s = torch.rand(1)
        self.assertTrue(foo(s))

        @torch.jit.script
        def foo(a):
            return 0.0 == float("0")

        s = torch.rand(1)
        self.assertTrue(foo(s))


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
