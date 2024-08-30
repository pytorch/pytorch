# Owner(s): ["module: dynamo"]

import contextlib
import dis
from typing import Callable, List

import torch
import torch._dynamo.test_case


class ReconstructTest(torch._dynamo.test_case.TestCase):
    @contextlib.contextmanager
    def register_bytecode_hook(self, hook):
        def _build_hook(check: Callable[[List[dis.Instruction]], None]):
            def hook(code, out_code):
                check(list(dis.get_instructions(out_code)))
                return code

            return hook

        hook = _build_hook(hook)

        torch._dynamo.reset()
        handle = torch._dynamo.convert_frame.register_bytecode_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    def test_ConstDict_optimize_reconstruct(self):
        """
        Emit code to reconstruct only the key that changed
        """

        def hook(instructions: List[dis.Instruction]):
            build_map = list(filter(lambda x: x.opname == "BUILD_MAP", instructions))
            self.assertEqual(len(build_map), 1)
            # reconstruct only d[40]
            self.assertEqual(build_map[0].argval, 1)

        def f(d, t):
            d[1] = t
            d[40] = t + 1

        t = torch.randn(3, 4)
        d = {1: t}
        f(d, t)

        with self.register_bytecode_hook(hook):
            d_opt = d.copy()
            opt_f = torch._dynamo.optimize("eager", nopython=True)(f)
            opt_f(d_opt, t)
            self.assertEqual(d, d_opt)

    def test_ConstDict_pop_reconstruct(self):
        """
        If something is pop'ed from the dict, we reconstruct everything
        """

        def hook(instructions: List[dis.Instruction]):
            build_map = list(filter(lambda x: x.opname == "BUILD_MAP", instructions))
            self.assertEqual(len(build_map), 1)
            # reconstruct everything
            self.assertEqual(build_map[0].argval, 2)

        def f(d, t):
            d.pop(2)
            d[40] = t + 1

        t = torch.randn(3, 4)
        d = {1: t, 2: t + 1}
        d_opt = d.copy()

        f(d, t)

        with self.register_bytecode_hook(hook):
            opt_f = torch._dynamo.optimize("eager", nopython=True)(f)
            opt_f(d_opt, t)
            self.assertEqual(d, d_opt)

    def test_ConstDict_clear_reconstruct(self):
        """
        If dict.clear() is used, we reconstruct everything
        """

        def hook(instructions: List[dis.Instruction]):
            build_map = list(filter(lambda x: x.opname == "BUILD_MAP", instructions))
            self.assertEqual(len(build_map), 1)
            # reconstruct everything
            self.assertEqual(build_map[0].argval, 1)

        def f(d, t):
            d.clear()
            d[3] = t + 3

        t = torch.randn(3, 4)
        d = {1: t, 2: t + 1}
        d_opt = d.copy()

        f(d, t)

        with self.register_bytecode_hook(hook):
            opt_f = torch._dynamo.optimize("eager", nopython=True)(f)
            opt_f(d_opt, t)
            self.assertEqual(d, d_opt)

    def test_create_dict_reconstruct(self):
        """
        If dict is created inside a function, everything needs to be reconstructed
        """

        def hook(instructions: List[dis.Instruction]):
            build_map = list(filter(lambda x: x.opname == "BUILD_MAP", instructions))
            self.assertEqual(len(build_map), 1)
            # reconstruct everything
            self.assertEqual(build_map[0].argval, 2)

        def f(t):
            return {1: t, 2: t + 1}

        t = torch.randn(3, 4)
        d = f(t)

        with self.register_bytecode_hook(hook):
            opt_f = torch._dynamo.optimize("eager", nopython=True)(f)
            d_opt = opt_f(t)
            self.assertEqual(d, d_opt)

    def test_functional_call_reconstruct(self):
        """
        PyTorch shouldn't codegen any key/value when functional_call is used
        """

        def hook(instructions: List[dis.Instruction]):
            build_map = list(filter(lambda x: x.opname == "BUILD_MAP", instructions))
            self.assertEqual(len(build_map), 1)
            # don't reconstruct anything
            self.assertEqual(build_map[0].argval, 0)

        m = torch.nn.Linear(3, 3)
        new_bias = torch.randn(3)
        new_weight = torch.randn(3, 3)

        def fn(new_weight, new_bias, x):
            return torch.func.functional_call(
                m, {"weight": new_weight, "bias": new_bias}, x
            )

        x = torch.randn(2, 3)
        expected = torch.nn.functional.linear(x, new_weight, new_bias)
        with self.register_bytecode_hook(hook):
            opt_fn = torch._dynamo.optimize("eager", nopython=True)(fn)
            got = opt_fn(new_weight, new_bias, x)
            self.assertEqual(expected, got)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
