# Owner(s): ["module: dynamo"]

import tempfile

import torch
import torch._dynamo.test_case
from torch._dynamo.package import CompilePackage, DiskDynamoStore
from torch._dynamo.precompile_package import default_guard_filter_fn
from torch._dynamo.resume_execution import ContinueExecutionCache
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


class _SuperParent:
    def scale(self, x):
        return x * 3


class _SuperReceiver(_SuperParent):
    def fast(self, x):
        x = x + 1
        torch._dynamo.graph_break()
        parent = super()
        return parent.scale(x)

    def cell(self, x):
        def keep():
            return self

        x = x + 1
        torch._dynamo.graph_break()
        parent = super()
        return parent.scale(x)


def _nullary_super(*args):
    x = args[0] + 1
    torch._dynamo.graph_break()
    try:
        super()
    except RuntimeError as error:
        return x + 1, str(error)


def fn_creator():
    var1 = 1

    def fn(x):
        x = x + 1
        var2 = 1
        torch._dynamo.graph_break()
        x = x + var1

        def inner_fn():
            return var2

        return x

    return fn


class ResumeFunctionTests(torch._dynamo.test_case.TestCase):
    @parametrize("kind", ("fast", "cell", "nullary"))
    def test_implicit_frame_package_roundtrip(self, kind):
        method = (
            _nullary_super if kind == "nullary" else getattr(_SuperReceiver(), kind)
        )

        def fn(x):
            return method(x)

        with tempfile.TemporaryDirectory() as path:
            ctx = DiskDynamoStore()
            package = CompilePackage(fn)
            opt_fn = torch._dynamo.optimize(
                "eager", package=package, guard_filter_fn=default_guard_filter_fn
            )(fn)
            x = torch.ones(3)
            expected = fn(x)
            self.assertEqual(opt_fn(x), expected)
            for name, backend in package.cached_backends.items():
                ctx.record_eager_backend(name, backend)
            ctx.save_package(package, path)
            torch._dynamo.reset()
            ContinueExecutionCache.generated_code_metadata.clear()
            with torch.compiler.set_stance("fail_on_recompile"):
                package, backends = ctx.load_package(fn, path)
                opt_fn = torch._dynamo.optimize(
                    package=package, guard_filter_fn=default_guard_filter_fn
                )(fn)
                package.install(backends)
                self.assertEqual(opt_fn(x), expected)
                self.assertEqual(opt_fn(x + 2), fn(x + 2))
            package.uninstall()

    def test_freevars(self):
        fn = fn_creator()
        opt_fn = torch.compile(fn, backend="eager")
        opt_fn(torch.randn(10))
        entries = [v for k, v in list(globals().items()) if k.startswith("__resume_at")]
        self.assertEqual(len(entries), 1)
        # When freevars are present, install_resume_function_global stores a
        # factory that closes over the code object (first closure cell).
        code = entries[0].__closure__[0].cell_contents
        # co_freevars of resume functions, are sorted concatenation of the original function's co_freevars and co_cellvars
        self.assertEqual(code.co_freevars, ("var1", "var2"))


instantiate_parametrized_tests(ResumeFunctionTests)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
