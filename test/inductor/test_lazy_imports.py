# Owner(s): ["module: inductor"]

import subprocess
import sys

from torch._inductor.test_case import run_tests, TestCase


class TestLazyImports(TestCase):
    def test_compile_fx_import_does_not_load_heavy_modules(self):
        code = """
import sys
import torch._inductor.compile_fx

heavy_modules = [
    'torch._inductor.codegen.common',
    'torch._inductor.fx_passes.pre_grad',
    'torch._inductor.fx_passes.post_grad',
    'torch._inductor.fx_passes.joint_graph',
    'torch._inductor.graph',
]

loaded = []
for mod in heavy_modules:
    if mod in sys.modules:
        loaded.append(mod)

if loaded:
    print(f"FAIL: The following modules were eagerly loaded: {loaded}")
    sys.exit(1)
else:
    print("PASS: Heavy modules were not eagerly loaded")
    sys.exit(0)
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"Test failed:\nstdout: {result.stdout}\nstderr: {result.stderr}",
        )

    def test_lazy_import_functions_exist(self):
        from torch._inductor.compile_fx import (
            _lazy_import_codegen_common,
            _lazy_import_fx_passes,
            _lazy_import_graph_lowering,
        )

        self.assertTrue(callable(_lazy_import_codegen_common))
        self.assertTrue(callable(_lazy_import_fx_passes))
        self.assertTrue(callable(_lazy_import_graph_lowering))

    def test_lazy_import_codegen_common_returns_expected_functions(self):
        from torch._inductor.compile_fx import _lazy_import_codegen_common

        get_wrapper_codegen_for_device, init_backend_registration = (
            _lazy_import_codegen_common()
        )

        self.assertTrue(callable(get_wrapper_codegen_for_device))
        self.assertTrue(callable(init_backend_registration))

    def test_lazy_import_fx_passes_returns_expected_functions(self):
        from torch._inductor.compile_fx import _lazy_import_fx_passes

        pre_grad_passes, joint_graph_passes, post_grad_passes, view_to_reshape = (
            _lazy_import_fx_passes()
        )

        self.assertTrue(callable(pre_grad_passes))
        self.assertTrue(callable(joint_graph_passes))
        self.assertTrue(callable(post_grad_passes))
        self.assertTrue(callable(view_to_reshape))

    def test_lazy_import_graph_lowering_returns_class(self):
        from torch._inductor.compile_fx import _lazy_import_graph_lowering

        GraphLowering = _lazy_import_graph_lowering()

        self.assertTrue(isinstance(GraphLowering, type))
        self.assertEqual(GraphLowering.__name__, "GraphLowering")

    def test_lazy_import_caching(self):
        from torch._inductor.compile_fx import (
            _lazy_import_codegen_common,
            _lazy_import_fx_passes,
            _lazy_import_graph_lowering,
        )

        codegen1 = _lazy_import_codegen_common()
        fx_passes1 = _lazy_import_fx_passes()
        graph1 = _lazy_import_graph_lowering()

        codegen2 = _lazy_import_codegen_common()
        fx_passes2 = _lazy_import_fx_passes()
        graph2 = _lazy_import_graph_lowering()

        self.assertIs(codegen1[0], codegen2[0])
        self.assertIs(codegen1[1], codegen2[1])
        self.assertIs(fx_passes1[0], fx_passes2[0])
        self.assertIs(graph1, graph2)


if __name__ == "__main__":
    run_tests()
