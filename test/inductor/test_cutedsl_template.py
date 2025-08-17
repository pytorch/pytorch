# Owner(s): ["module: inductor"]
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch._inductor.test_case import TestCase


try:
    import cutlass  # noqa: F401
    import cutlass.cute as cute  # noqa: F401

    HAS_CUTLASS = True
except ImportError:
    HAS_CUTLASS = False

if HAS_CUTLASS:
    from torch._inductor.codegen.cutedsl.cutedsl_kernel import CuteDSLTemplateKernel
    from torch._inductor.codegen.cutedsl.cutedsl_template import CuteDSLTemplate
    from torch._inductor.select_algorithm import PartialRender

CUTEDSL_ADD_TEMPLATE = r"""
{{gen_defines()}}

@cute.kernel
def {{kernel_name}}_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx
    m, n = gA.shape

    if thread_idx < m * n:
        mi = thread_idx // n
        ni = thread_idx % n

        if mi < m and ni < n:
            gC[mi, ni] = gA[mi, ni] + gB[mi, ni]

@cute.jit
def {{kernel_name}}_jit(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, stream):
    {{gen_defines()}}
    m, n = mA.shape
    total_threads = m * n
    num_blocks = (total_threads + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    kernel = {{kernel_name}}_kernel(mA, mB, mC)
    kernel.launch(
        grid=[num_blocks, 1, 1],
        block=[THREADS_PER_BLOCK, 1, 1],
        stream=stream
    )

{{def_kernel("input_a", "input_b", "output_c")}}
    cute_a = from_dlpack(input_a)
    cute_b = from_dlpack(input_b)
    cute_c = from_dlpack(output_c)

    {{kernel_name}}_jit(cute_a, cute_b, cute_c, cuda.CUstream(stream))
    return output_c
"""


@unittest.skipUnless(HAS_CUTLASS, "requires cutlass")
class TestCuteDSLTemplate(TestCase):
    """Test cases for CuteDSL template functionality."""

    def test_gen_imports(self):
        kernel = CuteDSLTemplateKernel(
            kernel_name="test_kernel",
            input_nodes=[],
            output_node=None,
        )

        imports = kernel.gen_imports()

        self.assertIn("import torch", imports)
        self.assertIn("import cutlass", imports)
        self.assertIn("import cutlass.cute as cute", imports)
        self.assertIn("from cutlass.cute.runtime import from_dlpack", imports)
        self.assertIsInstance(imports, str)

        lines = imports.strip().split("\n")
        self.assertEqual(len(lines), 5)

    def test_render_includes_imports(self):
        template_source = """@cute.kernel
def {{kernel_name}}_kernel():
    pass

{{def_kernel("input", "output")}}
    return output"""

        mock_template = MagicMock()
        mock_template.render = MagicMock(return_value=template_source)

        kernel = CuteDSLTemplateKernel(
            kernel_name="test_kernel",
            input_nodes=[],
            output_node=None,
        )

        result = kernel.render(mock_template)
        self.assertIsInstance(result, PartialRender)

        rendered_code = result._code

        # The imports might have leading whitespace, so strip it
        rendered_code_stripped = rendered_code.lstrip()

        self.assertTrue(
            rendered_code_stripped.startswith("import torch"),
            f"Code should start with 'import torch', got: {rendered_code_stripped[:50]}",
        )
        self.assertIn("import cutlass", rendered_code)
        self.assertIn("import cutlass.cute as cute", rendered_code)
        self.assertIn("from cutlass.cute.runtime import from_dlpack", rendered_code)
        self.assertIn("@cute.kernel", rendered_code)

    def test_template_env_contains_hooks(self):
        kernel = CuteDSLTemplateKernel(
            kernel_name="test_kernel",
            input_nodes=[],
            output_node=None,
        )

        captured_env = {}

        def mock_render(**kwargs):
            captured_env.update(kwargs)
            return "rendered"

        mock_template = MagicMock()
        mock_template.render = mock_render

        kernel.render(mock_template)

        self.assertIn("def_kernel", captured_env)
        self.assertIn("kernel_name", captured_env)
        self.assertTrue(callable(captured_env["def_kernel"]))

    def test_multiple_templates_unique_names(self):
        # Clean registry first
        test_name = f"unique_test_{id(self)}"
        if test_name in CuteDSLTemplate.all_templates:
            del CuteDSLTemplate.all_templates[test_name]

        _ = CuteDSLTemplate(
            name=test_name,
            source="template1",
        )

        with self.assertRaises(AssertionError):
            _ = CuteDSLTemplate(
                name=test_name,
                source="template2",
            )

    def test_indented_buffer_usage(self):
        kernel = CuteDSLTemplateKernel(
            kernel_name="test_kernel",
            input_nodes=[],
            output_node=None,
        )

        imports = kernel.gen_imports()

        lines = imports.strip().split("\n")
        for line in lines:
            if line:
                self.assertFalse(
                    line.startswith(" "), f"Line should not be indented: '{line}'"
                )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cutedsl_add_e2e(self):
        """End-to-end test with CuteDSL template including code generation verification."""
        from torch._inductor.ir import TensorBox
        from torch._inductor.lowering import lowerings
        from torch._inductor.utils import run_and_get_code

        template = CuteDSLTemplate(
            name="test_add_e2e",
            source=CUTEDSL_ADD_TEMPLATE,
        )

        def cutedsl_add_lowering(a: TensorBox, b: TensorBox) -> TensorBox:
            choices = []
            error = template.maybe_append_choice(
                choices,
                input_nodes=[a, b],
                layout=a.get_layout(),
                THREADS_PER_BLOCK=256,
            )

            if error or not choices:
                default_lowering = lowerings[torch.ops.aten.add.Tensor]
                return default_lowering(a, b)

            # Use the single choice directly (no autotuning)
            return choices[0].output_node()

        with patch.dict(lowerings, {torch.ops.aten.add.Tensor: cutedsl_add_lowering}):
            # Test function
            def test_add(x, y):
                return x + y

            device = "cuda"
            x = torch.randn(128, 4, device=device, dtype=torch.float32)
            y = torch.randn(128, 4, device=device, dtype=torch.float32)

            # Compile and get generated code
            compiled_fn = torch.compile(test_add, backend="inductor")
            result, (code,) = run_and_get_code(compiled_fn, x, y)

            # Verify CuteDSL code is present
            self.assertIn(
                "cute", code.lower(), "CuteDSL code should be in generated code"
            )
            # Verify parameter generation worked
            self.assertIn(
                "THREADS_PER_BLOCK", code, "Parameter should be in generated code"
            )

            # Verify correctness
            expected = x + y
            self.assertTrue(torch.allclose(result, expected, atol=1e-5))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cutedsl_add_e2e_autotune(self):
        """E2E test with multiple CuteDSL template variants for autotuning."""
        from torch._inductor.ir import TensorBox
        from torch._inductor.lowering import lowerings
        from torch._inductor.select_algorithm import autotune_select_algorithm

        template = CuteDSLTemplate(
            name="test_add_autotune",
            source=CUTEDSL_ADD_TEMPLATE,
        )

        def cutedsl_add_lowering(a: TensorBox, b: TensorBox) -> TensorBox:
            choices = []

            # Add multiple variants with different thread counts for autotuning
            thread_variants = [128, 256, 512]
            for threads in thread_variants:
                error = template.maybe_append_choice(
                    choices,
                    input_nodes=[a, b],
                    layout=a.get_layout(),
                    THREADS_PER_BLOCK=threads,
                )
                if error:
                    # Skip this variant if it fails
                    continue

            if not choices:
                default_lowering = lowerings[torch.ops.aten.add.Tensor]
                return default_lowering(a, b)

            # Use autotuning to select the best variant
            return autotune_select_algorithm(
                "cutedsl_add_autotune",
                choices,
                [a, b],
                a.get_layout(),
            )

        with patch.dict(lowerings, {torch.ops.aten.add.Tensor: cutedsl_add_lowering}):
            # Test function
            def test_add(x, y):
                return x + y

            device = "cuda"
            x = torch.randn(128, 128, device=device, dtype=torch.float32)
            y = torch.randn(128, 128, device=device, dtype=torch.float32)

            # Compile and run
            compiled_fn = torch.compile(test_add, backend="inductor")
            result = compiled_fn(x, y)

            # Verify correctness
            expected = x + y
            self.assertTrue(torch.allclose(result, expected, atol=1e-5))

    def test_gen_defines(self):
        """Test that gen_defines correctly generates CuteDSL parameter definitions."""
        kernel = CuteDSLTemplateKernel(
            kernel_name="test_kernel",
            input_nodes=[],
            output_node=None,
        )

        # Test integer parameters
        params = kernel.gen_defines(
            THREADS_PER_BLOCK=256,
            BLOCK_SIZE=128,
            ENABLE_FEATURE=True,
        )

        expected_lines = [
            "THREADS_PER_BLOCK: cutlass.Constexpr = 256",
            "BLOCK_SIZE: cutlass.Constexpr = 128",
            "ENABLE_FEATURE: cutlass.Constexpr = True",
        ]

        for expected_line in expected_lines:
            self.assertIn(expected_line, params)

        # Test float parameters
        params_float = kernel.gen_defines(SCALE_FACTOR=1.5)
        self.assertIn("SCALE_FACTOR: cutlass.Constexpr = 1.5", params_float)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
