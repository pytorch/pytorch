# Owner(s): ["module: inductor"]
import unittest
from unittest.mock import MagicMock, patch

from expecttest import assert_expected_inline

import torch
from torch._inductor.test_case import TestCase
from torch._inductor.virtualized import V
from torch.testing._internal.inductor_utils import MockGraphHandler


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

{{def_kernel("input_a", "input_b")}}
    cute_a = from_dlpack(input_a)
    cute_b = from_dlpack(input_b)
    cute_c = from_dlpack({{get_output()}})

    {{kernel_name}}_jit(cute_a, cute_b, cute_c, cuda.CUstream(stream))
    return {{get_output()}}
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
        self.assertEqual(len(lines), 8)

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

        assert_expected_inline(
            params,
            """\
THREADS_PER_BLOCK: cutlass.Constexpr = 256
BLOCK_SIZE: cutlass.Constexpr = 128
ENABLE_FEATURE: cutlass.Constexpr = True
""",
        )

        params_float = kernel.gen_defines(SCALE_FACTOR=1.5)
        assert_expected_inline(
            params_float,
            """\
SCALE_FACTOR: cutlass.Constexpr = 1.5
""",
        )

    def test_template_aliasing(self):
        """Test that template variables are correctly aliased to function arguments."""
        from torch._inductor.ir import Buffer

        mock_input1 = MagicMock(spec=Buffer)
        mock_input1.get_name.return_value = "buf_input1"

        mock_input2 = MagicMock(spec=Buffer)
        mock_input2.get_name.return_value = "buf_input2"

        mock_output = MagicMock(spec=Buffer)
        mock_output.get_name.return_value = "buf_output"

        mock_graph = MockGraphHandler()
        with V.set_graph_handler(mock_graph):
            kernel = CuteDSLTemplateKernel(
                kernel_name="test_aliasing",
                input_nodes=[mock_input1, mock_input2],
                output_node=mock_output,
            )

            def_kernel_hook = kernel.def_kernel("custom_a", "custom_b")
            self.assertEqual(def_kernel_hook, "<DEF_KERNEL>")

            self.assertIn("<DEF_KERNEL>", kernel.render_hooks)

            hook_fn = kernel.render_hooks["<DEF_KERNEL>"]
            generated_code = hook_fn()

            # Check that the generated code contains the expected aliasing statements
            self.assertIn("custom_a = arg_custom_a", generated_code)
            self.assertIn("custom_b = arg_custom_b", generated_code)

    def test_get_output_hook(self):
        """Test the get_output() template hook."""
        from torch._inductor.ir import Buffer

        mock_output = MagicMock(spec=Buffer)
        mock_output.get_name.return_value = "buf_test_output"

        mock_graph = MockGraphHandler()
        with V.set_graph_handler(mock_graph):
            kernel = CuteDSLTemplateKernel(
                kernel_name="test_output",
                input_nodes=[],
                output_node=mock_output,
            )

            with self.assertRaises(ValueError):
                # error if no output buffer
                result = kernel.get_output()

            kernel.args.output_buffers["buf_test_output"] = "arg_buf_test_output"
            result = kernel.get_output()
            self.assertEqual(result, "arg_buf_test_output")

    def test_modification_subgraph(self):
        """Test the modification() method and subgraph processing."""

        from torch._inductor.ir import Buffer

        mock_subgraph1 = MagicMock(spec=Buffer)
        mock_subgraph2 = MagicMock(spec=Buffer)
        subgraphs = [mock_subgraph1, mock_subgraph2]

        mock_output = MagicMock(spec=Buffer)
        mock_output.get_name.return_value = "buf_output"

        kernel = CuteDSLTemplateKernel(
            kernel_name="test_modification",
            input_nodes=[],
            output_node=mock_output,
            subgraphs=subgraphs,
        )

        result = kernel._get_subgraph(0)
        self.assertEqual(result, mock_subgraph1)

        result = kernel._get_subgraph(1)
        self.assertEqual(result, mock_subgraph2)

        with self.assertRaises(AssertionError):
            kernel._get_subgraph(2)

    def test_cutedsl_op_overrides(self):
        """Test the new CuteDSLOpOverrides class."""
        import torch
        from torch._inductor.codegen.common import CSEVariable
        from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
            CuteDSLOpOverrides,
        )
        from torch.utils._sympy.value_ranges import ValueRanges

        mock_cse_a = MagicMock(spec=CSEVariable)
        mock_cse_a.__str__.return_value = "tensor_a"
        mock_cse_a.dtype = torch.float32
        mock_cse_a.bounds = ValueRanges.unknown()

        mock_cse_b = MagicMock(spec=CSEVariable)
        mock_cse_b.__str__.return_value = "tensor_b"
        mock_cse_b.dtype = torch.float32
        mock_cse_b.bounds = ValueRanges.unknown()

        mock_graph = MockGraphHandler()
        with V.set_graph_handler(mock_graph):
            kernel = CuteDSLTemplateKernel(
                kernel_name="test_ops",
                input_nodes=[],
                output_node=None,
            )
            with V.set_kernel_handler(kernel):
                result = CuteDSLOpOverrides.add(mock_cse_a, mock_cse_b)
                self.assertIsInstance(result, CSEVariable)

                result = CuteDSLOpOverrides.mul(mock_cse_a, mock_cse_b)
                self.assertIsInstance(result, CSEVariable)

                result = CuteDSLOpOverrides.truediv(mock_cse_a, mock_cse_b)
                self.assertIsInstance(result, CSEVariable)

                result = CuteDSLOpOverrides.exp(mock_cse_a)
                self.assertIsInstance(result, CSEVariable)

                result = CuteDSLOpOverrides.sqrt(mock_cse_a)
                self.assertIsInstance(result, CSEVariable)

                with self.assertRaises(NotImplementedError):
                    result = CuteDSLOpOverrides.maximum(mock_cse_a, mock_cse_b)
                    result = CuteDSLOpOverrides.minimum(mock_cse_a, mock_cse_b)

        scalar_result = CuteDSLOpOverrides._ensure_tensor_ssa("5.0", mock_cse_a)
        self.assertEqual(scalar_result, "cute.full_like(tensor_a, 5.0)")

        tensor_result = CuteDSLOpOverrides._ensure_tensor_ssa(mock_cse_a, mock_cse_b)
        self.assertEqual(tensor_result, "tensor_a")

    def test_cse_integration(self):
        """Test CSE (Common Subexpression Elimination) integration."""
        from torch._inductor.codegen.common import CSE

        mock_graph = MockGraphHandler()
        with V.set_graph_handler(mock_graph):
            kernel = CuteDSLTemplateKernel(
                kernel_name="test_cse",
                input_nodes=[],
                output_node=None,
            )

            self.assertIsInstance(kernel.cse, CSE)
            self.assertEqual(kernel.cse.name_prefix, "tmp")

            with V.set_kernel_handler(kernel):
                test_expr = "x"
                var = kernel.cse.generate(kernel.body, test_expr, dtype=None)
                self.assertTrue(str(var).startswith("tmp"))


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
