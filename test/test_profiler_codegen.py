# Owner(s): ["module: fx"]
import os
import tempfile

import torch
import torch.nn as nn
from torch.fx import symbolic_trace
from torch.fx.profiler_codegen import ProfilerCodeGen
from torch.testing._internal.common_utils import run_tests, TestCase


class TestProfilerCodeGen(TestCase):
    """Tests for ProfilerCodeGen dual-path code generation."""

    def _trace_and_recompile(self, model, args=None):  # noqa: ARG002
        from torch.fx.experimental import _config as fx_config

        gm = symbolic_trace(model)
        old_val = fx_config.profiler_codegen
        fx_config.profiler_codegen = True
        try:
            gm.recompile()
        finally:
            fx_config.profiler_codegen = old_val
        return gm

    def test_generated_code_structure(self):
        """Generated source has three functions and correct dispatch logic."""

        class M(nn.Module):
            def forward(self, x):
                return torch.relu(x)

        gm = self._trace_and_recompile(M(), (torch.randn(4),))
        code = gm._code
        self.assertIn("def _forward_impl(", code)
        self.assertIn("def _forward_profiled(", code)
        self.assertIn("def forward(", code)
        self.assertIn("_is_profiler_enabled", code)

        # _forward_impl should NOT have _RecordFunctionFast
        impl_start = code.index("def _forward_impl(")
        impl_end = code.index("def _forward_profiled(")
        impl_section = code[impl_start:impl_end]
        self.assertNotIn("_RecordFunctionFast", impl_section)

        # _forward_profiled should have _RecordFunctionFast with op name in label
        profiled_start = code.index("def _forward_profiled(")
        forward_start = code.index("def forward(")
        profiled_section = code[profiled_start:forward_start]
        self.assertIn("_RecordFunctionFast", profiled_section)
        self.assertIn("relu", profiled_section)

    def test_output_correctness(self):
        """Both impl and profiled paths produce correct output."""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

            def forward(self, x):
                x = self.linear(x)
                x = torch.relu(x)
                x = x * 2.0
                return x

        model = M()
        inp = torch.randn(3, 10)
        expected = model(inp)

        gm = self._trace_and_recompile(model, (inp,))
        # impl path
        torch.testing.assert_close(gm(inp), expected)
        # profiled path
        with torch.profiler.profile():
            result_profiled = gm(inp)
        torch.testing.assert_close(result_profiled, expected)

    def test_operator_overloads(self):
        """Magic-method ops (+, *, -) work correctly."""

        class M(nn.Module):
            def forward(self, x, y):
                return (x + y) * x - y

        model = M()
        x, y = torch.randn(4), torch.randn(4)
        gm = self._trace_and_recompile(model, (x, y))
        torch.testing.assert_close(gm(x, y), model(x, y))

    def test_empty_graph(self):
        class M(nn.Module):
            def forward(self, x):
                return x

        gm = self._trace_and_recompile(M(), (torch.randn(4),))
        inp = torch.randn(4)
        torch.testing.assert_close(gm(inp), inp)

    def test_node_types(self):
        """call_module, call_method, get_attr are handled correctly."""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm1d(10)
                self.register_buffer("buf", torch.ones(10))

            def forward(self, x):
                x = self.bn(x)  # call_module
                x = x + self.buf  # get_attr + call_function
                x = x.sum()  # call_method
                return x

        model = M()
        model.eval()
        inp = torch.randn(3, 10)
        expected = model(inp)

        gm = self._trace_and_recompile(model, (inp,))
        torch.testing.assert_close(gm(inp), expected)

        # get_attr should NOT be wrapped, call_module/call_method/call_function should
        profiled_start = gm._code.index("def _forward_profiled(")
        forward_start = gm._code.index("def forward(")
        profiled_section = gm._code[profiled_start:forward_start]
        rf_count = profiled_section.count("_RecordFunctionFast")
        # bn (call_module) + add (call_function) + sum (call_method) = 3
        self.assertEqual(rf_count, 3, f"Expected 3 RecordFunctionFast, got {rf_count}")

    def test_recompile_skips_old_record_func(self):
        """ProfilerCodeGen should not inject old-style ## i ## markers."""

        class M(nn.Module):
            def forward(self, x):
                return torch.relu(x)

        gm = self._trace_and_recompile(M(), (torch.randn(4),))
        self.assertNotIn("## 0 ##", gm._code)
        self.assertNotIn("ENTER_GRAPH_PLACEHOLDER_KEY", gm._code)

    def test_default_codegen_unchanged(self):
        """Without ProfilerCodeGen, default codegen has no dual-path."""

        class M(nn.Module):
            def forward(self, x):
                return torch.relu(x)

        gm = symbolic_trace(M())
        gm.recompile()
        self.assertNotIn("_forward_impl", gm._code)
        self.assertNotIn("_forward_profiled", gm._code)

    def test_body_transformer(self):
        """_body_transformer is applied to both impl and profiled bodies."""

        class M(nn.Module):
            def forward(self, x):
                return torch.relu(x)

        gm = symbolic_trace(M())
        codegen = ProfilerCodeGen()
        marker = "# BODY_TRANSFORMER_WAS_HERE\n"
        codegen._body_transformer = lambda body: [marker] + body
        gm.graph.set_codegen(codegen)
        gm.recompile()
        self.assertEqual(gm._code.count("BODY_TRANSFORMER_WAS_HERE"), 2)

    def test_profiler_events_show_labels(self):
        """Profiler trace events contain our custom labels."""

        class M(nn.Module):
            def forward(self, x):
                return torch.relu(x)

        model = M()
        inp = torch.randn(4)
        gm = self._trace_and_recompile(model, (inp,))

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
        ) as prof:
            gm(inp)

        event_keys = [e.key for e in prof.key_averages()]
        self.assertTrue(
            any("relu" in key for key in event_keys),
            f"Expected 'relu' in profiler events, got: {event_keys}",
        )

    def test_codegen_dump_dir(self):
        """codegen_dump_dir writes content-addressed file matching gm._code."""
        from torch.fx.experimental import _config as fx_config

        model = torch.nn.Linear(4, 4)
        inp = torch.randn(1, 4)
        gm = self._trace_and_recompile(model, (inp,))

        with tempfile.TemporaryDirectory() as tmpdir:
            fx_config.codegen_dump_dir = tmpdir
            try:
                gm.recompile()
                files = os.listdir(tmpdir)
                self.assertEqual(len(files), 1)
                self.assertTrue(files[0].startswith("fx_"))
                self.assertTrue(files[0].endswith(".py"))

                with open(os.path.join(tmpdir, files[0])) as f:
                    content = f.read()
                compile(content, files[0], "exec")  # valid Python

                self.assertEqual(gm._codegen_dump_path, os.path.join(tmpdir, files[0]))

                # Idempotency: recompile again, no new files
                gm.recompile()
                self.assertEqual(len(os.listdir(tmpdir)), 1)
            finally:
                fx_config.codegen_dump_dir = ""

    def test_flag_auto_enables_profiler_codegen(self):
        """TORCH_FX_PROFILER_CODEGEN=1 auto-enables ProfilerCodeGen without manual set_codegen."""
        from torch.fx.experimental import _config as fx_config

        class M(nn.Module):
            def forward(self, x):
                return torch.relu(x)

        gm = symbolic_trace(M())
        old_val = fx_config.profiler_codegen
        fx_config.profiler_codegen = True
        try:
            gm.recompile()
        finally:
            fx_config.profiler_codegen = old_val

        self.assertIn("_forward_impl", gm._code)
        self.assertIn("_forward_profiled", gm._code)
        self.assertIsInstance(gm._graph._codegen, ProfilerCodeGen)

    def test_flag_off_no_profiler_codegen(self):
        """When flag is off, default codegen is used."""
        from torch.fx.experimental import _config as fx_config

        class M(nn.Module):
            def forward(self, x):
                return torch.relu(x)

        gm = symbolic_trace(M())
        old_val = fx_config.profiler_codegen
        fx_config.profiler_codegen = False
        try:
            gm.recompile()
        finally:
            fx_config.profiler_codegen = old_val

        self.assertNotIn("_forward_impl", gm._code)
        self.assertNotIn("_forward_profiled", gm._code)


if __name__ == "__main__":
    run_tests()
