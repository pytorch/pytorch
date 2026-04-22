# Owner(s): ["module: dynamo"]

import io
import os
import shutil
import sys
import tempfile
import unittest

import torch._dynamo.test_case
from torch._dynamo.repro.after_aot import (
    _extract_distributed_info,
    _get_compile_args,
    InputReader,
    InputWriter,
    save_graph_repro,
)
from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import IS_FBCODE, TEST_CUDA
from torch.utils._traceback import report_compile_source_on_error
from torch.utils._triton import has_triton


def strip_trailing_whitespace(r):
    return "\n".join([l.rstrip() for l in r.split("\n")])


class TestAfterAot(torch._dynamo.test_case.TestCase):
    @unittest.skipIf(IS_FBCODE, "NotImplementedError")
    def test_save_graph_repro(self):
        # TODO: This triggers CUDA context initialization, even though
        # it is CPU only
        saved_kernel_state = None
        if has_triton():
            import triton
            import triton.language as tl

            saved_kernel_state = (
                dict(kernel_side_table.id_to_kernel),
                dict(kernel_side_table.kernel_to_id),
                dict(kernel_side_table.constant_args),
            )
            kernel_side_table.reset_table()

            @triton.jit
            def _repro_kernel(x_ptr, y_ptr, size, BLOCK: tl.constexpr):
                pid = tl.program_id(0)
                offsets = pid * BLOCK + tl.arange(0, BLOCK)
                mask = offsets < size
                tl.store(
                    y_ptr + offsets,
                    tl.load(x_ptr + offsets, mask=mask),
                    mask=mask,
                )

            kernel_side_table.add_kernel(_repro_kernel)

        buf = io.StringIO()
        args = [torch.randn(4)]

        def f(x):
            return (x * x,)

        gm = make_fx(f)(*args)
        with tempfile.TemporaryDirectory() as d:
            save_graph_repro(buf, gm, args, "inductor_accuracy", save_dir=d)
            r = buf.getvalue()
            with report_compile_source_on_error():
                exec(r, {"__compile_source__": r})

            shutil.rmtree(os.path.join(d, "storages"))

            # Should still work even without the save dir
            with report_compile_source_on_error():
                exec(r, {"__compile_source__": r})

        if saved_kernel_state is not None:
            (
                kernel_side_table.id_to_kernel,
                kernel_side_table.kernel_to_id,
                kernel_side_table.constant_args,
            ) = saved_kernel_state

    @unittest.skipIf(sys.byteorder != "little", "checksum depends on endianness")
    def test_dump_tensor(self):
        def test(tensor, expected):
            with tempfile.TemporaryDirectory() as d:
                writer = InputWriter(d, stable_hash=True)
                writer.tensor("x", tensor)
                self.assertExpectedInline("\n".join(writer._lines), expected, skip=1)
                reader = InputReader(d)
                env = {"reader": reader, "torch": torch}
                # TODO: assert no logs
                exec("\n".join(writer._lines), env)
                self.assertEqual(reader.args[0], tensor)

        test(
            torch.zeros(3, 4),
            """\
buf0 = reader.storage('c17fd92682ca5b304ac71074b558dda9e8eb4d66', 48)
reader.tensor(buf0, (3, 4), is_leaf=True)  # x""",
        )
        test(
            torch.ones(3, 4, dtype=torch.int32),
            """\
buf0 = reader.storage('7c221e2da0c58c700cc2996644dd13d042bd552e', 48, dtype_hint=torch.int32)
reader.tensor(buf0, (3, 4), dtype=torch.int32, is_leaf=True)  # x""",
        )
        test(
            torch.empty((3, 4, 5, 6), memory_format=torch.channels_last).fill_(2),
            """\
buf0 = reader.storage('49ebab3961d6221e64c4c72b0aefd976bdd2afc4', 1440)
reader.tensor(buf0, (3, 4, 5, 6), (120, 1, 24, 4), is_leaf=True)  # x""",
        )

    def test_dump_opaque(self):
        """save_graph_repro should emit reader.opaque() for FakeScriptObject args."""
        from torch._library.fake_class_registry import FakeScriptObject

        fake_obj = FakeScriptObject(object(), "__torch__.MyClass", None)

        def f(x):
            return (x * x,)

        args = [torch.randn(4), fake_obj]
        gm = make_fx(f)(args[0])
        with gm.graph.inserting_before(next(iter(gm.graph.nodes))):
            gm.graph.placeholder("obj")
        gm.recompile()

        buf = io.StringIO()
        save_graph_repro(buf, gm, args, "inductor_accuracy")
        r = buf.getvalue()
        self.assertIn("reader.opaque('__torch__.MyClass')", r)

    @unittest.skipIf(not TEST_CUDA, "requires CUDA")
    def test_dump_generator(self):
        gen = torch.cuda.default_generators[0].clone_state()
        writer = InputWriter(None)
        writer.generator("fwd_rng_state_0", gen)
        self.assertExpectedInline(
            "\n".join(writer._lines),
            """reader.generator('cuda', 0)  # fwd_rng_state_0""",
        )
        reader = InputReader(None)
        env = {"reader": reader, "torch": torch}
        exec("\n".join(writer._lines), env)
        self.assertIsInstance(reader.args[0], torch._C.Generator)
        self.assertEqual(reader.args[0].device, torch.device("cuda", 0))

    @unittest.skipIf(not TEST_CUDA, "requires CUDA")
    def test_graphsafe_rng_repro(self):
        """save_graph_repro should emit reader.generator() for Generator args."""
        gen = torch.cuda.default_generators[0].clone_state()

        def f(x):
            return (x * x,)

        args = [torch.randn(4, device="cuda"), gen]
        gm = make_fx(f)(args[0])
        with gm.graph.inserting_before(next(iter(gm.graph.nodes))):
            gm.graph.placeholder("fwd_rng_state_0")
        gm.recompile()

        buf = io.StringIO()
        save_graph_repro(buf, gm, args, "inductor_accuracy")
        r = buf.getvalue()
        self.assertIn("reader.generator('cuda', 0)", r)
        self.assertNotIn("reader.unsupported(", r)

    @unittest.skipIf(not torch.distributed.is_available(), "requires distributed")
    def test_extract_distributed_info_skips_non_string_group_name(self):
        """_extract_distributed_info should skip ops where group_name is an FX Node."""
        gm = torch.fx.GraphModule({}, torch.fx.Graph())
        g = gm.graph
        x = g.placeholder("x")
        group_name = g.placeholder("group_name")
        ar = g.call_function(
            torch.ops._c10d_functional.all_reduce.default,
            args=(x, "sum", group_name),
        )
        g.output(ar)
        gm.recompile()

        result = _extract_distributed_info(gm)
        self.assertEqual(result, {})

    def test_get_compile_args_symbolic_tracing(self):
        """_get_compile_args extracts FakeTensor/SymInt from symbolic tracing."""

        def f(x, size):
            return (x[:size],)

        args = [torch.randn(10), 5]
        gm = make_fx(f, tracing_mode="symbolic")(*args)
        result = _get_compile_args(gm, args)
        # Should NOT be the same object — metadata was extracted
        self.assertIsNot(result, args)
        # First element should be a FakeTensor (not a concrete tensor)
        self.assertIsInstance(result[0], torch._subclasses.FakeTensor)
        # Second element should be a SymInt (not a concrete int)
        self.assertIsInstance(result[1], torch.SymInt)

    def test_get_compile_args_preserves_shapes(self):
        """_get_compile_args preserves symbolic shape info from traced graph."""

        def f(x):
            return (x * 2,)

        args = [torch.randn(4, 8)]
        gm = make_fx(f, tracing_mode="symbolic")(*args)
        result = _get_compile_args(gm, args)
        # FakeTensor should preserve the shape
        self.assertEqual(result[0].shape, torch.Size([4, 8]))

    def test_get_compile_args_real_tracing_returns_concrete(self):
        """_get_compile_args returns original args for real-mode tracing.

        make_fx with tracing_mode='real' produces FakeTensors in placeholder
        metadata, but from different FakeTensorModes.  Extracting these would
        cause a FakeTensorMode mismatch in Inductor.  _get_compile_args must
        detect this and return concrete args instead.
        """

        def f(x, y):
            return (x + y,)

        args = [torch.randn(4), torch.randn(4)]
        gm = make_fx(f, tracing_mode="real")(*args)
        result = _get_compile_args(gm, args)
        # For real tracing (no SymInt inputs), should return args as-is
        self.assertIs(result, args)

    def test_get_compile_args_empty_graph(self):
        """_get_compile_args handles empty graph with no placeholders."""
        gm = torch.fx.GraphModule({}, torch.fx.Graph())
        gm.graph.output(None)
        gm.recompile()
        args = [torch.randn(4)]
        result = _get_compile_args(gm, args)
        self.assertIs(result, args)

    def test_get_compile_args_e2e_symbolic_compile(self):
        """E2E: compile_fx_inner fails with concrete args but succeeds
        with _get_compile_args for symbolically-traced graphs.

        This is the minimal repro for the 'NameError: name s48 is not
        defined' bug in fx_graph_runnable repro scripts.
        """
        from torch._inductor.compile_fx import compile_fx_inner

        def f(n, x, y):
            sliced = x[:n]
            scaled = sliced * (n - 1)
            return (scaled + y[:n],)

        N = 16
        concrete_args = [N, torch.randn(32), torch.randn(32)]
        gm = make_fx(f, tracing_mode="symbolic")(*concrete_args)

        # BUG: concrete args cause NameError on undefined symbolic variable
        with self.assertRaises(NameError):
            compiled = compile_fx_inner(gm, concrete_args)
            self.assertNotIsInstance(compiled, str)
            compiled(list(concrete_args))

        # FIX: _get_compile_args extracts symbolic metadata
        symbolic_args = _get_compile_args(gm, concrete_args)
        compiled = compile_fx_inner(gm, symbolic_args)
        self.assertNotIsInstance(compiled, str)
        result = compiled(list(concrete_args))
        self.assertEqual(result[0].shape, torch.Size([N]))

    def test_get_compile_args_e2e_real_no_fake_mode_mismatch(self):
        """E2E: compile_fx_inner fails when given FakeTensors from
        different FakeTensorModes (extracted from real-mode traced graph
        placeholder metadata) but succeeds with _get_compile_args which
        returns concrete args for real-mode tracing.

        This is the minimal repro for the FakeTensorMode mismatch
        AssertionError that affected 85/126 graphs in the model extractor.
        """
        from torch._inductor.compile_fx import compile_fx_inner

        def f(x, y):
            return (x + y,)

        args = [torch.randn(4), torch.randn(4)]
        gm = make_fx(f, tracing_mode="real")(*args)

        # Verify that real-mode tracing creates FakeTensors with
        # different FakeTensorModes in placeholder metadata
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        fake_modes = set()
        for n in placeholders:
            val = n.meta.get("val")
            if isinstance(val, torch._subclasses.FakeTensor):
                fake_modes.add(id(val.fake_mode))
        self.assertGreater(len(fake_modes), 1, "Expected different FakeTensorModes")

        # BUG: manually extracting FakeTensors causes mode mismatch
        fake_args = [n.meta["val"] for n in placeholders]
        with self.assertRaisesRegex(Exception, "fake mode.*doesn't match"):
            compile_fx_inner(gm, fake_args)

        # FIX: _get_compile_args returns concrete args for real mode
        compile_args = _get_compile_args(gm, args)
        self.assertIs(compile_args, args)
        compiled = compile_fx_inner(gm, compile_args)
        self.assertNotIsInstance(compiled, str)
        result = compiled(list(args))
        self.assertEqual(result[0].shape, torch.Size([4]))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
