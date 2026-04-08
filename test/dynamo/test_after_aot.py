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
    InputReader,
    InputWriter,
    save_graph_repro,
)
from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import IS_FBCODE, requires_cuda
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


    @requires_cuda
    def test_backed_symbols_with_symint_input(self):
        """Test that backed symbols from symint inputs are properly defined.

        When make_fx re-traces with tracing_mode="symbolic", symint inputs
        create symbols that don't correspond to input tensor dimensions.
        These symbols need explicit definitions in Inductor's wrapper code.
        """
        from torch._dynamo.repro.after_aot import _capture_backed_symbols_for_repro
        from torch._inductor.compile_fx import compile_fx_inner

        device = torch.device("cuda")

        class Repro(torch.nn.Module):
            def forward(self, output_size, data, repeats):
                repeated = torch.ops.aten.repeat_interleave.Tensor(
                    repeats, output_size=output_size
                )
                result = torch.ops.aten.index.Tensor(data, [repeated % data.shape[0]])
                return (torch.ops.aten.sum.default(result),)

        # symint must be DIFFERENT from tensor dimensions to avoid unification
        output_size = 512
        data = torch.randn(1024, 128, device=device)
        repeats = torch.tensor(
            [2] * (output_size // 2), dtype=torch.int64, device=device
        )

        model = Repro()
        args = [output_size, data, repeats]

        gm = make_fx(model, tracing_mode="symbolic")(*args)

        # Capture backed symbols - this is the fix being tested
        _capture_backed_symbols_for_repro(gm)

        # This would raise NameError without the fix
        compiled = compile_fx_inner(gm, args)
        result = compiled(list(args))
        self.assertIsNotNone(result)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
