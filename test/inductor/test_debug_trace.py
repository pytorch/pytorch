# Owner(s): ["module: inductor"]
import logging
import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch._inductor import config, test_operators
from torch._inductor.pretty_print_ir import (
    format_post_lowering_ir,
    format_pre_fusion_ir,
)
from torch._inductor.utils import fresh_cache
from torch.testing._internal.common_utils import skipIfWindows
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU, requires_gpu
from torch.testing._internal.logging_utils import multiple_logs_to_string


try:
    try:
        from . import test_torchinductor
    except ImportError:
        import test_torchinductor  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
except unittest.SkipTest:
    if __name__ == "__main__":
        sys.exit(0)
    raise


def filesize(filename: Path):
    if not filename.exists():
        raise AssertionError(f"{filename} is missing")
    return os.stat(filename).st_size


@config.patch("trace.enabled", True)
class TestDebugTrace(test_torchinductor.TestCase):
    def _compile_pretty_ir(self, fn, *inputs):
        """Compile fn and return its pretty IR log for each stage."""
        log_streams, ctx = multiple_logs_to_string(
            "torch._inductor.debug", "ir_post_lowering_pretty", "ir_pre_fusion_pretty"
        )
        with config.patch("force_disable_caches", True), ctx():
            actual = torch.compile(fn, fullgraph=True)(*inputs)
        self.assertEqual(actual, fn(*inputs))
        return tuple(stream.getvalue().strip() for stream in log_streams)

    def test_ir_pre_fusion_pretty_unsupported(self):
        class UnsupportedNode:
            @staticmethod
            def get_name():
                return "op0"

        self.assertEqual(
            format_pre_fusion_ir([UnsupportedNode()]),
            "kernel op0:\n    unimplemented UnsupportedNode",
        )

    @requires_gpu()
    def test_ir_pre_fusion_pretty(self):
        def fn(a):
            return torch.sin(a + 1), a.sum(dim=1)

        inp = torch.randn(4, 8, device=GPU_TYPE)
        post_lowering, pre_fusion = self._compile_pretty_ir(fn, inp)
        self.assertExpectedInline(
            post_lowering,
            """\
POST-LOWERING PRETTY IR
kernel op0(
    arg0_1: f32[4, 8]
) -> buf0: f32[4, 8]:
    for p0 in [0, 4):
        for p1 in [0, 8):
            tmp0: f32 = arg0_1[p1 + 8*p0]
            tmp1: f32 = 1.0
            tmp2: f32 = tmp0 + tmp1
            tmp3: f32 = sin(tmp2)
            buf0[p1 + 8*p0] = tmp3

kernel op1(
    arg0_1: f32[4, 8]
) -> buf1: f32[4]:
    for p0 in [0, 4):
        acc_0: f32 = 0
        for r0 in [0, 8):
            tmp0: f32 = arg0_1[r0 + 8*p0]
            acc_0 = acc_0 + tmp0
        buf1[p0] = acc_0""",
        )
        self.assertExpectedInline(
            pre_fusion,
            """\
PRE-FUSION PRETTY IR
kernel op0(
    arg0_1: f32[4, 8]
) -> buf0: f32[4, 8]:
    for p0 in [0, 4):
        for p1 in [0, 8):
            tmp0: f32 = arg0_1[p1 + 8*p0]
            tmp1: f32 = 1.0
            tmp2: f32 = tmp0 + tmp1
            tmp3: f32 = sin(tmp2)
            buf0[p1 + 8*p0] = tmp3

kernel op1(
    arg0_1: f32[4, 8]
) -> buf1: f32[4]:
    for p0 in [0, 4):
        acc_0: f32 = 0
        for r0 in [0, 8):
            tmp0: f32 = arg0_1[r0 + 8*p0]
            acc_0 = acc_0 + tmp0
        buf1[p0] = acc_0""",
        )

    def test_ir_post_lowering_pretty_unsupported(self):
        class UnsupportedOperation:
            @staticmethod
            def get_operation_name():
                return "op0"

        self.assertEqual(
            format_post_lowering_ir([UnsupportedOperation()]),
            "kernel op0:\n    unimplemented UnsupportedOperation",
        )

    @requires_gpu()
    def test_ir_post_lowering_pretty(self):
        def fn(a):
            return torch.sin(a + 1), a.sum(dim=1), a.unsqueeze(1) * 2

        inp = torch.randn(4, 8, device=GPU_TYPE)
        post_lowering, pre_fusion = self._compile_pretty_ir(fn, inp)
        # Post-lowering loops keep their lowered shape, including size-1 dims.
        self.assertExpectedInline(
            post_lowering,
            """\
POST-LOWERING PRETTY IR
kernel op0(
    arg0_1: f32[4, 8]
) -> buf0: f32[4, 8]:
    for p0 in [0, 4):
        for p1 in [0, 8):
            tmp0: f32 = arg0_1[p1 + 8*p0]
            tmp1: f32 = 1.0
            tmp2: f32 = tmp0 + tmp1
            tmp3: f32 = sin(tmp2)
            buf0[p1 + 8*p0] = tmp3

kernel op1(
    arg0_1: f32[4, 8]
) -> buf1: f32[4]:
    for p0 in [0, 4):
        acc_0: f32 = 0
        for r0 in [0, 8):
            tmp0: f32 = arg0_1[r0 + 8*p0]
            acc_0 = acc_0 + tmp0
        buf1[p0] = acc_0

kernel op2(
    arg0_1: f32[4, 8]
) -> buf2: f32[4, 1, 8]:
    for p0 in [0, 4):
        for p1 in [0, 1):
            for p2 in [0, 8):
                tmp0: f32 = arg0_1[p2 + 8*p0]
                tmp1: f32 = 2.0
                tmp2: f32 = tmp0 * tmp1
                buf2[p2 + 8*p0] = tmp2""",
        )
        self.assertExpectedInline(
            pre_fusion,
            """\
PRE-FUSION PRETTY IR
kernel op0(
    arg0_1: f32[4, 8]
) -> buf0: f32[4, 8]:
    for p0 in [0, 4):
        for p1 in [0, 8):
            tmp0: f32 = arg0_1[p1 + 8*p0]
            tmp1: f32 = 1.0
            tmp2: f32 = tmp0 + tmp1
            tmp3: f32 = sin(tmp2)
            buf0[p1 + 8*p0] = tmp3

kernel op1(
    arg0_1: f32[4, 8]
) -> buf1: f32[4]:
    for p0 in [0, 4):
        acc_0: f32 = 0
        for r0 in [0, 8):
            tmp0: f32 = arg0_1[r0 + 8*p0]
            acc_0 = acc_0 + tmp0
        buf1[p0] = acc_0

kernel op2(
    arg0_1: f32[4, 8]
) -> buf2: f32[4, 1, 8]:
    for p0 in [0, 4):
        for p1 in [0, 8):
            tmp0: f32 = arg0_1[p1 + 8*p0]
            tmp1: f32 = 2.0
            tmp2: f32 = tmp0 * tmp1
            buf2[p1 + 8*p0] = tmp2""",
        )

    @requires_gpu()
    def test_ir_post_lowering_pretty_indirect(self):
        def fn(idx, table):
            return table[idx], torch.nn.functional.pad(table[idx], (1, 1))

        idx = torch.tensor([2, -3, 1], device=GPU_TYPE)
        table = torch.randn(3, 8, device=GPU_TYPE)
        post_lowering, pre_fusion = self._compile_pretty_ir(fn, idx, table)
        # Inside masked(), the indirect index is inlined into the load.
        self.assertExpectedInline(
            post_lowering,
            """\
POST-LOWERING PRETTY IR
kernel op0(
    arg0_1: i64[3],
    arg1_1: f32[3, 8]
) -> buf0: f32[3, 8]:
    for p0 in [0, 3):
        for p1 in [0, 8):
            tmp0: i64 = arg0_1[p0]
            tmp1: f32 = arg1_1[p1 + 8*wrap_neg(tmp0)]
            buf0[p1 + 8*p0] = tmp1

kernel op1(
    arg0_1: i64[3],
    arg1_1: f32[3, 8]
) -> buf1: f32[3, 10]:
    for p0 in [0, 3):
        for p1 in [0, 10):
            tmp0: i64 = (-1) + p1
            tmp1: i64 = 0
            tmp2: bool = tmp0 >= tmp1
            tmp3: i64 = (-1) + p1
            tmp4: i64 = 8
            tmp5: bool = tmp3 < tmp4
            tmp6: bool = tmp2 & tmp5
            tmp7: f32 = where(tmp6, arg1_1[(-1) + p1 + 8*wrap_neg(arg0_1[p0])], 0.0)
            buf1[p1 + 10*p0] = tmp7""",
        )
        self.assertExpectedInline(
            pre_fusion,
            """\
PRE-FUSION PRETTY IR
kernel op0(
    arg0_1: i64[3],
    arg1_1: f32[3, 8]
) -> buf0: f32[3, 8]:
    for p0 in [0, 3):
        for p1 in [0, 8):
            tmp0: i64 = arg0_1[p0]
            tmp1: f32 = arg1_1[p1 + 8*wrap_neg(tmp0)]
            buf0[p1 + 8*p0] = tmp1

kernel op1(
    arg0_1: i64[3],
    arg1_1: f32[3, 8]
) -> buf1: f32[3, 10]:
    for p0 in [0, 3):
        for p1 in [0, 10):
            tmp0: i64 = (-1) + p1
            tmp1: i64 = 0
            tmp2: bool = tmp0 >= tmp1
            tmp3: i64 = (-1) + p1
            tmp4: i64 = 8
            tmp5: bool = tmp3 < tmp4
            tmp6: bool = tmp2 & tmp5
            tmp7: f32 = where(tmp6, arg1_1[(-1) + p1 + 8*wrap_neg(arg0_1[p0])], 0.0)
            buf1[p1 + 10*p0] = tmp7""",
        )

    @requires_gpu()
    def test_ir_pretty_extern_kernel(self):
        def fn(x, w):
            return (x @ w).relu()

        x, w = torch.randn(4, 8, device=GPU_TYPE), torch.randn(8, 8, device=GPU_TYPE)
        post_lowering, pre_fusion = self._compile_pretty_ir(fn, x, w)
        self.assertExpectedInline(
            post_lowering,
            """\
POST-LOWERING PRETTY IR
extern_kernel op0(  # extern_kernels.mm
    arg1_1: f32[4, 8],
    arg0_1: f32[8, 8]
) -> buf0: f32[4, 8]

kernel op1(
    buf0: f32[4, 8]
) -> buf1: f32[4, 8]:
    for p0 in [0, 4):
        for p1 in [0, 8):
            tmp0: f32 = buf0[p1 + 8*p0]
            tmp1: f32 = relu(tmp0)
            buf1[p1 + 8*p0] = tmp1""",
        )
        self.assertExpectedInline(
            pre_fusion,
            """\
PRE-FUSION PRETTY IR
extern_kernel op0(  # extern_kernels.mm
    arg1_1: f32[4, 8],
    arg0_1: f32[8, 8]
) -> buf0: f32[4, 8]

kernel op1(
    buf0: f32[4, 8]
) -> buf1: f32[4, 8]:
    for p0 in [0, 4):
        for p1 in [0, 8):
            tmp0: f32 = buf0[p1 + 8*p0]
            tmp1: f32 = relu(tmp0)
            buf1[p1 + 8*p0] = tmp1""",
        )

    @requires_gpu()
    def test_ir_pretty_extern_kernel_multi_output(self):
        def fn(x):
            values, indices = torch.kthvalue(x, 2)
            return values + 1, indices

        inp = torch.randn(4, 5, device=GPU_TYPE)
        post_lowering, pre_fusion = self._compile_pretty_ir(fn, inp)
        # The MultiOutput selectors op1 and op2 are folded into op0's outputs.
        self.assertExpectedInline(
            post_lowering,
            """\
POST-LOWERING PRETTY IR
extern_kernel op0(  # torch.ops.aten.kthvalue.default
    arg0_1: f32[4, 5]
) -> (buf1: f32[4], buf2: i64[4])

kernel op3(
    buf1: f32[4]
) -> buf3: f32[4]:
    for p0 in [0, 4):
        tmp0: f32 = buf1[p0]
        tmp1: f32 = 1.0
        tmp2: f32 = tmp0 + tmp1
        buf3[p0] = tmp2""",
        )
        self.assertExpectedInline(
            pre_fusion,
            """\
PRE-FUSION PRETTY IR
extern_kernel op0(  # torch.ops.aten.kthvalue.default
    arg0_1: f32[4, 5]
) -> (buf1: f32[4], buf2: i64[4])

kernel op3(
    buf1: f32[4]
) -> buf3: f32[4]:
    for p0 in [0, 4):
        tmp0: f32 = buf1[p0]
        tmp1: f32 = 1.0
        tmp2: f32 = tmp0 + tmp1
        buf3[p0] = tmp2""",
        )

    def test_debug_trace(self):
        @torch.compile
        def fn(a, b):
            a = test_operators.realize(a + 1) + 2
            return torch.matmul(a, b)

        (pre_fusion_stream, post_fusion_stream), ctx = multiple_logs_to_string(
            "torch._inductor.debug", "ir_pre_fusion", "ir_post_fusion"
        )

        # TODO(aakhundov): make this work with fresh_cache
        # instead of force_disable_caches. currently, with the latter
        # enabled, we get `inductor [('fxgraph_cache_hit', 1)]` in
        # the counters: so the cache is actually hit and the test fails.
        with config.patch(
            {
                "trace.debug_dir": tempfile.mkdtemp(),
                "force_disable_caches": True,
            }
        ):
            with (
                self.assertLogs(
                    logging.getLogger("torch._inductor.debug"), level=logging.WARNING
                ) as cm,
                ctx(),
            ):
                fn(torch.randn(16, 16), torch.randn(16, 16))

        m = None
        for log_line in cm.output:
            # Search for warning message with debug trace file path.
            m = re.match(r"WARNING.* debug trace: (.*)", log_line)
            if m:
                break
        self.assertTrue(m, "debug trace file path not found in logs")
        # For type checking, have to ensure it's not none.
        if m is None:
            raise AssertionError
        filename = Path(m.group(1))
        self.assertTrue(filename.is_dir())
        self.assertGreater(filesize(filename / "fx_graph_readable.py"), 512)
        self.assertGreater(filesize(filename / "fx_graph_runnable.py"), 512)
        self.assertGreater(filesize(filename / "fx_graph_transformed.py"), 512)
        self.assertGreater(filesize(filename / "output_code.py"), 1024)

        pre_fusion_logs = pre_fusion_stream.getvalue().strip()
        self.assertExpectedInline(
            pre_fusion_logs,
            """\
BEFORE FUSION
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', c0, {c0: 256})]
op0.unmet_dependencies = []
op0.met_dependencies = [MemoryDep('arg0_1', c0, {c0: 256})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cpu', torch.float32, size=[16, 16], stride=[16, 1])
    buf0.users = [NodeUser(node=SchedulerNode(name='op1'), can_inplace=True, is_weak=False)]
]
op0.group.device = cpu
op0.group.iteration = ((256,), ())
op0.sizes = ([256], [])
arg0_1_layout = FixedLayout('cpu', torch.float32, size=[16, 16], stride=[16, 1])
buf0_layout = FixedLayout('cpu', torch.float32, size=[16, 16], stride=[16, 1])
class op0_loop_body:
    var_ranges = {p0: 256}
    index0 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        constant = ops.constant(1.0, torch.float32)
        add = ops.add(load, constant)
        get_index_1 = self.get_index('index0')
        store = ops.store('buf0', get_index_1, add, None)
        return store


op1: SchedulerNode(ComputedBuffer)
op1.writes = [MemoryDep('buf1', c0, {c0: 256})]
op1.unmet_dependencies = [MemoryDep('buf0', c0, {c0: 256})]
op1.met_dependencies = []
op1.min_input_distance = 1
op1.max_input_distance = 1
op1.outputs = [
    buf1: ComputedBuffer
    buf1.layout = FixedLayout('cpu', torch.float32, size=[16, 16], stride=[16, 1])
    buf1.users = [NodeUser(node=ExternKernelSchedulerNode(name='op2'), can_inplace=False, is_weak=False)]
]
op1.group.device = cpu
op1.group.iteration = ((256,), ())
op1.sizes = ([256], [])
buf0_layout = FixedLayout('cpu', torch.float32, size=[16, 16], stride=[16, 1])
buf1_layout = FixedLayout('cpu', torch.float32, size=[16, 16], stride=[16, 1])
class op1_loop_body:
    var_ranges = {p0: 256}
    index0 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('buf0', get_index)
        constant = ops.constant(2.0, torch.float32)
        add = ops.add(load, constant)
        get_index_1 = self.get_index('index0')
        store = ops.store('buf1', get_index_1, add, None)
        return store


op2: ExternKernelSchedulerNode(ExternKernelOut)
op2.writes = [StarDep(name='buf2', mode=None)]
op2.unmet_dependencies = [StarDep(name='buf1', mode=None)]
op2.met_dependencies = [StarDep(name='arg1_1', mode=None)]
op2.min_input_distance = 2
op2.max_input_distance = 2
op2.outputs = [
    buf2: ExternKernelOut
    buf2.layout = FixedLayout('cpu', torch.float32, size=[16, 16], stride=[16, 1])
    buf2.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op2.node.kernel = extern_kernels.mm""",
        )

        post_fusion_logs = post_fusion_stream.getvalue().strip()
        self.assertExpectedInline(
            post_fusion_logs,
            """\
AFTER FUSION
op0_op1: FusedSchedulerNode(SchedulerNode,SchedulerNode)
op0_op1.writes = [MemoryDep('buf0', c0, {c0: 256}), MemoryDep('buf1', c0, {c0: 256})]
op0_op1.unmet_dependencies = []
op0_op1.met_dependencies = [MemoryDep('arg0_1', c0, {c0: 256})]
op0_op1.min_input_distance = 0
op0_op1.max_input_distance = 1
op0_op1.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cpu', torch.float32, size=[16, 16], stride=[16, 1])
    buf0.users = [NodeUser(node=SchedulerNode(name='op1'), can_inplace=True, is_weak=False)]
    buf1: ComputedBuffer
    buf1.layout = FixedLayout('cpu', torch.float32, size=[16, 16], stride=[16, 1])
    buf1.users = [NodeUser(node=ExternKernelSchedulerNode(name='op2'), can_inplace=False, is_weak=False)]
]
op0_op1.snodes[0] =
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', c0, {c0: 256})]
op0.unmet_dependencies = []
op0.met_dependencies = [MemoryDep('arg0_1', c0, {c0: 256})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cpu', torch.float32, size=[16, 16], stride=[16, 1])
    buf0.users = [NodeUser(node=SchedulerNode(name='op1'), can_inplace=True, is_weak=False)]
]
op0.group.device = cpu
op0.group.iteration = ((256,), ())
op0.sizes = ([256], [])
arg0_1_layout = FixedLayout('cpu', torch.float32, size=[16, 16], stride=[16, 1])
buf0_layout = FixedLayout('cpu', torch.float32, size=[16, 16], stride=[16, 1])
class op0_loop_body:
    var_ranges = {p0: 256}
    index0 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        constant = ops.constant(1.0, torch.float32)
        add = ops.add(load, constant)
        get_index_1 = self.get_index('index0')
        store = ops.store('buf0', get_index_1, add, None)
        return store
op0_op1.snodes[1] =
op1: SchedulerNode(ComputedBuffer)
op1.writes = [MemoryDep('buf1', c0, {c0: 256})]
op1.unmet_dependencies = [MemoryDep('buf0', c0, {c0: 256})]
op1.met_dependencies = []
op1.min_input_distance = 1
op1.max_input_distance = 1
op1.outputs = [
    buf1: ComputedBuffer
    buf1.layout = FixedLayout('cpu', torch.float32, size=[16, 16], stride=[16, 1])
    buf1.users = [NodeUser(node=ExternKernelSchedulerNode(name='op2'), can_inplace=False, is_weak=False)]
]
op1.group.device = cpu
op1.group.iteration = ((256,), ())
op1.sizes = ([256], [])
buf0_layout = FixedLayout('cpu', torch.float32, size=[16, 16], stride=[16, 1])
buf1_layout = FixedLayout('cpu', torch.float32, size=[16, 16], stride=[16, 1])
class op1_loop_body:
    var_ranges = {p0: 256}
    index0 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('buf0', get_index)
        constant = ops.constant(2.0, torch.float32)
        add = ops.add(load, constant)
        get_index_1 = self.get_index('index0')
        store = ops.store('buf1', get_index_1, add, None)
        return store


op2: ExternKernelSchedulerNode(ExternKernelOut)
op2.writes = [StarDep(name='buf2', mode=None)]
op2.unmet_dependencies = [StarDep(name='buf1', mode=None)]
op2.met_dependencies = [StarDep(name='arg1_1', mode=None)]
op2.min_input_distance = 2
op2.max_input_distance = 2
op2.outputs = [
    buf2: ExternKernelOut
    buf2.layout = FixedLayout('cpu', torch.float32, size=[16, 16], stride=[16, 1])
    buf2.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op2.node.kernel = extern_kernels.mm""",
        )
        # intentionally only cleanup on success so debugging test is easier
        shutil.rmtree(filename)

    # AOT compiler have not supported windows yet.
    @skipIfWindows
    def test_debug_printer_const(self):
        """Test that having a const example_input does not break the debug printer."""

        class Model(torch.nn.Module):
            def forward(self, x, ks0):
                return x.sum()

        example_inputs = (
            torch.tensor([0, 3, 6], dtype=torch.int64),
            70,  # const input, that will be filtered in the examples
        )
        _ = torch._export.aot_compile(
            Model(),
            example_inputs,
        )

    @unittest.skipIf(not HAS_GPU, "requires GPU")
    def test_debug_multi_tempalte(self):
        class ToyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.l = torch.nn.Linear(100, 100)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(self.l(x))

        # no failure
        with (
            self.assertLogs(
                logging.getLogger("torch._inductor.debug"),
                level=logging.WARNING,
            ),
            fresh_cache(),
        ):
            m = ToyModel().to(device=GPU_TYPE)
            m = torch.compile(m, mode="max-autotune")
            input_tensor = torch.randn(100).to(device=GPU_TYPE)
            m(input_tensor)


class TestLogAutotuningResultsCallSite(test_torchinductor.TestCase):
    def test_log_results_forwards_required_args_to_debug_formatter(self):
        # Regression: AlgorithmSelectorCache.log_results omitted
        # prescreening_elapse, which raised TypeError under
        # TORCH_COMPILE_DEBUG=1 + LOG_AUTOTUNE_RESULTS=1 (those flags route
        # V.debug to the real DebugFormatter, whose signature requires it).
        import inspect

        from torch._inductor.debug import DebugFormatter
        from torch._inductor.select_algorithm import AlgorithmSelectorCache
        from torch._inductor.virtualized import V

        captured: dict = {}

        class _Recorder:
            def log_autotuning_results(self, *args, **kwargs):
                captured["args"] = args
                captured["kwargs"] = kwargs

        with V.set_debug_handler(_Recorder()):
            AlgorithmSelectorCache.log_results(
                name="test_op",
                input_nodes=[],
                timings={},
                elapse=0.0,
                precompile_elapse=0.0,
                prescreening_elapse=None,
            )

        self.assertIn("args", captured, "V.debug.log_autotuning_results was not called")
        # Bind the captured call to the real DebugFormatter signature; missing
        # required positional arg (e.g. prescreening_elapse) raises TypeError
        # exactly like the production failure under LOG_AUTOTUNE_RESULTS=1.
        sig = inspect.signature(DebugFormatter.log_autotuning_results)
        sig.bind(_Recorder(), *captured["args"], **captured["kwargs"])


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests
    from torch.testing._internal.inductor_utils import HAS_CPU

    if HAS_CPU:
        run_tests(needs="filelock")
