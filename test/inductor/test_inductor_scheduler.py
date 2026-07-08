# Owner(s): ["module: inductor"]

from unittest import skipIf
from unittest.mock import Mock

import sympy

import torch
import torch._inductor.config as inductor_config
import torch._inductor.ir as ir
import torch._inductor.metrics as metrics
import torch.utils.flop_counter
from torch._dynamo.utils import counters
from torch._inductor.dependencies import Dep, MemoryDep, ReadWrites
from torch._inductor.ir import GraphPartitionSignature
from torch._inductor.loop_body import MemoryEntry, MemoryUsageType
from torch._inductor.scheduler import (
    _get_benchmarkable_extern_fn,
    BaseSchedulerNode,
    ExternKernelSchedulerNode,
    NestedReduction,
    Scheduler,
)
from torch._inductor.sizevars import SizeVarAllocator
from torch._inductor.utils import fresh_inductor_cache, snode_args_kwargs
from torch._inductor.virtualized import V
from torch.testing._internal.common_cuda import SM70OrLater
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyCUDA,
    skipCUDAIf,
)
from torch.testing._internal.common_utils import (
    parametrize,
    run_tests,
    skipIfXpu,
    TestCase,
    xfailIfNoAcceleratorTriton,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU, IS_BIG_GPU
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import FloorDiv


def FlopCounterMode(*args, **kwargs):
    return torch.utils.flop_counter.FlopCounterMode(*args, **kwargs, display=False)


def get_total_flops(mode):
    return sum(v for _, v in mode.flop_counts["Global"].items())


def random_tensor(size, dtype, **kwargs):
    if dtype in [torch.half, torch.bfloat16, torch.float, torch.double]:
        return torch.randn(size, dtype=dtype, **kwargs)
    elif dtype in [torch.uint8, torch.int8, torch.short, torch.int, torch.long]:
        return torch.randint(0, 100, size, dtype=dtype, **kwargs)
    else:
        raise ValueError("Unsupported data type")


def cT(device, dtype):
    def T(*shape, requires_grad=False):
        return random_tensor(
            shape, requires_grad=requires_grad, device=device, dtype=dtype
        )

    return T


inductor_metrics_log = torch._logging.getArtifactLogger(__name__, "inductor_metrics")


def _test_cases(device, dtype):
    T = cT(device, dtype)

    def composite(x, y, z):
        tmp = torch.mm(x + 10, y / 12)
        return torch.mm(tmp, z)

    def composite_relu(x, y):
        tmp = torch.mm(x, y)
        return torch.relu(tmp)

    test_cases = [
        (torch.mm, [T(4, 5), T(5, 6)], {}),
        (torch.add, [T(4, 5), T(4, 5)], {}),
        (composite, [T(5, 4), T(4, 3), T(3, 12)], {}),
        (composite_relu, [T(5, 4), T(4, 3)], {}),
    ]
    return test_cases


class TestScheduler(TestCase):
    def _extern_snode_for_op(self, op_overload, python_kernel_name):
        node = object.__new__(ir.ExternKernel)
        node.op_overload = op_overload
        node.python_kernel_name = python_kernel_name
        snode = object.__new__(ExternKernelSchedulerNode)
        snode.node = node
        return snode

    def test_get_benchmarkable_extern_fn_uses_op_overload(self):
        self.assertIsNone(_get_benchmarkable_extern_fn(Mock(spec=BaseSchedulerNode)))
        self.assertIs(
            _get_benchmarkable_extern_fn(
                self._extern_snode_for_op(torch.ops.aten.mm.out, "renamed_mm")
            ),
            torch.ops.aten.mm,
        )
        self.assertIs(
            _get_benchmarkable_extern_fn(
                self._extern_snode_for_op(
                    torch.ops.aten._scaled_mm.out, "extern_kernels.mm"
                )
            ),
            torch.ops.aten._scaled_mm,
        )
        self.assertIsNone(
            _get_benchmarkable_extern_fn(
                self._extern_snode_for_op(None, "extern_kernels.mm")
            )
        )
        self.assertIsNone(
            _get_benchmarkable_extern_fn(
                self._extern_snode_for_op(
                    torch.ops.aten.relu.out, "extern_kernels.relu"
                )
            )
        )

    def test_snode_args_kwargs_removes_filled_positional_kwargs(self):
        snode = Mock()
        snode.node = Mock()
        snode.node.inputs = [torch.empty(2, 2), torch.empty(2, 2)]
        snode.node.constant_args = ()
        snode.node.kwargs = {"out_dtype": torch.float16}
        snode.node.op_overload = torch.ops.aten.mm.dtype_out
        snode.node.fill_non_provided_args.side_effect = lambda args, kwargs: [
            *args,
            kwargs["out_dtype"],
        ]

        args, kwargs = snode_args_kwargs(snode)

        self.assertEqual(args[2], torch.float16)
        self.assertEqual(kwargs, {})

    def test_snode_args_kwargs_preserves_keyword_only_kwargs(self):
        snode = Mock()
        snode.node = Mock()
        snode.node.inputs = [
            torch.empty(2, 2),
            torch.empty(2, 2),
            torch.empty(2, 2),
        ]
        snode.node.constant_args = ()
        snode.node.kwargs = {"alpha": 2}
        snode.node.op_overload = torch.ops.aten.addmm.out
        snode.node.fill_non_provided_args.side_effect = lambda args, kwargs: args

        args, kwargs = snode_args_kwargs(snode)

        self.assertEqual(len(args), 3)
        self.assertEqual(kwargs, {"alpha": 2})

    def test_snode_args_kwargs_unflattens_fallback_kernel_args(self):
        node = object.__new__(ir.FallbackKernel)
        node.inputs = [torch.empty(2, 3), torch.empty(2, 3)]
        node.constant_args = (1,)
        node.kwargs = {}
        node.op_overload = torch.ops.aten.cat.default
        node.unflatten_args = lambda tensor_args, constant_args: (
            [list(tensor_args)],
            {"dim": constant_args[0]},
        )
        node.fill_non_provided_args = lambda args, kwargs: [*args, kwargs["dim"]]
        snode = Mock()
        snode.node = node

        args, kwargs = snode_args_kwargs(snode)

        self.assertEqual([tuple(t.shape) for t in args[0]], [(2, 3), (2, 3)])
        self.assertEqual(args[1], 1)
        self.assertEqual(kwargs, {})

    def test_fusable_read_and_write_broadcast_requires_index_equivalence(self):
        d0, d1, d2 = sympy.symbols("d0 d1 d2", integer=True, nonnegative=True)
        w0, w1 = sympy.symbols("w0 w1", integer=True, nonnegative=True)

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.mutation_renames = {}
        scheduler.mode_requires_synchronization = lambda mode: False

        graph = Mock(sizevars=SizeVarAllocator())
        with V.set_graph_handler(graph):
            write = MemoryDep("buf", 32 * w0 + w1, (w0, w1), (128, 32))
            simple_write = MemoryDep("buf", w0, (w0,), (16,))
            s0, s1 = sympy.symbols("s0 s1", integer=True, positive=True)
            exact_gapped = MemoryDep("buf", 33 * d0 + d1, (d0, d1), (128, 32))
            cases = [
                (
                    "quotient broadcast",
                    MemoryDep(
                        "buf",
                        32 * d0 + FloorDiv(d1, 128),
                        (d0, d1),
                        (128, 4096),
                    ),
                    write,
                    False,
                    True,
                ),
                (
                    "quotient tail remains",
                    MemoryDep(
                        "buf",
                        32 * d0 + FloorDiv(d1, 128) + d1,
                        (d0, d1),
                        (128, 4096),
                    ),
                    write,
                    False,
                    False,
                ),
                (
                    "pure broadcast",
                    MemoryDep("buf", d1, (d0, d1), (1024, 16)),
                    simple_write,
                    False,
                    True,
                ),
                (
                    "dynamic dense",
                    MemoryDep("buf", s1 * d0 + d1, (d0, d1), (s0, s1)),
                    MemoryDep("buf", s1 * w0 + w1, (w0, w1), (s0, s1)),
                    False,
                    True,
                ),
                (
                    "exact gapped",
                    exact_gapped,
                    exact_gapped,
                    True,
                    True,
                ),
                (
                    "producer broadcast",
                    MemoryDep("buf", d0, (d0, d1), (8, 4)),
                    MemoryDep("buf", w1, (w0, w1), (8, 4)),
                    False,
                    False,
                ),
                (
                    "producer alias",
                    MemoryDep("buf", d0 + d1, (d0, d1), (2, 2)),
                    MemoryDep("buf", w0 + w1, (w0, w1), (2, 2)),
                    False,
                    False,
                ),
            ]
            for name, read, write, expected_default, expected_relaxed in cases:
                with self.subTest(name):
                    self.assertEqual(
                        scheduler.fusable_read_and_write(read, write),
                        expected_default,
                    )
                    self.assertEqual(
                        scheduler.fusable_read_and_write(
                            read,
                            write,
                            allow_index_equivalence=True,
                        ),
                        expected_relaxed,
                    )

            normalized_exact_gapped_read = MemoryDep(
                "buf", 33 * d0 + d1, (d0, d1, d2), (128, 32, 7)
            )
            normalized_exact_gapped_write = MemoryDep(
                "buf", 33 * w0 + w1, (w0, w1), (128, 32)
            )
            with inductor_config.patch(loop_ordering_after_fusion=True):
                self.assertTrue(
                    scheduler.fusable_read_and_write(
                        normalized_exact_gapped_read,
                        normalized_exact_gapped_write,
                    )
                )
                self.assertTrue(
                    scheduler.fusable_read_and_write(
                        normalized_exact_gapped_read,
                        normalized_exact_gapped_write,
                        allow_index_equivalence=True,
                    )
                )

    def test_nested_reduction_grouped_axis_from_ranges(self):
        grouped = Mock()
        graph = Mock(sizevars=SizeVarAllocator())

        with V.set_graph_handler(graph):
            grouped.get_ranges.return_value = ([128, 32], [16])
            self.assertEqual(
                NestedReduction.get_grouped_axis(
                    grouped,
                    outer_numel=128,
                    outer_rnumel=512,
                    group_size=16,
                ),
                NestedReduction.GroupedAxis.R,
            )

            grouped.get_ranges.return_value = ([8, 512], [16])
            self.assertEqual(
                NestedReduction.get_grouped_axis(
                    grouped,
                    outer_numel=128,
                    outer_rnumel=512,
                    group_size=16,
                ),
                NestedReduction.GroupedAxis.X,
            )

            grouped.get_ranges.return_value = ([32], [16])
            self.assertEqual(
                NestedReduction.get_grouped_axis(
                    grouped,
                    outer_numel=1,
                    outer_rnumel=512,
                    group_size=16,
                ),
                NestedReduction.GroupedAxis.R,
            )

            grouped.get_ranges.return_value = ([512], [16])
            self.assertEqual(
                NestedReduction.get_grouped_axis(
                    grouped,
                    outer_numel=16,
                    outer_rnumel=512,
                    group_size=16,
                ),
                NestedReduction.GroupedAxis.X,
            )

            grouped.get_ranges.return_value = ([32, 128], [16])
            self.assertIsNone(
                NestedReduction.get_grouped_axis(
                    grouped,
                    outer_numel=128,
                    outer_rnumel=512,
                    group_size=16,
                )
            )

            grouped.get_ranges.return_value = ([4096], [16])
            self.assertIsNone(
                NestedReduction.get_grouped_axis(
                    grouped,
                    outer_numel=128,
                    outer_rnumel=512,
                    group_size=16,
                )
            )

    def test_nested_reduction_axis_from_loop_body(self):
        outer_x0, outer_x1, outer_r = sympy.symbols("outer_x0 outer_x1 outer_r")
        grouped_x0, grouped_x1, grouped_r = sympy.symbols(
            "grouped_x0 grouped_x1 grouped_r"
        )

        def make_body(index, iter_vars, reduce_vars):
            body = Mock()
            body.iter_vars = iter_vars
            body.reduce_vars = reduce_vars
            body.indexing_exprs = {"load": index}
            body.memory_usage = {
                MemoryUsageType.LOAD: [MemoryEntry("load", "arg0_1", None)]
            }
            return body

        def make_reduction(index, iter_vars, reduce_vars):
            node = Mock()
            node.is_reduction.return_value = True
            node.get_ranges.return_value = ([16, 16], [16])
            node._body = make_body(index, iter_vars, reduce_vars)
            return node

        def classify(outer_index, grouped_index):
            outer = make_reduction(outer_index, (outer_x0, outer_x1), (outer_r,))
            grouped = make_reduction(
                grouped_index, (grouped_x0, grouped_x1), (grouped_r,)
            )
            outer_node = Mock()
            outer_node.get_nodes.return_value = [outer]
            return NestedReduction._get_grouped_axis_from_loop_body(outer_node, grouped)

        self.assertEqual(
            classify(
                256 * outer_x0 + 16 * outer_x1 + outer_r,
                256 * grouped_x0 + 16 * grouped_x1 + grouped_r,
            ),
            NestedReduction.GroupedAxis.R,
        )
        self.assertEqual(
            classify(
                outer_x0 + 16 * outer_x1 + 256 * outer_r,
                grouped_x0 + 16 * grouped_x1 + 256 * grouped_r,
            ),
            NestedReduction.GroupedAxis.R,
        )
        self.assertEqual(
            classify(
                256 * outer_x0 + 16 * outer_x1 + outer_r,
                256 * grouped_x0 + grouped_x1 + 16 * grouped_r,
            ),
            NestedReduction.GroupedAxis.X,
        )
        self.assertEqual(
            classify(
                outer_x0 + 16 * outer_x1 + outer_r,
                grouped_x0 + 16 * grouped_x1 + grouped_r,
            ),
            None,
        )

    def test_partition_signature_cleaning_only_removes_current_codegen_buffers(self):
        scheduler = Scheduler.__new__(Scheduler)

        live_input = Mock()
        preexisting_removed_input = Mock()
        codegen_removed_input = Mock()

        live_output = Mock()
        live_output.maybe_get_name.return_value = "live_output"
        preexisting_removed_output = Mock()
        preexisting_removed_output.maybe_get_name.return_value = (
            "preexisting_removed_output"
        )
        codegen_removed_output = Mock()
        codegen_removed_output.maybe_get_name.return_value = "codegen_removed_output"

        signature = GraphPartitionSignature(
            symbol_inputs=OrderedSet(),
            input_nodes={
                "live_input": live_input,
                "preexisting_removed_input": preexisting_removed_input,
                "codegen_removed_input": codegen_removed_input,
            },
            output_nodes=[
                live_output,
                preexisting_removed_output,
                codegen_removed_output,
            ],
            input_deallocation={
                "live_input": False,
                "preexisting_removed_input": True,
                "codegen_removed_input": False,
            },
            skip_cudagraph=False,
            constant_names=[
                "live_constant",
                "preexisting_removed_constant",
                "codegen_removed_constant",
            ],
        )

        removed_buffers_before_codegen = OrderedSet(
            [
                "preexisting_removed_input",
                "preexisting_removed_output",
                "preexisting_removed_constant",
            ]
        )
        removed_buffers_after_codegen = removed_buffers_before_codegen | OrderedSet(
            [
                "codegen_removed_input",
                "codegen_removed_output",
                "codegen_removed_constant",
            ]
        )
        removed_buffers_during_codegen = (
            removed_buffers_after_codegen - removed_buffers_before_codegen
        )

        cleaned = scheduler.clean_removed_buffer_from_partition_signatures(
            signature, removed_buffers_during_codegen
        )

        self.assertEqual(
            cleaned.input_nodes,
            {
                "live_input": live_input,
                "preexisting_removed_input": preexisting_removed_input,
            },
        )
        self.assertEqual(
            cleaned.input_deallocation,
            {"live_input": False, "preexisting_removed_input": True},
        )
        self.assertEqual(
            cleaned.output_nodes,
            [live_output, preexisting_removed_output],
        )
        self.assertEqual(
            cleaned.constant_names,
            ["live_constant", "preexisting_removed_constant"],
        )
        self.assertFalse(cleaned.skip_cudagraph)

    @dtypes(torch.float, torch.float16)
    @skipCUDAIf(not SM70OrLater, "GPU capability is < SM70")
    @xfailIfNoAcceleratorTriton
    def test_disable_get_estimated_runtime_logging(self, device, dtype):
        if device == "cpu":
            return
        tc = _test_cases(device, dtype)
        # turn off logging of inductor metrics so that they don't get logged
        torch._logging.set_logs(inductor_metrics=False)
        metrics.reset()
        for op, example_inputs, kwargs in tc:
            comp = torch.compile(op)
            torch._dynamo.reset()
            with fresh_inductor_cache():
                comp(*example_inputs, **kwargs)
            self.assertEqual(metrics.num_bytes_accessed, 0)
            self.assertEqual(any(m[1] for m in metrics.node_runtimes), False)
            self.assertEqual(any(m[1] for m in metrics.nodes_num_elem), False)
            metrics.reset()
        torch._logging.set_logs()

    @xfailIfNoAcceleratorTriton
    @skipIfXpu(
        msg="InvalidModule: Invalid SPIR-V module, "
        "https://github.com/intel/torch-xpu-ops/issues/2329"
    )
    @dtypes(torch.float, torch.float16)
    @skipCUDAIf(not SM70OrLater, "GPU capability is < SM70")
    @parametrize(
        "options",
        [
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
            },
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON,ATEN",
            },
        ],
    )
    @torch._inductor.config.patch(
        {"force_disable_caches": True, "shape_padding": False}
    )
    @skipIf(not IS_BIG_GPU, "we can't use Triton only as a backend for max autotune")
    def test_flop_counter_op(self, device, dtype, options):
        if device == "cpu":
            return

        tc = _test_cases(device, dtype)

        torch._logging.set_logs(inductor_metrics=True)
        for op, example_inputs, kwargs in tc:
            comp = torch.compile(op, options=options)
            # next two lines are required, otherwise the flops will be cached from previous runs of this function.
            torch._dynamo.reset()
            with fresh_inductor_cache():
                # actually run to set the counters
                comp(*example_inputs, **kwargs)
                with FlopCounterMode() as mode:
                    comp(*example_inputs, **kwargs)
            reference_flops = get_total_flops(mode)

            self.assertEqual(
                reference_flops,
                counters["inductor"]["flop_count"],
                msg=lambda msg: f"{msg}\nop = {op} reference flops = {reference_flops} != counters {counters['inductor']['flop_count']}",
            )
            if op != torch.add:
                self.assertNotEqual(
                    reference_flops, 0, msg=lambda msg: f"{msg}\nop = {op} is 0 flops"
                )
            counters["inductor"]["flop_count"] = 0
        torch._logging.set_logs()

    def test_fusion_prevent_too_many_reads_and_writes_prevents_fusion(self):
        """Test that fusion is prevented when unique I/O buffers exceed threshold"""
        # Setup: Create nodes with many unique I/O buffers
        # node1: reads [A, B, C], writes [D]
        # node2: reads [D, E, F], writes [G]
        # D becomes internal (node2 reads node1's write)
        # After fusion: unique I/O = {A, B, C, E, F, G} = 6 buffers
        scheduler = Mock(spec=Scheduler)
        scheduler.can_buffer_be_removed_through_fusion = Mock(return_value=False)

        node1 = self._create_mock_node(
            name="node1", reads=["A", "B", "C"], writes=["D"]
        )
        node2 = self._create_mock_node(
            name="node2", reads=["D", "E", "F"], writes=["G"]
        )

        # Execute: Check with threshold of 5 (should prevent fusion since 6 > 5)
        result = Scheduler.fusion_prevent_too_many_reads_and_writes(
            scheduler, node1, node2, threshold=5
        )

        # Assert: Fusion should be prevented (6 unique buffers > 5 threshold)
        self.assertTrue(result)

    def test_fusion_prevent_too_many_reads_and_writes_allows_fusion(self):
        """Test that fusion is allowed when intermediate buffers are removed"""
        # Setup: Create nodes where node2 reads node1's output
        # node1: reads [A, B], writes [C]
        # node2: reads [C, D], writes [E]
        # C becomes internal (node2 reads node1's write)
        # After fusion: unique I/O = {A, B, D, E} = 4 buffers
        scheduler = Mock(spec=Scheduler)
        scheduler.can_buffer_be_removed_through_fusion = Mock(return_value=False)

        node1 = self._create_mock_node(name="node1", reads=["A", "B"], writes=["C"])
        node2 = self._create_mock_node(name="node2", reads=["C", "D"], writes=["E"])

        # Execute: Check with threshold of 5 (should allow fusion since 4 <= 5)
        result = Scheduler.fusion_prevent_too_many_reads_and_writes(
            scheduler, node1, node2, threshold=5
        )

        # Assert: Fusion should be allowed (4 unique buffers <= 5 threshold)
        self.assertFalse(result)

    def _create_mock_node(self, name: str, reads: list[str], writes: list[str]) -> Mock:
        """Helper method to create a mock scheduler node with specified reads/writes"""
        node = Mock(spec=BaseSchedulerNode)
        node.get_name = Mock(return_value=name)
        node.get_nodes = Mock(return_value=[node])

        # Create mock Dep objects for reads and writes
        read_deps = OrderedSet()
        for read_name in reads:
            dep = Mock(spec=Dep)
            dep.name = read_name
            read_deps.add(dep)

        write_deps = OrderedSet()
        for write_name in writes:
            dep = Mock(spec=Dep)
            dep.name = write_name
            write_deps.add(dep)

        # Create mock ReadWrites object
        read_writes = Mock(spec=ReadWrites)
        read_writes.reads = read_deps
        read_writes.writes = write_deps

        node.read_writes = read_writes
        return node

    def test_prologue_fusion_uses_template_aliasing_hook(self):
        def make_prologue_and_template(hook_blocks: bool):
            prologue_node = Mock()
            template_node = Mock()
            template = Mock()

            prologue_node.get_name.return_value = "prologue"
            template_node.get_name.return_value = "template"
            prologue_node.is_template.return_value = False
            template_node.is_template.return_value = True
            prologue_node.is_reduction.return_value = False
            prologue_node.ancestors = OrderedSet()
            template_node.ancestors = OrderedSet(["prologue"])
            prologue_node.get_operation_names.return_value = OrderedSet(["prologue"])
            template_node.get_operation_names.return_value = OrderedSet(["template"])
            prologue_node.get_buffer_names.return_value = OrderedSet(["x"])
            template_node.get_buffer_names.return_value = OrderedSet(["out"])
            prologue_node.get_device.return_value = torch.device("cpu")
            template_node.get_device.return_value = torch.device("cpu")
            prologue_node.has_aliasing_or_mutation.return_value = False
            template_node.has_aliasing_or_mutation.return_value = True

            input_node = Mock()
            input_node.get_name.return_value = "x"
            template.inputs = [input_node]
            template.allow_prologue_fusion = True
            template.get_allowed_prologue_inps.return_value = OrderedSet(["x"])
            template.has_aliasing_or_mutation_for_prologue_fusion.return_value = (
                hook_blocks
            )
            template_node.get_template_node.return_value = template
            template_node.get_template_node_or_throw.return_value = template

            user = Mock()
            user.node = template_node
            output = Mock()
            output.users = [user]
            prologue_node.outputs = [output]
            prologue_node.get_nodes.return_value = [prologue_node]

            return prologue_node, template_node, template

        def can_fuse_prologue(hook_blocks: bool) -> bool:
            scheduler = Scheduler.__new__(Scheduler)
            scheduler.mutation_renames = {}
            scheduler._has_multi_stream_nodes = Mock(return_value=False)
            scheduler._nested_index_equivalent_dep_names = Mock(
                return_value=OrderedSet()
            )
            scheduler._score_fusion_memory_for_can_fuse = Mock(return_value=1_000_000)
            scheduler.check_prologue_fusion_heuristics_fusable = Mock(return_value=True)
            scheduler.can_fuse_vertical = Mock(return_value=True)
            backend = Mock()
            backend.can_fuse_vertical.return_value = True
            backend.can_fuse_horizontal.return_value = True
            scheduler.get_backend = Mock(return_value=backend)

            choices = Mock()
            choices.can_fuse.return_value = True
            choices.can_fuse_vertical.return_value = True
            choices.can_fuse_horizontal.return_value = True

            graph = Mock()
            graph.no_fuse_buffer_names = OrderedSet()

            prologue_node, template_node, template = make_prologue_and_template(
                hook_blocks
            )
            with V.set_graph_handler(graph), V.set_choices_handler(choices):
                result = Scheduler._can_fuse(scheduler, prologue_node, template_node)

            template.has_aliasing_or_mutation_for_prologue_fusion.assert_called_once_with(
                template_node
            )
            template_node.has_aliasing_or_mutation.assert_not_called()
            return result

        self.assertTrue(can_fuse_prologue(hook_blocks=False))
        self.assertFalse(can_fuse_prologue(hook_blocks=True))

    @xfailIfNoAcceleratorTriton
    @onlyCUDA
    def test_index_add_fusion_prevented(self):
        """
        Test that index_add_ (scatter with atomic_add mode) is not fused with
        subsequent reads from the same buffer, preventing read-after-write hazards.

        Regression test for: index_add_ followed by indexing was incorrectly fused,
        causing reads to occur before atomic writes completed.
        """

        def fn(f, batch):
            # Scatter: atomic writes to shared location
            f_u = f**2 + 0.00987654321
            n_batch = batch.max() + 1
            F_u_mol = torch.zeros((n_batch, f.shape[1]), device=f.device, dtype=f.dtype)
            F_u_mol.index_add_(0, batch, f_u)

            # Gather: reads from same buffer (requires synchronization)
            F_u_at_atom = F_u_mol[batch] + 1e-6
            return f_u / F_u_at_atom

        device = "cuda"
        f = torch.ones(1024, 1, device=device)
        batch = torch.zeros(1024, dtype=torch.long, device=device)

        # Eager execution (ground truth)
        eager_result = fn(f, batch)

        # Compiled execution (should match eager)
        compiled_fn = torch.compile(fn)
        compiled_result = compiled_fn(f, batch)

        # Verify results match (no fusion bug)
        self.assertTrue(
            torch.allclose(eager_result, compiled_result, rtol=1e-4, atol=1e-4),
            msg=f"index_add_ fusion bug detected: "
            f"eager={eager_result.mean().item():.6f}, "
            f"compiled={compiled_result.mean().item():.6f}",
        )

    @xfailIfNoAcceleratorTriton
    @onlyCUDA
    def test_atomic_add_no_fusion_correctness(self):
        """
        Test that atomic_add operations produce correct results.
        """

        def fn(x, idx):
            out = torch.zeros(10, device=x.device)
            out.index_add_(0, idx, x)  # atomic_add: scatter to shared locations
            return out[idx] + 1.0  # read from same buffer: requires sync

        device = "cuda"
        x = torch.ones(5, device=device)
        idx = torch.tensor([0, 1, 0, 1, 0], device=device, dtype=torch.long)

        # Eager (correct) result
        expected = fn(x, idx)

        # Compiled result: will be wrong if fusion bug exists
        compiled_fn = torch.compile(fn)
        torch._dynamo.reset()
        with fresh_inductor_cache():
            result = compiled_fn(x, idx)

        # This test will FAIL without the fusion prevention fix
        self.assertTrue(
            torch.allclose(expected, result),
            msg=f"Fusion bug detected! Expected {expected}, got {result}",
        )

    @xfailIfNoAcceleratorTriton
    @onlyCUDA
    def test_expand_reuse_does_not_realize_before_reduction(self):
        def fn(icrd1, icrd2, wcrd, ocrd, meta, input1, input2, weight, output):
            input1_selected = torch.index_select(input1, 2, icrd1)
            input2_selected = torch.index_select(input2, 2, icrd2)
            weight_selected = torch.index_select(weight, 3, wcrd)

            input1_expanded = input1_selected.view(B, U, 1, 1, -1)
            input2_expanded = input2_selected.view(B, 1, V, 1, -1)
            weight_expanded = weight_selected.view(1, U, V, W, -1)
            meta_expanded = meta.view(1, 1, 1, 1, -1)

            product = (
                meta_expanded * input1_expanded * input2_expanded * weight_expanded
            )
            product = torch.sum(product, dim=(1, 2))
            output.index_add_(2, ocrd, product)
            return output

        P = 20
        M = 10
        B = 10
        L = 23
        U = 4
        V = 4
        W = 4
        device = "cuda"

        torch.manual_seed(0)
        input1 = torch.rand((B, U, L), dtype=torch.float32, device=device)
        input2 = torch.rand((B, V, L), dtype=torch.float32, device=device)
        weight = torch.rand((U, V, W, M), dtype=torch.float32, device=device)
        output = torch.zeros((B, W, L), dtype=torch.float32, device=device)
        meta = torch.rand((P,), dtype=torch.float32, device=device)
        icrd1 = torch.randint(L, (P,), device=device)
        icrd2 = torch.randint(L, (P,), device=device)
        wcrd = torch.randint(M, (P,), device=device)
        ocrd = torch.arange(P, device=device)

        expected = fn(
            icrd1,
            icrd2,
            wcrd,
            ocrd,
            meta,
            input1,
            input2,
            weight,
            output.clone(),
        )

        torch._dynamo.reset()
        metrics.reset()
        with fresh_inductor_cache():
            actual = torch.compile(fn, backend="inductor", fullgraph=True)(
                icrd1,
                icrd2,
                wcrd,
                ocrd,
                meta,
                input1,
                input2,
                weight,
                output.clone(),
            )

        self.assertTrue(torch.allclose(expected, actual, atol=1e-4, rtol=1e-4))
        self.assertEqual(metrics.ir_nodes_pre_fusion, 2)
        self.assertEqual(metrics.generated_kernel_count, 1)

    @xfailIfNoAcceleratorTriton
    @onlyCUDA
    def test_expand_reuse_realizes_in_deterministic_mode(self):
        def fn(a, b, c, d, e):
            x = a * b * c * d * e
            y = x.view(8, 8, 1).expand(8, 8, 16)
            return y.sum(dim=1)

        def check_realizes():
            torch._dynamo.reset()
            metrics.reset()
            with fresh_inductor_cache():
                actual = torch.compile(fn, backend="inductor", fullgraph=True)(*args)

            self.assertTrue(torch.allclose(expected, actual, atol=1e-4, rtol=1e-4))
            self.assertEqual(metrics.ir_nodes_pre_fusion, 2)
            self.assertEqual(metrics.generated_kernel_count, 2)

        device = "cuda"
        torch.manual_seed(0)
        args = [
            torch.rand((8, 8), dtype=torch.float32, device=device) for _ in range(5)
        ]
        expected = fn(*args)

        prev_deterministic = torch.are_deterministic_algorithms_enabled()
        prev_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            check_realizes()
        finally:
            torch.use_deterministic_algorithms(
                prev_deterministic, warn_only=prev_warn_only
            )

        with inductor_config.patch(deterministic=True):
            check_realizes()


class TestScoreFusionMemory(TestCase):
    """
    Tests for _score_fusion_memory_by_buffer_overlap.

    These tests validate the fusion scoring logic that determines when nodes
    should be fused together based on their memory access patterns.

    Key scenarios:
    1. Exact matches: read/write has exact matches → should fuse (1 kernel)
    2. Large overlap (split/cat): reads on different offset but overlap is huge
       → should fuse because the benefit is large (1 kernel)
    3. Small overlap: reads on different offset but overlap is small → don't fuse (2 kernels)
    """

    @skipIf(not HAS_GPU, "GPU not available")
    @inductor_config.patch("score_fusion_memory_threshold", 1)
    @inductor_config.patch("min_overlap_ratio", 0.5)
    def test_exact_same_reads_should_fuse(self) -> None:
        """
        Case 1: Exact matches in read/write → should fuse into 1 kernel.

        Two operations reading from the exact same input tensor should be
        fused together since they can share the data read from memory.
        """

        def exact_reads(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # Both operations read the exact same input
            out1 = x * 2
            out2 = x + 1
            return out1, out2

        torch._dynamo.reset()
        metrics.reset()

        x = torch.randn(8, 512, device=GPU_TYPE, dtype=torch.float16)

        compiled_fn = torch.compile(exact_reads, backend="inductor", fullgraph=True)
        out1_eager, out2_eager = exact_reads(x)
        out1_compiled, out2_compiled = compiled_fn(x)

        self.assertTrue(torch.allclose(out1_eager, out1_compiled, atol=1e-3, rtol=1e-3))
        self.assertTrue(torch.allclose(out2_eager, out2_compiled, atol=1e-3, rtol=1e-3))
        # Should fuse into 1 kernel since both ops read exact same buffer
        self.assertEqual(metrics.generated_kernel_count, 1)

    @skipIf(not HAS_GPU, "GPU not available")
    @inductor_config.patch("score_fusion_memory_threshold", 1)
    @inductor_config.patch("min_overlap_ratio", 0.5)
    def test_split_cat_large_overlap_should_fuse(self) -> None:
        """
        Case 2: Reads on different offset but overlap is huge (split/cat) → should fuse into 1 kernel.

        Split operations read from the same input buffer at different offsets.
        Since the overlap is large (same underlying buffer), fusing these
        operations together saves reads and kernel launches.
        """

        def split_and_process(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            s1, s2, s3, s4 = torch.split(x, x.shape[-1] // 4, dim=-1)
            out1 = torch.cat([s4, s3], dim=-1)
            out2 = torch.cat([s2, s1], dim=-1)
            return out1, out2

        torch._dynamo.reset()
        metrics.reset()

        x = torch.randn(8, 512, device=GPU_TYPE, dtype=torch.float16)

        compiled_fn = torch.compile(
            split_and_process, backend="inductor", fullgraph=True
        )
        out1_eager, out2_eager = split_and_process(x)
        out1_compiled, out2_compiled = compiled_fn(x)

        self.assertTrue(torch.allclose(out1_eager, out1_compiled, atol=1e-3, rtol=1e-3))
        self.assertTrue(torch.allclose(out2_eager, out2_compiled, atol=1e-3, rtol=1e-3))
        # Should fuse into 1 kernel since all ops read from the same underlying buffer
        self.assertEqual(metrics.generated_kernel_count, 1)

    @skipIf(not HAS_GPU, "GPU not available")
    @inductor_config.patch("score_fusion_memory_threshold", 1)
    def test_partial_overlap_below_threshold(self) -> None:
        """
        Case 3: Partial overlap below the 0.5 threshold → should NOT fuse (2 kernels).

        Similar to test_split_cat_large_overlap_should_fuse, but each operation
        also reads from a separate large tensor, making the shared buffer portion
        less than 50% of total reads.

        Example scenario:
        - Split x into 4 slices: s1, s2, s3, s4 (each 25% of x)
        - op1 reads: s1 (from x, ~25%) + y (separate tensor, ~75%) → total 100%
        - op2 reads: s2 (from x, ~25%) + z (separate tensor, ~75%) → total 100%
        - Common buffer is x, but each op only reads 25% of their total from x
        - overlap_ratio = 25% / 100% = 0.25 < 0.5 threshold → score = 0
        - Result: 2 separate kernels (not fused)
        """

        def partial_overlap_split(
            x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # Split x into 4 parts, use different slices in each output
            s1, s2, _, _ = torch.split(x, x.shape[-1] // 4, dim=-1)
            # op1 reads: s1 (small slice of x) + y (large separate tensor)
            # op2 reads: s2 (small slice of x) + z (large separate tensor)
            # The slices s1 and s2 come from the same buffer x,
            # but each is only ~25% of total reads for that op
            out1 = torch.cat([s1, y, y, y], dim=-1)
            out2 = torch.cat([s2, z, z, z], dim=-1)
            return out1, out2

        torch._dynamo.reset()
        metrics.reset()

        # x is split into 4 parts (each 128 elements)
        # y and z are 3x larger (384 elements each)
        # So each op reads: 128 (from x slice) + 384 (from y or z) = 512 total
        # overlap_ratio = 128 / 512 = 0.25 < 0.5 threshold
        x = torch.randn(8, 512, device=GPU_TYPE, dtype=torch.float16)
        y = torch.randn(8, 128, device=GPU_TYPE, dtype=torch.float16)
        z = torch.randn(8, 128, device=GPU_TYPE, dtype=torch.float16)

        compiled_fn = torch.compile(
            partial_overlap_split, backend="inductor", fullgraph=True
        )
        out1_eager, out2_eager = partial_overlap_split(x, y, z)
        out1_compiled, out2_compiled = compiled_fn(x, y, z)

        self.assertTrue(torch.allclose(out1_eager, out1_compiled, atol=1e-3, rtol=1e-3))
        self.assertTrue(torch.allclose(out2_eager, out2_compiled, atol=1e-3, rtol=1e-3))
        # Should NOT fuse (2 kernels) because overlap_ratio = 0.25 < 0.5 threshold
        # The _score_fusion_memory_by_buffer_overlap returns 0 for this case
        self.assertEqual(metrics.generated_kernel_count, 2)


instantiate_device_type_tests(TestScheduler, globals(), allow_xpu=True)
instantiate_device_type_tests(TestScoreFusionMemory, globals(), allow_xpu=True)

if __name__ == "__main__":
    run_tests()
