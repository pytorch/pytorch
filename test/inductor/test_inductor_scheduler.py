# Owner(s): ["module: inductor"]

import contextlib
from unittest import skipIf
from unittest.mock import Mock, patch, PropertyMock

import sympy

import torch
import torch._inductor.config as inductor_config
import torch._inductor.ir as ir
import torch._inductor.metrics as metrics
import torch.utils.flop_counter
from torch._dynamo.utils import counters
from torch._inductor.codegen.common import CSEVariable
from torch._inductor.codegen.simd import (
    _GroupedReductionLayout,
    _PointwiseRemapHandler,
    _SubParentValueResolver,
    SIMDScheduling,
)
from torch._inductor.codegen.simd_kernel_features import (
    DisableReduction,
    EnableReduction,
)
from torch._inductor.dependencies import Dep, MemoryDep, ReadWrites, StarDep, WeakDep
from torch._inductor.ir import GraphPartitionSignature
from torch._inductor.loop_body import MemoryEntry, MemoryUsageType
from torch._inductor.scheduler import (
    _get_benchmarkable_extern_fn,
    BaseSchedulerNode,
    ExternKernelSchedulerNode,
    ForeachKernelSchedulerNode,
    FusedNestedReductions,
    MemoryDepMatch,
    NestedReduction,
    OrderedParentNodes,
    Scheduler,
    SchedulerNode,
    SubParentAccessRelation,
    SubParentEpilogueCandidate,
    SubParentEpilogueGrouping,
    SubParentOutputGroup,
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
    DeterministicGuard,
    parametrize,
    run_tests,
    TestCase,
    xfailIfNoAcceleratorTriton,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU, IS_BIG_GPU
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.symbol import make_symbol, SymT
from torch.utils._sympy.value_ranges import ValueRanges


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
    def _mock_base_snode(self, name, device=None):
        node = Mock()
        node.get_name.return_value = name
        node.get_first_name.return_value = name
        node.get_device.return_value = device
        node.get_nodes.return_value = [node]
        node.get_buffer_names.return_value = OrderedSet()
        node.used_buffer_names.return_value = OrderedSet()
        node.is_template.return_value = False
        node.is_reduction.return_value = False
        return node

    def _extern_snode_for_op(self, op_overload, python_kernel_name):
        node = object.__new__(ir.ExternKernel)
        node.op_overload = op_overload
        node.python_kernel_name = python_kernel_name
        snode = object.__new__(ExternKernelSchedulerNode)
        snode.node = node
        return snode

    def _mock_schedule_node(
        self,
        name,
        *,
        reads=(),
        writes=(),
        ancestors=(),
        group=(8, 16),
        is_reduction=False,
    ):
        node = self._mock_base_snode(name)
        node.group = (None, group)
        node.ancestors = OrderedSet(ancestors)
        node.get_operation_names.return_value = OrderedSet([name])
        node.get_buffer_names.return_value = OrderedSet(writes)
        node.is_reduction.return_value = is_reduction
        if is_reduction:
            node.__class__ = SchedulerNode
            node.node = Mock(spec=ir.ComputedBuffer)
            node.node.data = Mock()

        def make_dep(dep_name):
            return MemoryDep(dep_name, sympy.S.Zero, (), ())

        node.read_writes = ReadWrites(
            OrderedSet(make_dep(dep_name) for dep_name in reads),
            OrderedSet(make_dep(dep_name) for dep_name in writes),
            OrderedSet(),
        )
        return node

    def _make_sub_parent_value_resolver(
        self,
        access_relations,
        *,
        kernel=None,
        family=None,
        factor=2,
    ):
        """Build a resolver with only the collaborators relevant to a unit case."""
        if kernel is None:
            kernel = Mock(_load_mask=None, _load_other=None)
        return _SubParentValueResolver(
            Mock(),
            kernel,
            Mock(),
            Mock() if family is None else family,
            access_relations=access_relations,
            sub_parent_factor=factor,
        )

    def test_generate_node_schedule_required_boundary_discards_optional_split(self):
        first = self._mock_schedule_node(
            "first", reads=("x",), writes=("a",), is_reduction=True
        )
        second = self._mock_schedule_node("second", reads=("y",), writes=("b",))
        final = self._mock_schedule_node("final", reads=("z",), writes=("c",))

        schedule = SIMDScheduling(None).generate_node_schedule(
            [first, second, final], 8, 16, required_post_reduction_index=2
        )

        self.assertEqual(
            schedule, [first, second, DisableReduction, EnableReduction, final]
        )

    def test_generate_node_schedule_required_boundary_reuses_enable_marker(self):
        first = self._mock_schedule_node("first", is_reduction=True)
        outside = self._mock_schedule_node("outside", group=(8, 1))
        final = self._mock_schedule_node("final")

        schedule = SIMDScheduling(None).generate_node_schedule(
            [first, outside, final], 8, 16, required_post_reduction_index=2
        )

        self.assertEqual(
            schedule, [first, DisableReduction, outside, EnableReduction, final]
        )

    def test_generate_node_schedule_required_boundary_reuses_final_loop(self):
        reduction = self._mock_schedule_node("reduction", is_reduction=True)
        post_reduction = self._mock_schedule_node(
            "post_reduction", ancestors=("reduction",)
        )
        final = self._mock_schedule_node("final")

        schedule = SIMDScheduling(None).generate_node_schedule(
            [reduction, post_reduction, final],
            8,
            16,
            required_post_reduction_index=2,
        )

        self.assertEqual(
            schedule,
            [reduction, DisableReduction, EnableReduction, post_reduction, final],
        )

    def test_generate_node_schedule_rejects_invalid_required_boundary(self):
        first = self._mock_schedule_node("first")
        outside = self._mock_schedule_node("outside", group=(8, 1))

        with self.assertRaisesRegex(AssertionError, "unique main-body node"):
            SIMDScheduling(None).generate_node_schedule(
                [first, outside], 8, 16, required_post_reduction_index=1
            )

        second = self._mock_schedule_node("second")
        with self.assertRaisesRegex(AssertionError, "follow a reduction loop"):
            SIMDScheduling(None).generate_node_schedule(
                [first, second], 8, 16, required_post_reduction_index=1
            )

        reduction = self._mock_schedule_node("reduction", is_reduction=True)
        later_reduction = self._mock_schedule_node("later_reduction", is_reduction=True)
        with self.assertRaisesRegex(AssertionError, "main-body pointwise nodes"):
            SIMDScheduling(None).generate_node_schedule(
                [reduction, second, later_reduction],
                8,
                16,
                required_post_reduction_index=1,
            )

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

    def test_fuse_two_nodes_propagates_mempool(self):
        scheduler = object.__new__(Scheduler)
        device = torch.device("cuda", 0)
        node1 = self._mock_base_snode("node1", device)
        node2 = self._mock_base_snode("node2", device)
        node3 = self._mock_base_snode("node3", device)
        node3.get_nodes.return_value = [node1, node2]
        backend = Mock()
        backend.fuse.return_value = node3
        scheduler.get_backend = Mock(return_value=backend)
        scheduler.node_to_stream = {node1: 0, node2: 0}
        scheduler.node_to_mempool = {node1: (7, 0), node2: (7, 0)}
        scheduler.name_to_fused_node = {}
        fused_nodes = OrderedSet([node1, node2])

        self.assertIs(
            Scheduler.fuse_two_nodes(scheduler, node1, node2, fused_nodes), node3
        )

        self.assertEqual(scheduler.node_to_mempool[node3], (7, 0))
        self.assertEqual(scheduler.node_to_stream[node3], 0)
        self.assertIn(node3, fused_nodes)
        self.assertNotIn(node1, fused_nodes)
        self.assertNotIn(node2, fused_nodes)

    def test_nested_reduction_fuse_with_propagates_mempool(self):
        scheduler = object.__new__(Scheduler)
        node1 = self._mock_base_snode("node1")
        node2 = self._mock_base_snode("node2")
        other = self._mock_base_snode("other")
        grouped_node = self._mock_base_snode("grouped_node")
        stage = Mock()
        plan = Mock(nested_stage=stage)
        scheduler.node_to_mempool = {node2: (7, 0)}

        nested = object.__new__(FusedNestedReductions)
        nested.scheduler = scheduler
        nested.node1 = node1
        nested.node2 = node2
        with (
            patch.object(
                FusedNestedReductions,
                "_plan_append",
                return_value=(grouped_node, plan),
            ),
            patch.object(FusedNestedReductions, "__init__", return_value=None),
        ):
            FusedNestedReductions.fuse_with(nested, other)

        self.assertEqual(scheduler.node_to_mempool[grouped_node], (7, 0))

    def test_nested_reduction_append_requires_complete_domains(self):
        scheduler = Mock()
        nested = object.__new__(FusedNestedReductions)
        nested.scheduler = scheduler
        nested.node1 = self._mock_base_snode("parent")
        nested.node2 = self._mock_base_snode("grouped")
        nested.node2.get_operation_names.return_value = OrderedSet(["grouped"])

        first = self._mock_base_snode("first")
        second = self._mock_base_snode("second")
        other = self._mock_base_snode("other")
        other.get_nodes.return_value = [first, second]
        other.ancestors = OrderedSet(["grouped"])
        domains = ((first, NestedReduction.PointwiseDomain.REDUCED),)
        plan = Mock(nested_stage=Mock(pointwise_domains=domains))

        with patch.object(
            FusedNestedReductions,
            "_plan_append",
            return_value=(Mock(), plan),
        ):
            result = nested._plan_fusion_with(other)

        self.assertIsNone(result)

    @inductor_config.patch(combo_kernel_max_num_nodes=16)
    def test_combo_kernel_grouping_respects_mempool(self):
        scheduler = Mock()
        device = torch.device("cuda", 0)
        pool_node1 = self._mock_base_snode("pool_node1", device)
        pool_node2 = self._mock_base_snode("pool_node2", device)
        default_node = self._mock_base_snode("default_node", device)
        other_pool_node = self._mock_base_snode("other_pool_node", device)
        scheduler._topological_sort_nodes.return_value = [
            [pool_node1, default_node, pool_node2, other_pool_node]
        ]
        scheduler.node_to_stream = {
            pool_node1: 0,
            pool_node2: 0,
            default_node: 0,
            other_pool_node: 0,
        }
        scheduler.get_node_stream.side_effect = scheduler.node_to_stream.__getitem__
        scheduler.node_to_mempool = {
            pool_node1: (7, 0),
            pool_node2: (7, 0),
            default_node: None,
            other_pool_node: (8, 0),
        }

        groups = ForeachKernelSchedulerNode._default_group_nodes_for_combo_kernels(
            scheduler
        )

        self.assertEqual(
            groups, [[pool_node1, pool_node2], [default_node], [other_pool_node]]
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

    def test_sub_parent_resolver_rejects_inconsistent_name_contract(self):
        d0 = sympy.Symbol("d0", integer=True)
        access = MemoryDep("buf0", d0, (d0,), (sympy.Integer(16),))
        direct = SubParentAccessRelation((access,), access, None, False)
        lane = SubParentAccessRelation((access,), access, 0, False)
        other_lane = SubParentAccessRelation((access,), access, 1, False)
        required = SubParentAccessRelation((access,), access, 0, True)
        other_source = MemoryDep("buf0", d0 + 1, (d0,), (sympy.Integer(16),))
        other = SubParentAccessRelation((other_source,), access, 0, False)
        with V.set_graph_handler(Mock(sizevars=SizeVarAllocator())):
            with self.assertRaisesRegex(AssertionError, "mixed direct and lane"):
                self._make_sub_parent_value_resolver((direct, lane))
            with self.assertRaisesRegex(
                AssertionError, "consumer access has multiple lanes"
            ):
                self._make_sub_parent_value_resolver((lane, other_lane))
            with self.assertRaisesRegex(AssertionError, "mixed source roles"):
                self._make_sub_parent_value_resolver((lane, required))
            with self.assertRaisesRegex(AssertionError, "mixed source accesses"):
                self._make_sub_parent_value_resolver((lane, other))

    def test_sub_parent_resolver_uses_planned_lane_set(self):
        d0 = sympy.Symbol("d0", integer=True, nonnegative=True)
        source = MemoryDep("buf0", d0, (d0,), (sympy.Integer(16),))
        relations = tuple(
            SubParentAccessRelation(
                (source,),
                MemoryDep("buf0", 4 * d0 + lane, (d0,), (sympy.Integer(4),)),
                lane,
                True,
            )
            for lane in (0, 2)
        )
        kernel = Mock(_load_mask=None, _load_other=None)
        kernel.cse.contains_value.return_value = True
        with V.set_graph_handler(Mock(sizevars=SizeVarAllocator())):
            resolver = self._make_sub_parent_value_resolver(
                relations,
                kernel=kernel,
                family=Mock(lane_index_subs={}, lane_source_sizes=()),
                factor=4,
            )
            source_value = Mock(shape=("X", "PARENT"))
            lane_values = tuple(Mock() for _ in range(4))
            resolver._values = {"buf0": [source_value]}
            resolver._materialize = Mock(return_value=lane_values)
            self.assertIs(resolver.resolve_load("buf0", 4 * d0), lane_values[0])
            self.assertIs(resolver.resolve_load("buf0", 4 * d0 + 2), lane_values[2])
            with self.assertRaisesRegex(AssertionError, "unplanned lane 1"):
                resolver.resolve_load("buf0", 4 * d0 + 1)

    def test_sub_parent_source_capture_is_role_aware(self):
        d0 = sympy.Symbol("d0", integer=True)
        external = MemoryDep("external", d0, (d0,), (sympy.Integer(16),))
        internal = MemoryDep("internal", d0, (d0,), (sympy.Integer(16),))
        relations = (
            SubParentAccessRelation((external,), external, None, False),
            SubParentAccessRelation((internal,), internal, None, True),
        )
        kernel = Mock(_load_mask=None, _load_other=None)
        resolver = self._make_sub_parent_value_resolver(
            relations,
            kernel=kernel,
        )
        external_parent = Mock(shape=("X", "PARENT"))
        external_group = Mock(shape=("X", "GROUP"))
        external_replacement = Mock(shape=("X", "PARENT"))
        internal_store = Mock(shape=("X", "GROUP"))
        internal_replay = Mock(shape=("X", "GROUP"))

        resolver._record("external", external_parent, store=False)
        resolver._record("external", external_group, store=False)
        resolver._record("external", Mock(shape=("X", "STORE")), store=True)
        resolver._record("internal", Mock(shape=("X", "LOAD")), store=False)
        resolver._record("unplanned", Mock(shape=("X", "OTHER")), store=False)
        self.assertEqual(
            resolver._values,
            {"external": OrderedSet([external_parent, external_group])},
        )

        resolver._record("external", external_replacement, store=False)
        resolver._record("internal", internal_store, store=True)
        self.assertEqual(
            resolver._values["external"],
            OrderedSet([external_parent, external_group, external_replacement]),
        )
        self.assertEqual(resolver._values["internal"], OrderedSet([internal_store]))
        resolver._materialized[internal_store] = Mock()
        resolver._record("internal", internal_replay, store=True)
        self.assertEqual(resolver._values["internal"], OrderedSet([internal_replay]))
        self.assertNotIn(internal_store, resolver._materialized)

        materialized = Mock()
        resolver._kernel.cse.contains_value.return_value = True
        resolver._materialize = Mock(side_effect=(None, materialized))
        self.assertIs(resolver.resolve_load("external", sympy.Integer(0)), materialized)

    def test_sub_parent_required_source_must_remain_live(self):
        d0 = sympy.Symbol("d0", integer=True)
        access = MemoryDep("buf0", d0, (d0,), (sympy.Integer(16),))
        relation = SubParentAccessRelation((access,), access, None, True)
        kernel = Mock(_load_mask=None, _load_other=None)
        kernel.cse.contains_value.return_value = False
        resolver = self._make_sub_parent_value_resolver(
            (relation,),
            kernel=kernel,
        )

        with self.assertRaisesRegex(AssertionError, "lost required .*source 'buf0'"):
            resolver.resolve_load("buf0", sympy.Integer(0))
        kernel.cse.contains_value.return_value = True
        resolver._values = {"buf0": [Mock(shape=("X", "GROUP"))]}
        resolver._materialize = Mock(return_value=None)
        with self.assertRaisesRegex(AssertionError, "lost required .*source 'buf0'"):
            resolver.materialize_sources((relation,))

    def test_sub_parent_resolver_uses_masked_load_ownership(self):
        d0 = sympy.Symbol("d0", integer=True)
        graph_handler = Mock(sizevars=SizeVarAllocator())
        with V.set_graph_handler(graph_handler):
            access = MemoryDep("buf0", d0, (d0,), (sympy.Integer(16),)).normalize()
        relation = SubParentAccessRelation((access,), access, None, False)
        value = Mock()
        resolver = self._make_sub_parent_value_resolver(
            (relation,),
            kernel=Mock(_load_mask="source_mask", _load_other=0.0),
        )
        resolver._inner.load.return_value = value
        with V.set_graph_handler(graph_handler):
            self.assertIs(resolver.load("buf0", sympy.Integer(0)), value)
        self.assertEqual(resolver._values, {})

        resolver._values["buf0"] = [value]
        resolver._kernel.cse.contains_value.return_value = True
        resolver._kernel._load_mask = "consumer_mask"
        resolver._kernel._load_other = None
        (resolved,) = resolver.resolve_sources("buf0")
        self.assertIs(resolved, value)

        resolver._kernel._load_other = 7.0
        self.assertEqual(resolver.resolve_sources("buf0"), ())

    def test_sub_parent_external_fallback_and_atomic_store(self):
        resolver = Mock()
        resolver.is_planned.return_value = True
        resolver.resolve_load.return_value = None
        family = Mock()
        family.remap_index.return_value = sympy.Integer(7)
        family.ensure_active.return_value = contextlib.nullcontext()
        value = Mock(use_count=1)
        kernel = Mock(num_load=0)
        kernel.cse.invalidated_stores = OrderedSet()
        kernel.load.return_value = value
        inner = Mock()
        handler = _PointwiseRemapHandler(
            inner,
            kernel,
            family=family,
            value_resolver=resolver,
        )

        self.assertIs(handler.load("buf0", sympy.Integer(3)), value)
        resolver.resolve_load.assert_called_once_with("buf0", sympy.Integer(3))
        inner.load.assert_not_called()
        kernel.load.assert_called_once_with("buf0", sympy.Integer(7))
        resolver = object.__new__(_SubParentValueResolver)
        resolver._record = Mock()
        resolver._inner = Mock()
        for mode in ("atomic_add", "atomic_xchg", "tma"):
            resolver.store("buf0", sympy.Integer(0), Mock(), mode)
        resolver._record.assert_not_called()

    def test_group_width_equal_to_child_width_is_direct(self):
        x_tree, r_tree = Mock(), Mock()
        layout = _GroupedReductionLayout(x_tree, r_tree, sympy.Integer(2), True)
        value = Mock(dtype=torch.float32, shape=("X", "CHILD"))
        family = Mock()
        kernel = Mock()

        with (
            patch.object(_GroupedReductionLayout, "child_block", return_value="CHILD"),
            patch.object(
                _GroupedReductionLayout,
                "num_groups_str",
                new_callable=PropertyMock,
                return_value="CHILD",
            ),
        ):
            result = layout.materialize_value_at_sub_parent_resolution(
                kernel, family, 2, value
            )

        self.assertIs(result, value)
        family.set_value_masks.assert_called_once_with(kernel, (value,))
        kernel.emit_broadcast_via_reshape.assert_not_called()

    @parametrize(
        "load_mask,parent_lanes,shape,returns_raw",
        (
            (None, None, ("XBLOCK", "GROUP"), True),
            ("mask", None, ("XBLOCK", "GROUP"), False),
            (None, frozenset({0}), ("XBLOCK", "GROUP"), False),
            (None, None, ("XBLOCK", "CHILD"), False),
            (None, None, None, False),
        ),
    )
    def test_sub_parent_group_width_raw_resolution(
        self, load_mask, parent_lanes, shape, returns_raw
    ):
        source = CSEVariable(
            "source", ValueRanges.unknown(), torch.float32, shape=shape
        )
        child = CSEVariable(
            "child", ValueRanges.unknown(), torch.float32, shape=("XBLOCK", "CHILD")
        )
        resolver = object.__new__(_SubParentValueResolver)
        resolver._contracts = {
            "buf0": Mock(parent_lanes=parent_lanes, source_is_internal=False)
        }
        resolver._kernel = Mock(_load_mask=load_mask)
        resolver._layout = Mock(num_groups_str="GROUP")
        resolver._layout.parent_dim.side_effect = (
            lambda candidate: None if candidate is None else str(candidate[-1])
        )
        resolver._layout.child_block.return_value = "CHILD"
        resolver._sub_parent_factor = 2
        resolver.resolve_sources = Mock(return_value=(source,))
        resolver.materialize_source = Mock(return_value=child)

        result = resolver.resolve_load("buf0", sympy.Integer(0))

        self.assertIs(result, source if returns_raw else child)
        if returns_raw:
            resolver.materialize_source.assert_not_called()
        else:
            resolver.materialize_source.assert_called_once_with(
                "buf0", source, sympy.Integer(0)
            )

        resolver._layout.child_block.return_value = "GROUP"
        self.assertFalse(resolver.is_group_width_shape(("XBLOCK", "GROUP")))

    def test_sub_parent_group_width_materialization_boundaries(self):
        group_shape = ("XBLOCK", "GROUP")
        child_shape = ("XBLOCK", "CHILD")
        group = CSEVariable(
            "group", ValueRanges.unknown(), torch.float32, shape=group_shape
        )
        group_product = CSEVariable(
            "group_product", ValueRanges.unknown(), torch.float32, shape=group_shape
        )
        scalar = CSEVariable(
            "scalar", ValueRanges.unknown(), torch.float32, shape=(1, 1)
        )
        fp8 = CSEVariable(
            "fp8", ValueRanges.unknown(), torch.float8_e4m3fn, shape=group_shape
        )
        decoded = CSEVariable(
            "decoded", ValueRanges.unknown(), torch.float32, shape=group_shape
        )
        child = CSEVariable(
            "child", ValueRanges.unknown(), torch.float16, shape=child_shape
        )
        resolver = Mock()
        resolver.is_group_width_shape.side_effect = lambda shape: shape == group_shape
        resolver.materialize_group_width.side_effect = (
            lambda value: child if value is group else value
        )
        family = Mock()
        family.remap_index.return_value = sympy.Integer(7)
        family.ensure_active.return_value = contextlib.nullcontext()
        inner = Mock()
        handler = _PointwiseRemapHandler(
            inner,
            Mock(),
            family=family,
            value_resolver=resolver,
        )

        operation_result = Mock()
        inner.rand.return_value = operation_result
        self.assertIs(handler.rand(scalar, group), operation_result)
        self.assertEqual(resolver.materialize_group_width.call_count, 2)
        resolver.materialize_group_width.assert_any_call(group)
        inner.rand.assert_called_once_with(scalar, child)

        resolver.materialize_group_width.reset_mock()
        inner.abs.return_value = group_product
        self.assertIs(handler.abs(group), group_product)
        resolver.materialize_group_width.assert_not_called()
        inner.abs.assert_called_once_with(group)

        resolver.materialize_group_width.reset_mock()
        handler.store("buf0", sympy.Integer(3), group)
        resolver.materialize_group_width.assert_called_once_with(group)
        inner.store.assert_called_once_with("buf0", sympy.Integer(7), child, mode=None)

        resolver.materialize_group_width.reset_mock()

        body = Mock(return_value=group)
        body.graph = object()
        inner.masked.side_effect = lambda mask, callback, other: callback()
        self.assertIs(handler.masked(child, body, 0.0), child)
        masked_body = inner.masked.call_args.args[1]
        self.assertIs(masked_body.graph, body.graph)
        resolver.materialize_group_width.assert_any_call(group)

        resolver.materialize_group_width.reset_mock()
        inner.mul.return_value = group_product
        self.assertIs(handler.mul(group, 2.0), group_product)
        resolver.materialize_group_width.assert_not_called()
        inner.mul.assert_called_once_with(group, 2.0)

        inner.reciprocal.return_value = group_product
        self.assertIs(handler.reciprocal(group), group_product)
        resolver.materialize_group_width.assert_not_called()
        inner.reciprocal.assert_called_once_with(group)

        inner.truediv.return_value = group_product
        self.assertIs(handler.truediv(scalar, group), group_product)
        resolver.materialize_group_width.assert_not_called()
        inner.truediv.assert_called_once_with(scalar, group)

        inner.mul.reset_mock()
        inner.mul.return_value = child
        self.assertIs(handler.mul(group, child), child)
        resolver.materialize_group_width.assert_any_call(group)
        inner.mul.assert_called_once_with(child, child)

        resolver.materialize_group_width.reset_mock()
        inner.to_dtype.return_value = fp8
        self.assertIs(
            handler.to_dtype(group, torch.float8_e4m3fn, torch.float32, False), fp8
        )
        inner.to_dtype.assert_called_once_with(
            group, torch.float8_e4m3fn, torch.float32, False
        )
        resolver.materialize_group_width.assert_not_called()

        inner.to_dtype.reset_mock()
        inner.to_dtype.return_value = decoded
        self.assertIs(
            handler.to_dtype(fp8, torch.float32, torch.float8_e4m3fn, False), decoded
        )
        resolver.materialize_group_width.assert_not_called()

        inner.to_dtype.reset_mock()
        inner.to_dtype.return_value = child
        with self.assertRaisesRegex(AssertionError, "did not preserve group width"):
            handler.to_dtype(group, torch.float16)
        resolver.materialize_group_width.assert_not_called()

    @parametrize("loop_ordering", [False, True])
    def test_fusable_read_and_write_requires_exact_index_match(self, loop_ordering):
        d0, d1, d2 = sympy.symbols("d0 d1 d2", integer=True, nonnegative=True)
        w0, w1 = sympy.symbols("w0 w1", integer=True, nonnegative=True)
        scheduler = Scheduler.__new__(Scheduler)

        # Gapped (stride 33 across a 32 wide dim) so loop merging cannot
        # collapse these deps and hide which branch accepted them.
        gapped = MemoryDep("buf", 33 * d0 + d1, (d0, d1), (128, 32))
        extended = MemoryDep("buf", 33 * d0 + d1, (d0, d1, d2), (128, 32, 7))
        narrowed = MemoryDep("buf", 33 * d0 + d1, (d0, d1), (64, 32))
        renamed = MemoryDep("buf", 33 * w0 + w1, (w0, w1), (128, 32))
        simple_write = MemoryDep("buf", w0, (w0,), (16,))
        equivalent_only = MemoryDep("buf", d1, (d0, d1), (1024, 16))
        non_equivalent = MemoryDep("buf", d0 + d1, (d0, d1), (2, 2))
        aliased_write = MemoryDep("buf", w0 + w1, (w0, w1), (2, 2))
        quotient_write = MemoryDep("buf", 32 * w0 + w1, (w0, w1), (128, 32))
        quotient = MemoryDep("buf", 32 * d0 + FloorDiv(d1, 128), (d0, d1), (128, 4096))
        quotient_tail = MemoryDep("buf", quotient.index + d1, (d0, d1), (128, 4096))
        s0, s1 = sympy.symbols("s0 s1", integer=True, positive=True)
        dense_read = MemoryDep("buf", s1 * d0 + d1, (d0, d1), (s0, s1))
        dense_write = MemoryDep("buf", s1 * w0 + w1, (w0, w1), (s0, s1))

        graph = Mock(sizevars=SizeVarAllocator())
        with (
            V.set_graph_handler(graph),
            inductor_config.patch(loop_ordering_after_fusion=loop_ordering),
        ):
            # _same_index_with_prefix_size: identical index, and read sizes
            # that cover the write sizes as a prefix.
            self.assertTrue(scheduler.fusable_read_and_write(gapped, gapped))
            self.assertTrue(scheduler.fusable_read_and_write(extended, gapped))
            self.assertFalse(scheduler.fusable_read_and_write(narrowed, gapped))
            # Normalization drops the unused d2 loop and canonicalizes symbols.
            self.assertEqual(
                scheduler.fusable_read_and_write(extended, renamed),
                loop_ordering,
            )
            self.assertFalse(
                scheduler.fusable_read_and_write(equivalent_only, simple_write)
            )
            self.assertTrue(
                scheduler._fusable_read_after_index_equivalence(
                    equivalent_only, simple_write
                )
            )
            self.assertFalse(
                scheduler._fusable_read_after_index_equivalence(
                    non_equivalent, aliased_write
                )
            )
            self.assertTrue(
                scheduler._fusable_read_after_index_equivalence(dense_read, dense_write)
            )
            self.assertTrue(
                scheduler._fusable_read_after_index_equivalence(
                    quotient, quotient_write
                )
            )
            self.assertFalse(
                scheduler._fusable_read_after_index_equivalence(
                    quotient_tail, quotient_write
                )
            )

    def test_sub_parent_broadcast_access_relation_frame_contract(self):
        s0, s1 = sympy.symbols("s0 s1", integer=True, nonnegative=True)
        d0, d1, d2 = sympy.symbols("d0 d1 d2", integer=True, nonnegative=True)
        source = MemoryDep("buf", 4 * s0 + s1, (s0, s1), (2, 4))
        good = MemoryDep("buf", 4 * d0 + d1, (d0, d1), (2, 4))
        used_extra_axis = MemoryDep(
            "buf", 8 * d0 + 2 * d1 + d2, (d0, d1, d2), (2, 4, 2)
        )
        regrouped = MemoryDep("buf", 2 * d0 + d1, (d0, d1), (4, 2))

        source_node = Mock()
        source_node.read_writes.writes = OrderedSet([source])
        graph = Mock(sizevars=SizeVarAllocator())
        with V.set_graph_handler(graph):
            for read, expected in (
                (good, True),
                (used_extra_axis, False),
                (regrouped, False),
            ):
                consumer_node = Mock()
                consumer_node.read_writes.reads = OrderedSet([read])
                relation = NestedReduction._sub_parent_broadcast_access_relations(
                    (source_node,), (consumer_node,), OrderedSet(["buf"])
                )
                self.assertEqual(relation is not None, expected)

            source_node.read_writes.writes = OrderedSet([source.normalize()])
            consumer_node.read_writes.reads = OrderedSet([regrouped.normalize()])
            self.assertIsNotNone(
                NestedReduction._sub_parent_broadcast_access_relations(
                    (source_node,), (consumer_node,), OrderedSet(["buf"])
                )
            )

            source_node.read_writes.writes.add(
                MemoryDep("buf", 4 * s0 + s1 + 1, (s0, s1), (2, 4))
            )
            self.assertIsNone(
                NestedReduction._sub_parent_broadcast_access_relations(
                    (source_node,), (consumer_node,), OrderedSet(["buf"])
                )
            )

            source_node.read_writes.writes = OrderedSet([source])
            consumer_node.read_writes.reads = OrderedSet([StarDep("buf")])
            self.assertIsNone(
                NestedReduction._sub_parent_broadcast_access_relations(
                    (source_node,), (consumer_node,), OrderedSet(["buf"])
                )
            )

    def test_sub_parent_internal_access_relation_preserves_emission_frame(self):
        s0, s1 = sympy.symbols("s0 s1", integer=True, nonnegative=True)
        d0, d1 = sympy.symbols("d0 d1", integer=True, nonnegative=True)

        def node(*, reads=(), writes=()):
            result = Mock()
            result.read_writes.reads = OrderedSet(reads)
            result.read_writes.writes = OrderedSet(writes)
            return result

        def group(output_lanes, *nodes):
            return SubParentOutputGroup(output_lanes=output_lanes, nodes=nodes)

        def relations(*groups):
            return NestedReduction._sub_parent_internal_access_relations(groups)

        source = MemoryDep("same", 4 * s0 + s1, (s0, s1), (2, 4))
        flattened = MemoryDep("same", d0, (d0,), (8,))
        broadcast_source = MemoryDep("cross", s0, (s0,), (8,))
        broadcast_read = MemoryDep("cross", d0, (d0, d1), (8, 3))
        graph = Mock(sizevars=SizeVarAllocator())
        with V.set_graph_handler(graph):
            same_group = relations(
                group(1, node(writes=(source,)), node(reads=(flattened,)))
            )
            self.assertIsNotNone(same_group)

            cross_group = relations(
                group(1, node(writes=(broadcast_source,))),
                group(3, node(reads=(broadcast_read,))),
            )
            self.assertIsNotNone(cross_group)

            wrong_frame = relations(
                group(1, node(writes=(source,))),
                group(3, node(reads=(flattened,))),
            )
            self.assertIsNone(wrong_frame)

            duplicate_writer = relations(
                group(
                    1,
                    node(writes=(source,)),
                    node(writes=(source,)),
                    node(reads=(flattened,)),
                )
            )
            self.assertIsNone(duplicate_writer)

            consumer_before_writer = relations(
                group(1, node(reads=(broadcast_read,))),
                group(3, node(writes=(broadcast_source,))),
            )
            self.assertIsNone(consumer_before_writer)

            self.assertIsNone(
                relations(
                    group(
                        1,
                        node(reads=(flattened,)),
                        node(writes=(source,)),
                    )
                )
            )

    def test_group_sub_parent_epilogue_nodes(self):
        first, second, third = Mock(), Mock(), Mock()
        self.assertEqual(
            NestedReduction._group_sub_parent_epilogue_nodes(
                (
                    SubParentEpilogueCandidate(first, 4, 1),
                    SubParentEpilogueCandidate(second, 4, 3),
                    SubParentEpilogueCandidate(third, 4, 3),
                )
            ),
            SubParentEpilogueGrouping(
                factor=4,
                output_groups=(
                    SubParentOutputGroup(1, (first,)),
                    SubParentOutputGroup(3, (second, third)),
                ),
            ),
        )
        self.assertIsNone(
            NestedReduction._group_sub_parent_epilogue_nodes(
                (
                    SubParentEpilogueCandidate(first, 2, 1),
                    SubParentEpilogueCandidate(second, 4, 3),
                )
            )
        )
        self.assertIsNone(
            NestedReduction._group_sub_parent_epilogue_nodes(
                (
                    SubParentEpilogueCandidate(first, 4, 3),
                    SubParentEpilogueCandidate(second, 4, 1),
                )
            )
        )

    @parametrize("extra_dep", [None, "memory", "star", "weak"])
    def test_planned_dependency_matches_only_relax_memory_deps(self, extra_dep):
        read_var, write_var = sympy.symbols(
            "read_var write_var", integer=True, nonnegative=True
        )
        read = MemoryDep("buf", read_var + 1, (read_var,), (4,))
        write = MemoryDep("buf", write_var, (write_var,), (4,))
        deps = OrderedSet([read])
        if extra_dep == "memory":
            deps.add(MemoryDep("buf", read_var + 2, (read_var,), (4,)))
        elif extra_dep == "star":
            deps.add(StarDep("buf"))
        elif extra_dep == "weak":
            deps.add(WeakDep("buf", "mutated"))

        producer = Mock()
        producer.get_name.return_value = "producer"
        producer.get_buffer_names.return_value = OrderedSet(["buf"])
        producer.get_operation_names.return_value = OrderedSet()
        producer.read_writes.writes = OrderedSet([write])
        consumer = Mock()
        consumer.get_name.return_value = "consumer"
        consumer.unmet_dependencies = deps
        consumer.read_writes.writes = OrderedSet()

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.mutation_renames = {}
        scheduler.fusable_weak_dep = Mock(return_value=False)
        scheduler.name_to_buf = {}
        scheduler.name_to_fused_node = {}

        self.assertFalse(scheduler.can_fuse_vertical(producer, consumer))
        self.assertEqual(
            scheduler._can_fuse_vertical_impl(
                producer,
                consumer,
                (MemoryDepMatch(write, read),),
            ),
            extra_dep is None,
        )

    @parametrize(
        "write_kind,planned_name",
        (("dense", "buf"), ("alias", "buf"), ("dense", "other")),
    )
    def test_nested_dependency_matches_require_injective_producer(
        self, write_kind, planned_name
    ):
        d0, d1 = sympy.symbols("d0 d1", integer=True, nonnegative=True)
        index = 2 * d0 + d1 if write_kind == "dense" else d0 + d1
        write = MemoryDep("buf", index, (d0, d1), (2, 2))
        read = MemoryDep("buf", d0 + 1, (d0,), (4,))
        planned_write = write.rename({"buf": planned_name})
        planned_read = read.rename({"buf": planned_name})
        producer = Mock()
        producer.get_buffer_names.return_value = OrderedSet(["buf"])
        producer.read_writes.writes = OrderedSet([write])
        consumer = Mock()
        consumer.read_writes.reads = OrderedSet([read])
        consumer.unmet_dependencies = OrderedSet([read])
        relation = SubParentAccessRelation((planned_write,), planned_read, None, True)
        plan = Mock(
            nested_stage=None,
            sub_parent_stages=(
                Mock(access_relations=(relation,), epilogue_nodes=(consumer,)),
            ),
        )
        plan.sub_parent_access_pairs.return_value = ((planned_write, planned_read),)
        scheduler = Scheduler.__new__(Scheduler)
        # Plans retain temporal names; mutation aliases are applied only while
        # matching the producer/read pair for fusion.
        scheduler.mutation_renames = {"buf": "renamed_buf", "other": "renamed_buf"}
        graph = Mock(sizevars=SizeVarAllocator())
        with V.set_graph_handler(graph):
            matches = scheduler._prove_staged_fusion_dependencies(
                producer, consumer, plan
            )
        expected = (
            (
                MemoryDepMatch(
                    write.rename(scheduler.mutation_renames),
                    read.rename(scheduler.mutation_renames),
                ),
            )
            if write_kind == "dense" and planned_name == "buf"
            else None
        )
        self.assertEqual(matches, expected)

    @parametrize("ownership", ["nested", "both", "none"])
    def test_nested_dependency_matches_scope_index_equivalence(self, ownership):
        d0, d1, w0 = sympy.symbols("d0 d1 w0", integer=True, nonnegative=True)
        write = MemoryDep("buf", w0, (w0,), (16,))
        read = MemoryDep("buf", d1, (d0, d1), (1024, 16))
        producer = Mock()
        producer.get_buffer_names.return_value = OrderedSet(["buf"])
        producer.read_writes.writes = OrderedSet([write])
        consumer = Mock()
        consumer.read_writes.reads = OrderedSet([read])
        consumer.unmet_dependencies = OrderedSet([read])
        plan = Mock(
            nested_stage=Mock(grouped_nodes=(consumer,))
            if ownership != "none"
            else None,
            sub_parent_stages=(Mock(access_relations=(), epilogue_nodes=(consumer,)),)
            if ownership == "both"
            else (),
        )
        plan.sub_parent_access_pairs.return_value = ()
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.mutation_renames = {}
        graph = Mock(sizevars=SizeVarAllocator())
        with V.set_graph_handler(graph):
            matches = scheduler._prove_staged_fusion_dependencies(
                producer, consumer, plan
            )
        expected = (MemoryDepMatch(write, read),) if ownership == "nested" else None
        self.assertEqual(matches, expected)

    @parametrize("reason", ["multiwrite", "tmp", "atomic"])
    def test_planned_dependency_matches_reject_unsafe_write(self, reason):
        d0 = sympy.Symbol("d0", integer=True, nonnegative=True)
        index = make_symbol(SymT.TMP, 0) if reason == "tmp" else d0
        mode = "atomic_add" if reason == "atomic" else None
        write = MemoryDep("buf", index, (d0,), (4,), mode)
        read = MemoryDep("buf", index + 1, (d0,), (4,), mode)
        writes = OrderedSet([write])
        if reason == "multiwrite":
            writes.add(MemoryDep("buf", d0 + 2, (d0,), (4,)))

        producer = Mock()
        producer.get_buffer_names.return_value = OrderedSet(["buf"])
        producer.read_writes.writes = writes
        consumer = Mock()
        consumer.read_writes.reads = OrderedSet([read])
        consumer.unmet_dependencies = OrderedSet([read])
        plan = Mock(
            nested_stage=None,
            sub_parent_stages=(Mock(access_relations=(), epilogue_nodes=(consumer,)),),
        )
        plan.sub_parent_access_pairs.return_value = ((write, read),)
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.mutation_renames = {}

        self.assertIsNone(
            scheduler._prove_staged_fusion_dependencies(producer, consumer, plan)
        )

    @parametrize("vertical_fusion_legal", [False, True])
    def test_empty_planned_dependency_matches_prevent_later_loop_rewrites(
        self, vertical_fusion_legal
    ):
        producer = self._mock_base_snode("producer", torch.device("cpu"))
        consumer = self._mock_base_snode("consumer", torch.device("cpu"))
        producer.has_strict_reduction.return_value = False
        consumer.has_strict_reduction.return_value = False
        producer.ancestors = OrderedSet()
        consumer.ancestors = OrderedSet(["producer"])
        producer.get_operation_names.return_value = OrderedSet(["producer"])
        consumer.get_operation_names.return_value = OrderedSet(["consumer"])

        scheduler = Scheduler.__new__(Scheduler)
        scheduler._fusion_blocked_by_placement = Mock(return_value=False)
        scheduler._prove_staged_fusion_dependencies = Mock(return_value=())
        scheduler._score_fusion_memory_for_can_fuse = Mock(
            side_effect=AssertionError("staged fusion must use staged scoring")
        )
        scheduler._score_staged_fusion_memory_for_can_fuse = Mock(return_value=0)
        scheduler.get_expand_dim_for_pointwise_nodes = Mock(
            side_effect=AssertionError("staged plan must prevent expansion")
        )
        scheduler.shared_data_after_reordering_loop = Mock(
            side_effect=AssertionError("staged plan must prevent reordering")
        )
        scheduler.shared_data_after_inverting_indexing = Mock(
            side_effect=AssertionError("staged plan must prevent inversion")
        )
        scheduler.can_fuse_vertical = Mock(
            side_effect=AssertionError("staged fusion must use staged legality")
        )
        scheduler._can_fuse_vertical_impl = Mock(return_value=vertical_fusion_legal)
        scheduler._try_reindex_pointwise_for_reduction = Mock(
            side_effect=AssertionError("staged plan must prevent reindexing")
        )
        backend = Mock()
        backend.can_fuse_vertical.return_value = True
        scheduler.get_backend = Mock(return_value=backend)
        graph = Mock(no_fuse_buffer_names=OrderedSet())
        choices = Mock()
        choices.can_fuse.return_value = True
        choices.can_fuse_vertical.return_value = True

        with (
            V.set_graph_handler(graph),
            V.set_choices_handler(choices),
            patch.object(NestedReduction, "is_candidate", return_value=True),
            patch.object(NestedReduction, "plan", return_value=Mock()),
            inductor_config.patch(
                {
                    "expand_dimension_for_pointwise_nodes": True,
                    "loop_ordering_after_fusion": True,
                    "loop_reindexing_after_fusion": True,
                    "loop_index_inversion_in_fusion": True,
                }
            ),
        ):
            self.assertEqual(
                scheduler._can_fuse(
                    producer,
                    consumer,
                    can_reorder=True,
                ),
                vertical_fusion_legal,
            )

        scheduler.get_expand_dim_for_pointwise_nodes.assert_not_called()
        scheduler.shared_data_after_reordering_loop.assert_not_called()
        scheduler.shared_data_after_inverting_indexing.assert_not_called()
        scheduler._try_reindex_pointwise_for_reduction.assert_not_called()

    def test_vertical_fusion_retries_after_reindexing(self):
        producer = self._mock_base_snode("producer", torch.device("cuda"))
        consumer = self._mock_base_snode("consumer", torch.device("cuda"))
        producer.has_strict_reduction.return_value = False
        consumer.has_strict_reduction.return_value = False
        producer.ancestors = OrderedSet()
        consumer.ancestors = OrderedSet(["producer"])
        producer.get_operation_names.return_value = OrderedSet(["producer"])
        consumer.get_operation_names.return_value = OrderedSet(["consumer"])

        scheduler = Scheduler.__new__(Scheduler)
        scheduler._fusion_blocked_by_placement = Mock(return_value=False)
        scheduler._score_fusion_memory_for_can_fuse = Mock(return_value=1_000_000)
        scheduler.can_fuse_vertical = Mock(side_effect=[False, True])
        scheduler._try_reindex_pointwise_for_reduction = Mock(return_value=True)
        backend = Mock()
        backend.can_fuse_vertical.return_value = True
        scheduler.get_backend = Mock(return_value=backend)
        graph = Mock(no_fuse_buffer_names=OrderedSet())
        choices = Mock()
        choices.can_fuse.return_value = True
        choices.can_fuse_vertical.return_value = True

        with (
            V.set_graph_handler(graph),
            V.set_choices_handler(choices),
            patch.object(NestedReduction, "is_candidate", return_value=False),
            patch.object(NestedReduction, "_is_enabled_for", return_value=False),
            inductor_config.patch(loop_reindexing_after_fusion=True),
        ):
            self.assertTrue(Scheduler._can_fuse(scheduler, producer, consumer))

        self.assertEqual(scheduler.can_fuse_vertical.call_count, 2)
        scheduler._try_reindex_pointwise_for_reduction.assert_called_once_with(
            producer, consumer
        )

    def test_nested_reduction_sub_parent_rate_preserves_group_axis(self):
        grouped = Mock()
        grouped.get_ranges.return_value = ([3, 6], [16])
        sub_parent = Mock()
        sub_parent.group = (None, (144, 1))
        sub_parent.get_ranges.return_value = ([3, 6, 8], [])
        graph = Mock(sizevars=SizeVarAllocator())

        with V.set_graph_handler(graph):
            context = NestedReduction.PointwiseDomainContext.create(
                grouped,
                grouped_numel=18,
                grouped_rnumel=16,
                grouped_axis=NestedReduction.GroupedAxis.R,
                group_size=16,
                parent_numel=3,
                parent_rnumel=96,
            )
            x_grouped_context = NestedReduction.PointwiseDomainContext.create(
                grouped,
                grouped_numel=18,
                grouped_rnumel=16,
                grouped_axis=NestedReduction.GroupedAxis.X,
                group_size=16,
                parent_numel=3,
                parent_rnumel=96,
            )
            rate = NestedReduction._nested_sub_parent_rate(sub_parent, context)
            sub_parent.get_ranges.return_value = ([3, 8, 6], [])
            cross_group_rate = NestedReduction._nested_sub_parent_rate(
                sub_parent, context
            )
            x_grouped_rate = NestedReduction._nested_sub_parent_rate(
                sub_parent, x_grouped_context
            )
        self.assertEqual(rate, (2, 1))
        self.assertIsNone(cross_group_rate)
        self.assertIsNone(x_grouped_rate)

    def test_nested_reduction_rejects_ambiguous_pointwise_domain(self):
        grouped = self._mock_schedule_node(
            "grouped", reads=("source",), writes=("reduced",), is_reduction=True
        )
        grouped.get_ranges.return_value = ([8], [2])
        consumer = self._mock_schedule_node(
            "consumer", reads=("reduced",), ancestors=("grouped",)
        )
        consumer.__class__ = SchedulerNode
        context = Mock(grouped_reduction=grouped, grouped_numel=8, grouped_rnumel=2)
        graph = Mock(sizevars=SizeVarAllocator())

        with (
            V.set_graph_handler(graph),
            patch.object(
                NestedReduction, "_pointwise_node_matches_domain", return_value=True
            ),
            patch.object(
                NestedReduction, "_nested_sub_parent_rate", return_value=(2, 1)
            ),
        ):
            result = NestedReduction._classify_grouped_pointwise_nodes(
                context, (grouped, consumer)
            )

        self.assertIsNone(result)

    def test_nested_reduction_rejects_template_nodes(self):
        outer = self._mock_schedule_node("outer", is_reduction=True)
        outer.node = Mock(spec=ir.TemplateBuffer)
        outer.get_nodes.return_value = (outer,)
        grouped = self._mock_schedule_node("grouped", is_reduction=True)
        grouped.get_nodes.return_value = (grouped,)
        context = Mock(grouped_axis=NestedReduction.GroupedAxis.R)

        self.assertFalse(
            NestedReduction._r_grouped_stage_accesses_match(outer, grouped, context, ())
        )

    @parametrize("writer_role", ["parent_stage", "local_input", "reduction"])
    def test_nested_sub_parent_rejects_parent_stage_live_source(self, writer_role):
        outer_reduction = self._mock_schedule_node(
            "outer_reduction", writes=("rstd",), is_reduction=True
        )
        writer = self._mock_schedule_node(
            "writer", writes=("source",), is_reduction=writer_role == "reduction"
        )
        grouped = self._mock_schedule_node(
            "grouped", reads=("source",), writes=("scale",), is_reduction=True
        )
        epilogue = self._mock_schedule_node(
            "epilogue",
            reads=("source", "scale"),
            writes=("packed",),
            ancestors=("writer", "grouped"),
        )
        for node in (outer_reduction, writer, grouped, epilogue):
            node.has_aliasing_or_mutation.return_value = False
        outer = Mock()
        outer.get_nodes.return_value = (outer_reduction, writer)
        outer.group = (None, (8, 16))
        context = Mock(grouped_reduction=grouped, grouped_rnumel=2)
        domains = [(epilogue, NestedReduction.PointwiseDomain.SUB_PARENT)]
        if writer_role == "local_input":
            domains.append(
                (writer, NestedReduction.PointwiseDomain.LOCAL_REDUCTION_INPUT)
            )
        relation = Mock(requires_live_source=True)
        relation.consumer_access.name = "source"
        grouping = Mock(output_groups=(Mock(output_lanes=1, nodes=(epilogue,)),))
        grouping.factor = 2
        graph = Mock(sizevars=SizeVarAllocator())

        with (
            V.set_graph_handler(graph),
            patch.object(
                NestedReduction, "_nested_sub_parent_rate", return_value=(2, 1)
            ),
            patch.object(
                NestedReduction,
                "_group_sub_parent_epilogue_nodes",
                return_value=grouping,
            ),
            patch.object(
                NestedReduction,
                "_sub_parent_internal_access_relations",
                return_value=(),
            ),
            patch.object(
                NestedReduction,
                "_sub_parent_epilogue_outputs_unread",
                return_value=True,
            ),
            patch.object(
                NestedReduction,
                "_try_get_sub_parent_access_relations",
                return_value=(relation,),
            ),
            patch.object(
                NestedReduction,
                "_sub_parent_broadcast_access_relations",
                return_value=(),
            ),
        ):
            stage = NestedReduction._plan_nested_sub_parent_stage(
                outer, (grouped, epilogue), context, domains
            )

        # Only a value produced inside the parent loop is dead once a looped
        # parent closes it; displaced and reduction writers stay live.
        if writer_role == "parent_stage":
            self.assertIsNone(stage)
        else:
            self.assertIsNotNone(stage)

    def test_sub_parent_parent_order_closes_final_loop_dependencies(self):
        source = self._mock_schedule_node("source", writes=("source",))
        sibling = self._mock_schedule_node("sibling", writes=("sibling",))
        reduction = self._mock_schedule_node(
            "reduction", writes=("reduction",), is_reduction=True
        )
        output = self._mock_schedule_node(
            "output",
            reads=("source", "sibling"),
            writes=("output",),
            ancestors=("source", "sibling"),
        )
        graph = Mock(sizevars=SizeVarAllocator())

        with (
            V.set_graph_handler(graph),
            patch.object(
                NestedReduction, "_pointwise_node_matches_domain", return_value=True
            ),
        ):
            result = NestedReduction._order_sub_parent_parent_nodes(
                (source, sibling, reduction, output),
                OrderedSet(["source"]),
                8,
                16,
            )

        self.assertEqual(
            result,
            OrderedParentNodes(
                nodes=(reduction, source, sibling, output),
                required_post_reduction_index=1,
            ),
        )

    def test_sub_parent_parent_order_stops_at_reduced_output(self):
        source_input = self._mock_schedule_node(
            "source_input", writes=("source_input",)
        )
        reduction = self._mock_schedule_node(
            "reduction", writes=("reduction",), is_reduction=True
        )
        source = self._mock_schedule_node(
            "source",
            reads=("source_input", "reduction"),
            writes=("source",),
            ancestors=("source_input", "reduction"),
        )
        graph = Mock(sizevars=SizeVarAllocator())

        with (
            V.set_graph_handler(graph),
            patch.object(
                NestedReduction, "_pointwise_node_matches_domain", return_value=True
            ),
        ):
            result = NestedReduction._order_sub_parent_parent_nodes(
                (source_input, reduction, source),
                OrderedSet(["source"]),
                8,
                16,
            )

        self.assertEqual(
            result,
            OrderedParentNodes(
                nodes=(reduction, source_input, source),
                required_post_reduction_index=1,
            ),
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
            scheduler._mempool_nodes = False
            scheduler.node_to_mempool = {}
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
            msg=lambda msg: f"{msg}\nindex_add_ fusion bug detected: "
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
            msg=lambda msg: f"{msg}\nFusion bug detected! Expected {expected}, got {result}",
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

    @xfailIfNoAcceleratorTriton
    @onlyCUDA
    @parametrize("op", ["select_scatter", "index_put"])
    # Both settings are pinned so the test can only pass via graph_fanout:
    # deterministic mode makes expand realize src on its own, and the read
    # threshold decides whether src counts as expensive at all.
    @inductor_config.patch(deterministic=False, realize_reads_threshold=4)
    def test_scatter_realizes_expensive_src(self, op):
        def src(a, b, c, d, e):
            return a[..., 1] * b[..., 0] + c[..., 1] * d[..., 0] + e[..., 2]

        if op == "select_scatter":

            def fn(base, *args):
                return torch.select_scatter(base, src(*args), dim=2, index=1)
        else:

            def fn(base, *args):
                index = torch.arange(base.shape[0], device=base.device)
                base.index_put_((index,), src(*args))
                return base

        device = "cuda"
        torch.manual_seed(0)
        base_size = (32, 32, 26) if op == "select_scatter" else (26, 32, 32)
        base = torch.rand(base_size, dtype=torch.float32, device=device)
        args = [
            torch.rand((32, 32, 26), dtype=torch.float32, device=device)
            for _ in range(5)
        ]
        expected = fn(base.clone(), *args)

        torch._dynamo.reset()
        metrics.reset()
        with DeterministicGuard(False), fresh_inductor_cache():
            compiled = torch.compile(fn, backend="inductor", fullgraph=True)
            actual = compiled(base.clone(), *args)

        self.assertEqual(expected, actual)
        # src must not be inlined into the scatter loop, which would recompute it
        # once per element of the broadcast dim.
        self.assertEqual(metrics.ir_nodes_pre_fusion, 2)
        self.assertEqual(metrics.generated_kernel_count, 2)


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
