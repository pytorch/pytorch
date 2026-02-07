# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import copy
import re
import unittest
import warnings

import torch
import torch._dynamo
import torch.distributed as dist
import torch.testing._internal.common_methods_invocations as common_ops
from torch.distributed._local_tensor import LocalTensorMode, reconcile_args
from torch.distributed.tensor import (
    distribute_tensor,
    DTensor,
    init_device_mesh,
    Replicate,
    Shard,
)
from torch.distributed.tensor._ops.single_dim_strategy import _ShardingPlaceholder
from torch.overrides import resolve_name
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
)
from torch.testing._internal.common_methods_invocations import DecorateInfo, op_db
from torch.testing._internal.common_utils import run_tests, suppress_warnings, TestCase
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorConverter,
    DTensorOpTestBase,
    validate_sharding_rule_sample,
)
from torch.testing._internal.distributed._tensor.dtensor_xfails import (
    dtensor_compiled_xfails,
    dtensor_xfails,
    multi_threaded_xfails,
    no_strategy_xfails,
    unbacked_dtensor_xfails,
)
from torch.utils import _pytree as pytree
from torch.utils._debug_mode import _OpCall, DebugMode
from torch.utils._pytree import tree_flatten, tree_map


# rewrite common size variables to sth can be sharded evenly
# we can enable uneven shards later, but need to adjust more on
# sample inputs (i.e. view/reshape need to adjust shape size as well)
common_ops.L = 24
common_ops.M = 12
common_ops.S = 4
common_ops.XS = 2


def skipOps(op_db, test_case_name, base_test_name, to_skip):
    all_opinfos = op_db
    for xfail in to_skip:
        op_name, variant_name, device_type, dtypes, expected_failure = xfail
        matching_opinfos = [
            o
            for o in all_opinfos
            if o.name == op_name and o.variant_test_name == variant_name
        ]
        assert len(matching_opinfos) >= 1, f"Couldn't find OpInfo for {xfail}"
        for opinfo in matching_opinfos:
            decorators = list(opinfo.decorators)
            if expected_failure:
                decorator = DecorateInfo(
                    unittest.expectedFailure,
                    test_case_name,
                    base_test_name,
                    device_type=device_type,
                    dtypes=dtypes,
                )
                decorators.append(decorator)
            else:
                decorator = DecorateInfo(
                    unittest.skip("Skipped!"),
                    test_case_name,
                    base_test_name,
                    device_type=device_type,
                    dtypes=dtypes,
                )
                decorators.append(decorator)
            opinfo.decorators = tuple(decorators)

    # This decorator doesn't modify fn in any way
    def wrapped(fn):
        return fn

    return wrapped


def repurpose_ops(op_db, base_test_name, derived_test_name):
    """
    Copies op info database and for the decorators that applied to base test class updates
    them to apply to derived test class. The class update is required because decorators are applied
    only if the class name matches (it doesn't consider base classes).

    Specifically we use this function to create two test classes (one for multi-threaded and one for
    local tensor flavors) that share common test body but different rules for skip or fail.

    Args:
        op_db: List of OpInfo objects to be repurposed.
        base_test_name: The original test class name to be replaced.
        derived_test_name: The new test class name to set in decorators.

    Returns:
        list: A new list of OpInfo objects with updated target class names for the
        decorator.
    """
    repurposed_ops = []
    for opinfo in op_db:
        opinfo_copy = copy.deepcopy(opinfo)
        for decorator in list(opinfo_copy.decorators):
            if hasattr(decorator, "cls_name") and decorator.cls_name == base_test_name:
                decorator.cls_name = derived_test_name
        repurposed_ops.append(opinfo_copy)
    return repurposed_ops


# Add a list of ops that are currently failing BW pass
skip_bw = [
    None,  # corresponds to the transpose ops 'H' and 'T'
    "torch.bucketize",
    "torch.conj_physical",
    "torch.eq",
    "torch.isfinite",
    "torch.isnan",
]


OP_DB_WORLD_SIZE = 4
# DEVICE_TYPE = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() >= OP_DB_WORLD_SIZE else "cpu"
# TODO: debug cuda illegal memory access issue and re-enable cuda tests
DEVICE_TYPE = "cpu"


class TestDTensorOps(TestCase):
    __test__ = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.__test__ = True

    @property
    def world_size(self) -> int:
        return OP_DB_WORLD_SIZE

    def iter_valid_samples(
        self,
        op,
        dtype,
        requires_grad=False,
        sample_filter=None,
        needs_deepcopy=False,
    ):
        """
        Iterate over valid samples for an op, yielding (args, kwargs) tuples.

        Args:
            op: The OpInfo object
            dtype: The dtype to use for sample inputs
            requires_grad: Whether tensors should require grad
            sample_filter: Optional callable(args, kwargs) -> bool to filter samples
            needs_deepcopy: If True, yields deepcopied args/kwargs and skips
                            samples that can't be deepcopied
        """
        samples = op.sample_inputs(DEVICE_TYPE, dtype, requires_grad=requires_grad)
        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs

            if sample_filter and not sample_filter(args, kwargs):
                continue

            if needs_deepcopy:
                try:
                    args = copy.deepcopy(args)
                    kwargs = copy.deepcopy(kwargs)
                except NotImplementedError:
                    continue

            yield args, kwargs

    def run_opinfo_test(self, dtype, op, requires_grad=True, sample_filter=None):
        self.mesh = init_device_mesh(DEVICE_TYPE, (self.world_size,))

        def test():
            for args, kwargs in self.iter_valid_samples(
                op, dtype, requires_grad=requires_grad, sample_filter=sample_filter
            ):
                self.run_dtensor_crossref(op.op, args, kwargs)

        self.check_dtensor_func(test, op)

    def assert_ref_dtensor_equal(self, dtensor_rs, rs):
        flat_dtensor_rs = pytree.tree_leaves(dtensor_rs)
        flat_rs = pytree.tree_leaves(rs)
        self.assertEqual(len(flat_dtensor_rs), len(flat_rs))
        for dtensor_r, r in zip(flat_dtensor_rs, flat_rs):
            if not isinstance(r, torch.Tensor):
                continue

            self.assertIsInstance(dtensor_r, torch.Tensor)
            self.assertEqualOnRank(
                dtensor_r.shape,
                r.shape,
                f"Shape mismatch! original shape:{r.shape}, dtensor shape: {dtensor_r.shape}",
            )
            self.assertEqualOnRank(
                dtensor_r.requires_grad,
                r.requires_grad,
                "op result requires_grad mismatch!"
                f"original requires_grad: {r.requires_grad}, "
                f"dtensor requires_grad: {dtensor_r.requires_grad}",
            )

            self.assertEqualOnRank(dtensor_r, r)

    def assertEqualOnRank(self, x, y, msg=None, *, rank=0) -> None:
        raise NotImplementedError

    def run_dtensor_crossref(self, func, args, kwargs):
        to_dtensor = DTensorConverter(self.mesh, args, kwargs)

        def concat_res_if_necessary(func, res: object) -> object:
            # concat the result on corresponding dim for ops like
            # split, so that we can call backward on a single tensor
            if (resolve_name(func) is not None) and ("split" in resolve_name(func)):
                dim = args[2] if len(args) == 3 else 0
                return torch.cat(res, dim=dim)
            else:
                return res

        # TODO: also handle cases where func raise an exception
        op_args, op_kwargs = reconcile_args(args, kwargs)
        rs = func(*op_args, **op_kwargs)
        rs = concat_res_if_necessary(func, rs)

        def to_replicate(e: object) -> object:
            return e.full_tensor() if isinstance(e, DTensor) else e

        # Suppress warnings, this doesn't matter for test_meta.py
        # but it does matter if you want to use this decorator
        # for cross-ref testing, as some tests may be looking at
        # errors
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # for every comb of sharding choices, we test if it works
            for dtensor_args, dtensor_kwargs in to_dtensor:
                # Only attempt if we managed to convert all tensors to DTensor
                # (if any of them failed, we're in a mixed tensor situation and
                # this is not allowed in DTensor)
                try:
                    if to_dtensor.successful():
                        # Handle special cases first if there's any
                        # Suppress warnings, this doesn't matter for test_meta.py
                        # but it does matter if you want to use this decorator
                        # for cross-ref testing, as some tests may be looking at
                        # errors
                        dtensor_rs = func(*dtensor_args, **dtensor_kwargs)

                        # we need to skip tests containing tensors of zero elements for now.
                        # see issue: https://github.com/pytorch/PiPPy/issues/470
                        # TODO remove this once issue above fixed.
                        flat_args = pytree.tree_leaves(dtensor_rs)
                        if any(
                            isinstance(e, torch.Tensor) and e.numel() == 0
                            for e in flat_args
                        ):
                            continue

                        # redistribute/all_gather the results to compare with normal output
                        dtensor_rs = tree_map(to_replicate, dtensor_rs)
                        dtensor_rs = concat_res_if_necessary(func, dtensor_rs)
                        try:
                            if resolve_name(func) not in skip_bw:
                                if isinstance(dtensor_rs, DTensor):
                                    dtensor_rs.to_local().sum().backward()
                                elif isinstance(dtensor_rs, tuple):
                                    dtensor_rs[0].to_local().sum().backward()

                        except Exception as e:
                            # TODO(anj): Remove this guard exception after gaining more confidence.
                            if torch.distributed.get_rank() == 0:
                                print(
                                    f"failed to run BW: {resolve_name(func)}, {func}, {str(e)})"
                                )
                        self.assert_ref_dtensor_equal(dtensor_rs, rs)
                    else:
                        raise RuntimeError(
                            f"Failed to convert args to DTensor; "
                            f"originally (*{args}, **{kwargs})"
                        )
                except Exception as e:
                    raise RuntimeError(
                        f"{str(e)}\n\nFailed to run: {resolve_name(func)}, with (*{dtensor_args}, **{dtensor_kwargs})"
                    ) from e
        return rs

    def check_dtensor_func(self, test_func, opinfo, dry_run=False):
        try:
            test_func()
        except Exception:
            if not dry_run:
                raise
            if dist.get_rank() == 0:
                if opinfo.variant_test_name:
                    print(f"xfail('{opinfo.name}', '{opinfo.variant_test_name}'),")
                else:
                    print(f"xfail('{opinfo.name}'),")

    def run_one_hot(self):
        ops = [op for op in op_db if op.name == "nn.functional.one_hot"]
        assert len(ops) == 1
        op = ops[0]
        # num_classes = -1 appears to have a bug with dtensor.max().item()
        self.run_opinfo_test(
            torch.int64,
            op,
            requires_grad=False,
            sample_filter=lambda args, kwargs: kwargs.get("num_classes") != -1,
        )

    def run_mean(self):
        self.mesh = init_device_mesh(DEVICE_TYPE, (self.world_size,))

        shape = [2 * self.world_size + 1, 2 * self.world_size]
        tensor = (
            torch.arange(shape[0] * shape[1], dtype=torch.float32)
            .reshape(shape)
            .to(DEVICE_TYPE)
        )

        for is_evenly_shardable in [True, False]:
            if is_evenly_shardable:
                placement = [Shard(1)]
                reduce_dim = 1
            else:
                placement = [Shard(0)]
                reduce_dim = 0
            dtensor = distribute_tensor(tensor, self.mesh, placement)

            with DebugMode(record_torchfunction=False) as debug_mode:
                mean = dtensor.mean(dim=reduce_dim)
                full_tensor = mean.full_tensor()

            self.assertEqual(full_tensor, tensor.mean(dim=reduce_dim))

            if is_evenly_shardable:
                self.assertTrue("P(avg)->R" in debug_mode.debug_string())
            else:
                self.assertTrue("S(0)->R" in debug_mode.debug_string())

    def test_embedding_error_msg(self):
        self.mesh_2d = init_device_mesh(
            DEVICE_TYPE, (2, self.world_size // 2), mesh_dim_names=("dp", "tp")
        )
        self.mesh_1d = self.mesh_2d["tp"]

        weight_global = torch.randn(2048, 256, device=DEVICE_TYPE)
        weight_dtensor = distribute_tensor(weight_global, self.mesh_1d, [Shard(0)])

        input_global = torch.randint(0, 2048, (16, 2048), device=DEVICE_TYPE)
        input_dtensor = distribute_tensor(
            input_global, self.mesh_2d, [Shard(0), Replicate()]
        )

        expected_error_msg = (
            "Sharding propagation failed for aten.embedding.default"
            "(Spec(f32[2048, 256](S(0))), Spec(i64[16, 2048](S(0)R))) "
            "on DeviceMesh((dp=2, tp=2), "
        )
        with self.assertRaisesRegex(RuntimeError, re.escape(expected_error_msg)):
            _ = torch.ops.aten.embedding.default(weight_dtensor, input_dtensor)


class TestMultiThreadedDTensorOps(DTensorOpTestBase, TestDTensorOps):
    _op_db = repurpose_ops(op_db, "TestDTensorOps", "TestMultiThreadedDTensorOps")

    @suppress_warnings
    @ops(_op_db, allowed_dtypes=(torch.float,))
    @skipOps(
        _op_db,
        "TestMultiThreadedDTensorOps",
        "test_dtensor_op_db",
        dtensor_xfails | no_strategy_xfails | multi_threaded_xfails,
    )
    def test_dtensor_op_db(self, dtype, op):
        self.run_opinfo_test(dtype, op)

    def test_mean(self):
        self.run_mean()

    def test_one_hot(self):
        self.run_one_hot()


class TestLocalDTensorOps(TestDTensorOps):
    _op_db = repurpose_ops(op_db, "TestDTensorOps", "TestLocalDTensorOps")

    def setUp(self) -> None:
        super().setUp()
        torch.distributed.init_process_group("fake", rank=0, world_size=self.world_size)
        self.fake_pg = torch.distributed.distributed_c10d._get_default_group()

    def tearDown(self):
        super().tearDown()
        # Clear sharding propagation cache to avoid stale mesh references
        # between tests that destroy and recreate process groups
        from torch.distributed.tensor.debug import _clear_sharding_prop_cache

        _clear_sharding_prop_cache()
        try:
            dist.destroy_process_group()
        except AssertionError:
            pass

    @suppress_warnings
    @ops(_op_db, allowed_dtypes=(torch.float,))
    @skipOps(
        _op_db,
        "TestLocalDTensorOps",
        "test_dtensor_op_db",
        dtensor_xfails | no_strategy_xfails,
    )
    def test_dtensor_op_db(self, dtype, op):
        self.run_opinfo_test(dtype, op)

    def test_mean(self):
        with LocalTensorMode(frozenset(range(self.world_size))):
            self.run_mean()

    def test_one_hot(self):
        self.run_one_hot()

    def run_opinfo_test(self, dtype, op, requires_grad=True, sample_filter=None):
        with LocalTensorMode(frozenset(range(self.world_size))):
            super().run_opinfo_test(dtype, op, requires_grad, sample_filter)

    def assertEqualOnRank(self, x, y, msg=None, *, rank=0):
        self.assertEqual(x, y, msg)


class TestUnbackedDTensorOps(TestDTensorOps):
    """
    Test suite for DTensor ops with unbacked symints.

    This runs correctness tests with tensor dimensions marked as unbacked
    and the op compiled with fullgraph=True to catch DDEs during tracing.
    """

    _op_db = repurpose_ops(op_db, "TestDTensorOps", "TestUnbackedDTensorOps")

    def setUp(self) -> None:
        super().setUp()
        torch.distributed.init_process_group("fake", rank=0, world_size=self.world_size)

    def tearDown(self):
        super().tearDown()
        from torch.distributed.tensor.debug import _clear_sharding_prop_cache

        _clear_sharding_prop_cache()
        torch._dynamo.reset()
        try:
            dist.destroy_process_group()
        except AssertionError:
            pass

    def assertEqualOnRank(self, x, y, msg=None, *, rank=0):
        self.assertEqual(x, y, msg)

    def _has_valid_unbacked_dims(self, t: torch.Tensor) -> bool:
        """Check if tensor has dimensions that can be marked as unbacked."""
        return t.ndim > 0 and any(s >= 2 for s in t.shape)

    def _sample_has_valid_unbacked_dims(self, args, kwargs) -> bool:
        """Check if any tensor in args/kwargs has valid unbacked dimensions."""
        all_tensors = [
            x for x in tree_flatten((args, kwargs))[0] if isinstance(x, torch.Tensor)
        ]
        return any(self._has_valid_unbacked_dims(t) for t in all_tensors)

    def _mark_unbacked(self, t: torch.Tensor) -> None:
        """Mark all eligible dimensions of a tensor as unbacked."""
        for i in range(t.ndim):
            if t.shape[i] >= 2:
                torch._dynamo.decorators.mark_unbacked(t, i)

    def run_dtensor_crossref(self, func, args, kwargs):
        """
        Override to add unbacked marking and fullgraph compilation.

        Same as parent but:
        1. Marks DTensor dimensions as unbacked before running
        2. Wraps the op in @torch.compile(backend="eager", fullgraph=True)
        """
        to_dtensor = DTensorConverter(self.mesh, args, kwargs)

        def concat_res_if_necessary(func, res: object) -> object:
            if (resolve_name(func) is not None) and ("split" in resolve_name(func)):
                dim = args[2] if len(args) == 3 else 0
                return torch.cat(res, dim=dim)
            return res

        op_args, op_kwargs = reconcile_args(args, kwargs)
        rs = func(*op_args, **op_kwargs)
        rs = concat_res_if_necessary(func, rs)

        def to_replicate(e: object) -> object:
            return e.full_tensor() if isinstance(e, DTensor) else e

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for dtensor_args, dtensor_kwargs in to_dtensor:
                try:
                    if not to_dtensor.successful():
                        raise RuntimeError(
                            f"Failed to convert args to DTensor; "
                            f"originally (*{args}, **{kwargs})"
                        )

                    pytree.tree_map_only(
                        lambda x: isinstance(x, DTensor)
                        and self._has_valid_unbacked_dims(x),
                        self._mark_unbacked,
                        (dtensor_args, dtensor_kwargs),
                    )

                    # Compile with fullgraph=True to catch DDEs
                    torch._dynamo.reset()

                    @torch.compile(backend="eager", fullgraph=True)
                    def compiled_func(*a, **kw):
                        return func(*a, **kw)

                    compiled_func(*dtensor_args, **dtensor_kwargs)

                except Exception as e:
                    raise RuntimeError(
                        f"{str(e)}\n\nFailed to run: {resolve_name(func)}, "
                        f"with (*{dtensor_args}, **{dtensor_kwargs})"
                    ) from e
        return rs

    @suppress_warnings
    @ops(_op_db, allowed_dtypes=(torch.float,))
    @skipOps(
        _op_db,
        "TestUnbackedDTensorOps",
        "test_unbacked_dtensor_op_db",
        unbacked_dtensor_xfails,
    )
    def test_unbacked_dtensor_op_db(self, dtype, op):
        # Filter to samples with valid unbacked dimensions
        self.run_opinfo_test(
            dtype,
            op,
            requires_grad=False,
            sample_filter=self._sample_has_valid_unbacked_dims,
        )


class TestSingleDimStrategies(DTensorOpTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def _extract_aten_op_and_args(self, torch_op, args, kwargs):
        with DebugMode(store_original_args=True) as debug_mode:
            try:
                torch_op(*args, **kwargs)
            except Exception:
                self.skipTest(f"Op {torch_op} failed on replicated DTensors")

        for op in debug_mode.operators:
            if isinstance(op, _OpCall) and "aten" in str(op.op):
                return op.op, op.args, op.kwargs

        self.skipTest(f"Op {torch_op} failed to extract aten op")

    @suppress_warnings
    @ops(op_db, allowed_dtypes=(torch.float,))
    def test_single_dim_strategy(self, dtype, op):
        torch.manual_seed(42)
        mesh = init_device_mesh(DEVICE_TYPE, (self.world_size,))
        sharding_prop = DTensor._op_dispatcher.sharding_propagator

        try:
            samples = list(op.sample_inputs(DEVICE_TYPE, dtype, requires_grad=False))
        except Exception:
            self.skipTest(f"Failed to get sample inputs for {op.name}")
        if not samples:
            self.skipTest(f"No sample inputs for {op.name}")

        sample = samples[0]
        args = (sample.input,) + tuple(sample.args)

        # create Replicated DTensors
        try:
            dtensor_args, dtensor_kwargs = pytree.tree_map_only(
                torch.Tensor,
                lambda t: distribute_tensor(t, mesh, (Replicate(),)),
                (args, sample.kwargs),
            )
        except Exception:
            self.skipTest(f"Failed to create replicate DTensors for {op.name}")

        # extract aten op/args/kwargs
        aten_op, aten_args, aten_kwargs = self._extract_aten_op_and_args(
            op.op, dtensor_args, dtensor_kwargs
        )

        single_dim_strats = sharding_prop.op_single_dim_strategy_funcs
        if aten_op not in single_dim_strats:
            self.skipTest(f"No single-dim strategy for {op.name}: {aten_op}")

        # extract tensor_meta, full tensors
        all_tensor_meta = []

        def _collect_tensor_meta(dt):
            meta = dt._spec.tensor_meta
            all_tensor_meta.append(meta)
            return meta

        args_meta, kwargs_meta = pytree.tree_map_only(
            DTensor, _collect_tensor_meta, (aten_args, aten_kwargs)
        )
        full_args, full_kwargs = pytree.tree_map_only(
            torch.Tensor, lambda t: t.full_tensor(), (aten_args, aten_kwargs)
        )

        # enumerate strategies, replace placeholders with Shard
        strategies = pytree.tree_map_only(
            _ShardingPlaceholder,
            lambda s: Shard(s.dim),
            single_dim_strats[aten_op](aten_op, args_meta, kwargs_meta),
        )
        # TODO(pianpwk): handle multi-output once that lands for single-dim
        for output_placement, *input_placements in strategies:
            # skip strategies with invalid shards
            def is_invalid_shard(meta, p):
                ndim = len(meta.shape)
                if (
                    not isinstance(p, Shard)
                    or ndim == 0
                    or p.dim >= ndim
                    or meta.shape[p.dim] == 0
                    or meta.shape[p.dim] % self.world_size != 0
                ):
                    return True
                return False

            if any(
                is_invalid_shard(t, p)
                for t, p in zip(all_tensor_meta, input_placements)
            ):
                continue

            # add the validate_sharding_rule function
            self.assertTrue(
                validate_sharding_rule_sample(
                    aten_op,
                    full_args,
                    full_kwargs,
                    input_placements,
                    (output_placement,),
                    mesh,
                ),
                f"{op.name}: {input_placements} -> {(output_placement,)} failed",
            )


class TestCompiledDTensorOps(TestDTensorOps):
    """
    Test DTensor ops compile successfully with aot_eager backend.
    Uses fake PG for speed - focuses on compilation, not output correctness.
    """

    _op_db = repurpose_ops(op_db, "TestDTensorOps", "TestCompiledDTensorOps")

    def setUp(self) -> None:
        super().setUp()
        torch.distributed.init_process_group("fake", rank=0, world_size=self.world_size)

    def tearDown(self):
        super().tearDown()
        from torch.distributed.tensor.debug import _clear_sharding_prop_cache

        _clear_sharding_prop_cache()
        torch._dynamo.reset()
        try:
            dist.destroy_process_group()
        except AssertionError:
            pass

    def assertEqualOnRank(self, x, y, msg=None, *, rank=0):
        # Skip output comparison - we only care that compilation succeeds
        pass

    def run_dtensor_crossref(self, func, args, kwargs):
        """
        Override to compile with aot_eager and verify compilation succeeds.
        Does not check output correctness.
        """
        to_dtensor = DTensorConverter(self.mesh, args, kwargs)

        for dtensor_args, dtensor_kwargs in to_dtensor:
            if not to_dtensor.successful():
                continue

            torch._dynamo.reset()

            @torch.compile(backend="aot_eager", fullgraph=True)
            def compiled_func(*a, **kw):
                return func(*a, **kw)

            # Just run - if it compiles and runs without error, we pass
            compiled_func(*dtensor_args, **dtensor_kwargs)

    @suppress_warnings
    @ops(_op_db, allowed_dtypes=(torch.float,))
    @skipOps(
        _op_db,
        "TestCompiledDTensorOps",
        "test_compiled_dtensor_op_db",
        dtensor_compiled_xfails,
    )
    def test_compiled_dtensor_op_db(self, dtype, op):
        self.run_opinfo_test(dtype, op, requires_grad=False)


# only instantiate tests for DEVICE_TYPE alone (i.e. either CPU or GPU)
instantiate_device_type_tests(
    TestMultiThreadedDTensorOps, globals(), only_for=(DEVICE_TYPE,)
)

instantiate_device_type_tests(TestLocalDTensorOps, globals(), only_for=(DEVICE_TYPE,))

instantiate_device_type_tests(
    TestUnbackedDTensorOps, globals(), only_for=(DEVICE_TYPE,)
)

instantiate_device_type_tests(
    TestSingleDimStrategies, globals(), only_for=(DEVICE_TYPE,)
)

instantiate_device_type_tests(
    TestCompiledDTensorOps, globals(), only_for=(DEVICE_TYPE,)
)

if __name__ == "__main__":
    run_tests()
