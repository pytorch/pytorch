# Owner(s): ["oncall: jit"]
import copy
import gc
import weakref
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import sympy

import torch
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.source import LocalSource
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx._graph_pickler import _ShapeEnvPickleData, _UnpickleState, GraphPickler
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import (
    DimDynamic,
    ShapeEnv,
    StatelessSymbolicContext,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils._sympy.value_ranges import ValueRanges


class TestSymNodeCache(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def setUp(self):
        super().setUp()
        self.previous = torch._C._symnode_cache_enabled()
        torch._C._set_symnode_cache_enabled(True)
        self.addCleanup(torch._C._set_symnode_cache_enabled, self.previous)

    def symbol(self, env, name, hint):
        expr = env.create_symbol(
            hint, LocalSource(name), dynamic_dim=DimDynamic.DYNAMIC
        )
        return env.create_symintnode(expr, hint=hint)

    def test_hits_preserve_result_identity(self):
        env = ShapeEnv()
        size = self.symbol(env, "size", 7)
        before = torch._C._symnode_cache_stats()["hits"]
        results = [torch.empty((size, 3), device="meta").numel() for _ in range(8)]
        self.assertGreater(torch._C._symnode_cache_stats()["hits"], before)
        self.assertEqual(
            [value.node.expr for value in results], [size.node.expr * 3] * 8
        )
        self.assertEqual(len({id(value.node) for value in results}), 8)

    @parametrize("operation", ("permute", "split"))
    def test_views_do_not_snapshot_cache_metadata(self, operation):
        sample = torch.empty(5, 5, 5)
        with torch._dynamo.config.patch(use_cpp_fake_tensor=True):
            mode = FakeTensorMode(shape_env=ShapeEnv(), static_shapes=False)
            tensor = mode.from_tensor(
                sample,
                source=LocalSource("input"),
                symbolic_context=StatelessSymbolicContext(
                    dynamic_sizes=[DimDynamic.DYNAMIC] * sample.ndim
                ),
            )
            with mode, enable_python_dispatcher():
                before = torch._C._symnode_cache_stats()["metadata_snapshots"]
                for _ in range(5):
                    if operation == "permute":
                        result = tensor.permute(2, 0, 1)
                    else:
                        result = tensor.split([1, 3, 1])
                self.assertEqual(
                    torch._C._symnode_cache_stats()["metadata_snapshots"], before
                )
                if operation == "permute":
                    self.assertEqual(result.shape, (5, 5, 5))
                else:
                    self.assertEqual(
                        [part.shape for part in result],
                        [(1, 5, 5), (3, 5, 5), (1, 5, 5)],
                    )

    def test_replacement_invalidates_cached_result(self):
        env = ShapeEnv()
        size = self.symbol(env, "size", 7)
        other = self.symbol(env, "other", 7)
        for _ in range(3):
            torch.empty((size, 3), device="meta").numel()
        env._set_replacement(size.node.expr, other.node.expr, "cache test")
        actual = torch.empty((size, 3), device="meta").numel()
        self.assertEqual(actual.node._expr, other.node._expr * 3)

    def test_range_invalidates_cached_result(self):
        env = ShapeEnv()
        size = self.symbol(env, "size", 10)
        a = size.node._expr
        b = self.symbol(env, "b", 9).node._expr
        c = self.symbol(env, "c", 4).node._expr
        d = self.symbol(env, "d", 5).node._expr
        env._set_replacement(b, c + d, "cache test")
        env._update_var_to_range(a, ValueRanges(2, 100))
        env._set_replacement(a, b + 1, "cache test")

        # Resolving a to c + d + 1 is rejected until it fits a's range.
        tensor = torch.empty_strided((2,), (size,), device="meta")
        for _ in range(3):
            self.assertEqual(tensor.unsqueeze(0).stride(0).node._expr, 2 * b + 2)
        env._update_var_to_range(c, ValueRanges(2, 50))
        env._update_var_to_range(d, ValueRanges(2, 49))
        result = tensor.unsqueeze(0).stride(0)
        self.assertEqual(result.node._expr, 2 * c + 2 * d + 2)

    def test_range_and_guard_parity(self):
        expressions = []
        guards = []
        for enabled in (False, True):
            torch._C._set_symnode_cache_enabled(enabled)
            env = ShapeEnv()
            size = self.symbol(env, "size", 7)
            for _ in range(3):
                torch.empty((size, 3), device="meta").numel()
            env._update_var_to_range(size.node.expr, ValueRanges(2, 20))
            result = torch.empty((size, 3), device="meta").numel()
            self.assertTrue(result > 10)
            expressions.append(str(result.node.expr))
            guards.append([str(guard.expr) for guard in env.guards])
        self.assertEqual(expressions[0], expressions[1])
        self.assertEqual(guards[0], guards[1])

    def test_cached_concrete_query_is_invalidated(self):
        env = ShapeEnv()
        size = self.symbol(env, "size", 7)
        tensor = torch.empty((size, 3), device="meta")
        for _ in range(3):
            self.assertIsInstance(tensor.clone().numel(), torch.SymInt)
        with env.patch_source_specialization(LocalSource("size"), lambda s: s == 7):
            for _ in range(3):
                self.assertEqual(tensor.clone().numel(), 21)
        result = tensor.clone().numel()
        self.assertIsInstance(result, torch.SymInt)
        self.assertEqual(result.node.expr, size.node.expr * 3)

    def test_graph_pickler_invalidates_retained_tensor(self):
        env = ShapeEnv()
        mode = FakeTensorMode(shape_env=env)
        cache = env._native_symnode_cache
        size = self.symbol(env, "size", 7)
        tensor = torch.empty((size, 3), device="meta")
        for _ in range(3):
            self.assertIsInstance(tensor.clone().numel(), torch.SymInt)
        with env.patch_source_specialization(LocalSource("size"), lambda s: s == 7):
            for _ in range(3):
                self.assertEqual(tensor.clone().numel(), 21)
            self.assertIs(GraphPickler.loads(GraphPickler.dumps(env), mode), env)
            self.assertIs(env._native_symnode_cache, cache)
            for _ in range(3):
                self.assertEqual(tensor.clone().numel(), 21)
        result = tensor.clone().numel()
        self.assertIsInstance(result, torch.SymInt)
        self.assertEqual(result.node._expr, 3 * size.node._expr)

    def test_graph_pickler_restore_clears_concrete_query(self):
        env = ShapeEnv()
        mode = FakeTensorMode(shape_env=env)
        size = self.symbol(env, "size", 7)
        tensor = torch.empty((size, 3), device="meta")
        for _ in range(3):
            self.assertIsInstance(tensor.clone().numel(), torch.SymInt)
        data = GraphPickler.dumps(env)
        env._set_replacement(size.node._expr, sympy.Integer(7), "cache test")
        for _ in range(3):
            self.assertEqual(tensor.clone().numel(), 21)
        self.assertIs(GraphPickler.loads(data, mode), env)
        result = tensor.clone().numel()
        self.assertIsInstance(result, torch.SymInt)
        self.assertEqual(result.node._expr, 3 * size.node._expr)

    @parametrize("enabled", (False, True))
    def test_graph_pickler_ignores_serialized_native_cache(self, enabled):
        torch._C._set_symnode_cache_enabled(enabled)
        env = ShapeEnv()
        mode = FakeTensorMode(shape_env=env)
        cache = env._native_symnode_cache
        data = _ShapeEnvPickleData(env)
        self.assertNotIn("_native_symnode_cache", data.data)
        data.data["_native_symnode_cache"] = torch._C._SymNodeCache()
        self.assertIs(data.unpickle(_UnpickleState(mode)), env)
        self.assertIs(env._native_symnode_cache, cache)

    def test_concurrent_shared_adapter_initialization(self):
        env = ShapeEnv()
        size = self.symbol(env, "size", 7)
        torch.empty_strided((2,), (size,), device="meta").unsqueeze(0).stride(0)
        # empty_strided clones the stride adapter after computing storage size.
        tensor = torch.empty_strided((2,), (size,), device="meta")
        before = torch._C._symnode_cache_stats()
        barrier = Barrier(4)

        def query(_):
            barrier.wait(timeout=30)
            return tensor.unsqueeze(0).stride(0)

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(query, range(4)))
        after = torch._C._symnode_cache_stats()
        self.assertEqual(after["metadata_snapshots"], before["metadata_snapshots"] + 1)
        self.assertEqual(after["misses"], before["misses"])
        self.assertEqual(
            [value.node._expr for value in results], [2 * size.node._expr] * 4
        )
        self.assertEqual(len({id(value.node) for value in results}), 4)

    def test_concurrent_shared_adapter_materialization(self):
        env = ShapeEnv()
        size = self.symbol(env, "size", 7)
        torch.empty((2, size), device="meta").stride(0)
        # The contiguous stride is a lazy native multiplication cache hit.
        tensor = torch.empty((2, size), device="meta")
        before = torch._C._symnode_cache_stats()["materializations"]
        barrier = Barrier(4)

        def query(_):
            barrier.wait(timeout=30)
            return tensor.stride(0)

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(query, range(4)))
        self.assertEqual(
            torch._C._symnode_cache_stats()["materializations"], before + 1
        )
        self.assertEqual([value.node._expr for value in results], [size.node._expr] * 4)
        self.assertEqual(len({id(value.node) for value in results}), 1)

    def test_proxy_and_unbacked_paths(self):
        def fn(x):
            first = x.view(x.shape[0], -1)
            second = x.view(x.shape[0], -1)
            return first + second

        sample = torch.randn(5, 3, 2)
        graph = make_fx(fn, tracing_mode="symbolic")(sample)
        self.assertEqual(graph(sample), fn(sample))
        changed = torch.randn(8, 3, 2)
        self.assertEqual(graph(changed), fn(changed))
        env = ShapeEnv()
        size = env.create_unbacked_symint()
        result = torch.empty((size, 3), device="meta").numel()
        self.assertEqual(result.node.expr, size.node.expr * sympy.Integer(3))

    def test_cache_does_not_retain_environment(self):
        env = ShapeEnv()
        size = self.symbol(env, "size", 7)
        for _ in range(3):
            torch.empty((size, 3), device="meta").numel()
        duplicate = copy.deepcopy(env._native_symnode_cache)
        duplicate.clear()
        reference = weakref.ref(env)
        del size, env
        ShapeEnv._inner_evaluate_expr.cache_clear()
        gc.collect()
        self.assertIsNone(reference())


instantiate_parametrized_tests(TestSymNodeCache)

if __name__ == "__main__":
    run_tests()
