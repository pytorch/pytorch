# Owner(s): ["module: inductor"]
import random
import types

import sympy

from torch._inductor import config
from torch._inductor.codegen.simd_kernel_features import SIMDKernelFeatures
from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.scheduler import _SetUnion
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import HardwareClassification


def _features(deps):
    node = types.SimpleNamespace(
        read_writes=types.SimpleNamespace(reads=deps, writes=[])
    )
    return types.SimpleNamespace(indexing_scheduler_nodes=lambda: [node])


class FastCompilePyTest(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_overflow_check_matches_slow_path(self):
        x0, x1, s0 = sympy.symbols("x0 x1 s0", integer=True, nonnegative=True)
        rng = random.Random(0)
        for _ in range(300):
            stride = rng.choice([1, 7, 1 << 12, 1 << 20, 1 << 31])
            offset = rng.choice([0, 5, (1 << 31) - 1, 1 << 31, -(1 << 31) - 1])
            index = stride * x0 + rng.choice([1, 3]) * x1 + offset
            size0 = rng.choice([1, 4, 1 << 11, s0])
            dep = MemoryDep("buf0", index, (x0, x1), (size0, rng.choice([1, 8])))
            results = []
            for fast in (False, True):
                with config.patch(fast_compile_py=fast):
                    results.append(
                        SIMDKernelFeatures.any_index_expr_overflows_int32(
                            _features([dep])
                        )
                    )
            self.assertEqual(results[0], results[1], msg=str(dep))

    def test_overflow_check_follows_shape_env_ranges(self):
        # A symbolic stride is bounded by the current ShapeEnv, which changes
        # between graphs; the same dep must not reuse an earlier answer.
        from torch._guards import tracing, TracingContext
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv
        from torch.utils._sympy.value_ranges import ValueRanges

        x0 = sympy.Symbol("x0", integer=True, nonnegative=True)
        s0 = sympy.Symbol("s0", integer=True, positive=True)
        dep = MemoryDep("buf0", s0 * x0, (x0,), (1 << 20,))
        results = []
        for s0_max in (None, 1 << 20):
            shape_env = ShapeEnv()
            if s0_max is not None:
                shape_env.var_to_range[s0] = ValueRanges(1, s0_max)
            with tracing(TracingContext(FakeTensorMode(shape_env=shape_env))):
                for fast in (False, True):
                    with config.patch(fast_compile_py=fast):
                        results.append(
                            SIMDKernelFeatures.any_index_expr_overflows_int32(
                                _features([dep])
                            )
                        )
        self.assertEqual(results, [False, False, True, True])

    def test_output_names_set_follows_graph_outputs(self):
        graph = types.SimpleNamespace(graph_outputs=["a"])
        graph.get_output_names = lambda: [str(o) for o in graph.graph_outputs]
        self.assertEqual(GraphLowering.output_names_set(graph), frozenset({"a"}))
        graph.graph_outputs = ["a", "b"]
        self.assertEqual(GraphLowering.output_names_set(graph), frozenset({"a", "b"}))

    def test_set_union_membership(self):
        union = _SetUnion({1, 2}, {3}, set())
        self.assertIn(3, union)
        self.assertIn(1, union)
        self.assertNotIn(4, union)


if __name__ == "__main__":
    run_tests()
