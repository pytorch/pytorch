# Owner(s): ["module: inductor"]

"""Tests for the to_padded_blocked lowering."""

import torch
import torch._inductor.config as inductor_config
from torch._inductor import inductor_prims
from torch._inductor.compile_fx import compile_fx
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


LOGICAL_ROW_CHUNK = 96
PHYSICAL_ROW_CHUNK = 128
ROW_INNER = 32
COL_CHUNK = 4
COL_INNER = 2
PADDING_VALUE = 127


def _padded_dims(rows, cols):
    padded_rows = (
        (rows + LOGICAL_ROW_CHUNK - 1) // LOGICAL_ROW_CHUNK * PHYSICAL_ROW_CHUNK
    )
    padded_cols = (cols + COL_CHUNK - 1) // COL_CHUNK * COL_CHUNK
    return padded_rows, padded_cols


def _scatter(values):
    return inductor_prims.to_padded_blocked(
        values,
        LOGICAL_ROW_CHUNK,
        PHYSICAL_ROW_CHUNK,
        ROW_INNER,
        COL_INNER,
        PADDING_VALUE,
    )


@inductor_config.patch("force_disable_caches", True)
@instantiate_parametrized_tests
class ToPaddedBlockedTest(TestCase):
    __unittest_skip__ = not HAS_GPU

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    @parametrize("shape", [(7, 96), (96, 4), (100, 8), (2048, 96)])
    @parametrize("dynamic", [False, True])
    def test_matches_eager(self, shape, dynamic):
        rows, cols = shape
        padded_rows, padded_cols = _padded_dims(rows, cols)
        values = torch.randint(0, 255, shape, dtype=torch.uint8, device=GPU_TYPE)
        expected = _scatter(values)

        def f(values):
            return _scatter(values)

        if dynamic:
            torch._dynamo.mark_dynamic(values, 0, min=1, max=1 << 14)
        actual, code = run_and_get_code(torch.compile(f, fullgraph=True), values)
        self.assertEqual(actual, expected)
        self.assertEqual(actual.numel(), padded_rows * padded_cols)
        self.assertNotIn("index_put", "".join(code))

    @parametrize("shape", [(7, 96), (100, 8)])
    def test_padding_positions_hold_padding_value(self, shape):
        rows, cols = shape
        padded_rows, padded_cols = _padded_dims(rows, cols)
        values = torch.zeros(shape, dtype=torch.uint8, device=GPU_TYPE)

        def f(values):
            return _scatter(values)

        actual = torch.compile(f, fullgraph=True)(values)
        # Sources were all zero, so every remaining slot must be padding.
        self.assertEqual(int((actual != 0).sum()), actual.numel() - rows * cols)
        self.assertTrue(bool((actual[actual != 0] == PADDING_VALUE).all()))

    def test_dynamic_shapes_emit_no_layout_guards(self):
        shape_envs = []

        def backend(gm, example_inputs):
            ctx = torch._guards.TracingContext.try_get()
            shape_envs.append(ctx.fake_mode.shape_env)
            return compile_fx(gm, example_inputs)

        run = torch.compile(_scatter, fullgraph=True, backend=backend, dynamic=True)
        # Shapes vary in magnitude and in divisibility by both chunk sizes.
        for shape in [(2048, 96), (100, 8), (7, 96), (2049, 100)]:
            values = torch.randint(0, 255, shape, dtype=torch.uint8, device=GPU_TYPE)
            torch._dynamo.mark_dynamic(values, 0, min=2, max=1 << 20)
            torch._dynamo.mark_dynamic(values, 1, min=2, max=1 << 20)
            self.assertEqual(run(values), _scatter(values))

        # One compile serves every shape: the padded extents stay symbolic, so
        # nothing specializes on size or on divisibility by a chunk.
        self.assertEqual(len(shape_envs), 1)
        env = shape_envs[0]
        self.assertEqual(env.replacements, {})
        deferred = env.deferred_runtime_asserts
        self.assertEqual([ra for v in deferred.values() for ra in v], [])
        # Whatever remains must be inductor's per-buffer 32-bit indexing bound,
        # which every lowering emits and which cannot force a recompile.
        for guard in env.guards:
            self.assertTrue(str(guard.expr).endswith("<= 2147483647"), str(guard.expr))

    @parametrize(
        "logical_row_chunk,physical_row_chunk,row_inner",
        [
            (128, 96, 32),  # physical chunk smaller than the logical chunk
            (96, 128, 48),  # physical chunk not a multiple of 2 * row_inner
        ],
    )
    def test_invalid_layout_rejected(
        self, logical_row_chunk, physical_row_chunk, row_inner
    ):
        values = torch.zeros((97, 4), dtype=torch.uint8, device=GPU_TYPE)

        def f(values):
            return inductor_prims.to_padded_blocked(
                values,
                logical_row_chunk,
                physical_row_chunk,
                row_inner,
                COL_INNER,
                PADDING_VALUE,
            )

        with self.assertRaisesRegex(Exception, "invalid blocked layout parameters"):
            torch.compile(f, fullgraph=True)(values)


if __name__ == "__main__":
    run_tests()
