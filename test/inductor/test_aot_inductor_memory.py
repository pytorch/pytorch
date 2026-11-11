# Owner(s): ["module: inductor"]

import gc
import unittest

import torch
import torch._export.config
import torch.nn as nn
from torch._inductor.test_case import run_tests, TestCase
from torch.fx.experimental.proxy_tensor import _FAKE_TENSOR_ID_TO_PROXY_MAP_FOR_EXPORT
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU_AND_TRITON


def _export_small_module() -> None:
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU())
    torch.export.export(model, (torch.randn(2, 8),))


class ExportLeakDetectionMapTest(TestCase):
    """The debug map must only be populated when its consumers are enabled.

    Deliberately not GPU-gated: the behaviour under test is a pure-Python
    conditional, so gating it on GPU+Triton would leave it uncovered on the
    CPU-only runs that make up most of CI.
    """

    def setUp(self):
        super().setUp()
        # The map is cleared at the start of a trace, not the end, so a previous
        # test's trace can leave entries behind.
        _FAKE_TENSOR_ID_TO_PROXY_MAP_FOR_EXPORT.clear()
        self.addCleanup(_FAKE_TENSOR_ID_TO_PROXY_MAP_FOR_EXPORT.clear)

    def test_map_stays_empty_when_detection_is_off(self):
        self.assertFalse(
            torch._export.config.detect_non_strict_fake_tensor_leaks,
            "this test assumes the default is off",
        )
        _export_small_module()
        self.assertEqual(
            len(_FAKE_TENSOR_ID_TO_PROXY_MAP_FOR_EXPORT),
            0,
            "the trace populated the leak-detection map even though detection "
            "is off; each entry holds a Node, and through it the graph, its "
            "owning module, and every parameter",
        )

    def test_map_is_populated_when_detection_is_on(self):
        """The debug path must keep working -- that is the risk of gating it."""
        with torch._export.config.patch(detect_non_strict_fake_tensor_leaks=True):
            _export_small_module()
            self.assertGreater(
                len(_FAKE_TENSOR_ID_TO_PROXY_MAP_FOR_EXPORT),
                0,
                "detection is on but the map was not populated, so leak "
                "detection can no longer see anything",
            )


@unittest.skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
class AOTInductorMemoryTest(TestCase):
    """Coarse end-to-end check that a compile does not retain the weights.

    The precise assertion lives in ExportLeakDetectionMapTest; this only
    confirms the effect is visible in allocator accounting.
    """

    def _compile_and_drop(self, depth: int) -> None:
        # Citrine C3: build on the device rather than allocating on CPU and
        # moving, which would briefly hold a second copy of every weight -- in
        # a test whose whole subject is retained weights.
        model = nn.Sequential(
            *[
                nn.Linear(512, 512, device=GPU_TYPE, dtype=torch.float16)
                for _ in range(depth)
            ]
        ).eval()
        example = (torch.randn(64, 512, dtype=torch.float16, device=GPU_TYPE),)
        with torch.no_grad():
            gm = torch.export.export(model, example).module()
            torch._inductor.aot_compile(gm, example, options={"max_autotune": False})
            del gm
        del model, example

    def test_aot_compile_releases_the_weights(self):
        # Warm up: first compile pulls in machinery whose allocations are not
        # what this test is about.
        self._compile_and_drop(depth=1)
        gc.collect()

        baseline = torch.accelerator.memory_allocated()
        self._compile_and_drop(depth=4)
        gc.collect()
        leaked = torch.accelerator.memory_allocated() - baseline

        # 4 x 512x512 fp16 weights is ~2 MB; anything near that means the
        # parameters were retained rather than freed.
        self.assertLess(
            leaked,
            512 * 1024,
            f"aot_compile retained {leaked / 1e6:.1f} MB after the model was dropped",
        )

    def test_repeated_compiles_do_not_accumulate(self):
        """The retention scaled with the number of compiles, so check growth."""
        self._compile_and_drop(depth=1)
        gc.collect()

        baseline = torch.accelerator.memory_allocated()
        for _ in range(3):
            self._compile_and_drop(depth=4)
        gc.collect()

        self.assertLess(
            torch.accelerator.memory_allocated() - baseline,
            512 * 1024,
            "memory grew across repeated aot_compile calls",
        )


if __name__ == "__main__":
    run_tests()
