# Owner(s): ["module: fx"]

#
# Tests compiling the inductor tests in a subprocess.
#

import contextlib
import importlib
import os
import sys
import time
from unittest.mock import patch

import torch
import torch.library
from torch._inductor.compile_fx import _InProcessFxCompile, FxCompile, FxCompileMode
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import TEST_WITH_ASAN
from torch.testing._internal.inductor_utils import GPU_TYPE, RUN_CPU, RUN_GPU


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
import inductor.test_torchinductor  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
from inductor.test_torchinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
    check_model,
    check_model_gpu,
    copy_tests,
    TestFailure,
)


importlib.import_module("filelock")

# xfail by default, set is_skip=True to skip
test_failures = {
    # TypeError: cannot pickle 'generator' object
    "test_layer_norm": TestFailure(("cpu", "cuda"), is_skip=True),
    "test_remove_noop_slice": TestFailure(("xpu"), is_skip=True),
    "test_remove_noop_slice1": TestFailure(("xpu"), is_skip=True),
    "test_remove_noop_slice_scatter": TestFailure(("xpu"), is_skip=True),
    "test_remove_noop_view_default": TestFailure(("xpu"), is_skip=True),
    "test_remove_noop_view_dtype": TestFailure(("xpu"), is_skip=True),
}


class TestSubprocess(TestCase):
    def setUp(self):
        torch._dynamo.reset()
        FxCompile._reset_stats()

        TestCase.setUp(self)

        self._stack = contextlib.ExitStack()
        self._stack.enter_context(
            patch(
                "torch._inductor.compile_fx.fx_compile_mode",
                FxCompileMode.SUBPROCESS,
            )
        )

    def tearDown(self):
        # Check that the test didn't instigate an in-process compile - which
        # would mean that something about the fx graph failed to serialize. If
        # some tests are expected to fail then we should probably add a list of
        # expected failures here.
        self.assertEqual(
            FxCompile._compile_stats[type(_InProcessFxCompile)].codegen_and_compile, 0
        )
        self._stack.close()
        TestCase.tearDown(self)
        torch._dynamo.reset()

    @patch("torch._inductor.compile_fx.fx_compile_async", True)
    def test_async(self):
        # Test that async+subprocess works.
        from torch._inductor.compile_fx_async import _AsyncFxCompile

        @torch.compile(fullgraph=True, backend="inductor")
        def model_add(x, y):
            out = x
            for i in range(500):
                out = torch.add(out, y)
            return out

        _AsyncFxCompile._reset_stats()

        with contextlib.ExitStack() as stack:
            # TODO: Turn off local caches - they don't play nice w/ async currently.
            stack.enter_context(
                torch._inductor.config.patch(
                    autotune_local_cache=False, fx_graph_cache=False
                )
            )
            stack.enter_context(
                torch._functorch.config.patch(enable_autograd_cache=False)
            )

            # How long to wait (in seconds) before giving up.
            TIMEOUT = 300
            # If non-None then how often (in seconds) to print a TICK message.
            TICK_REPORT = None

            start = time.time()
            last_report = start
            while _AsyncFxCompile._stat_compiled_runs < 4:
                # Sleep a bit so we don't drive the CPU unnecessarily.
                time.sleep(0.25)

                x = torch.randn(100, 100)
                y = torch.randn(100, 100)
                model_add(x, y)

                # DEBUGGING: Print a periodic message so we know we're still
                # running...
                now = time.time()
                if TICK_REPORT is not None and (now - last_report > TICK_REPORT):
                    print(f"*** TICK {int(now - start)}")
                    last_report = now

                if now - start > TIMEOUT:
                    raise RuntimeError(
                        "Test timed out before producing a compiled artifact."
                    )

            self.assertEqual(_AsyncFxCompile._stat_compiled_runs, 4)
            # Make sure we ran eager at least once. Normally this will be
            # something like 80.
            self.assertGreater(_AsyncFxCompile._stat_eager_runs, 0)
            self.assertEqual(_AsyncFxCompile._stat_bg_started, 1)
            self.assertEqual(_AsyncFxCompile._stat_bg_finished, 1)


if RUN_CPU:

    class CpuTests(TestSubprocess):
        common = check_model
        device = "cpu"

    copy_tests(
        inductor.test_torchinductor.CommonTemplate, CpuTests, "cpu", test_failures
    )

if RUN_GPU and not TEST_WITH_ASAN:

    class GPUTests(TestSubprocess):
        common = check_model_gpu
        device = GPU_TYPE

    copy_tests(
        inductor.test_torchinductor.CommonTemplate, GPUTests, GPU_TYPE, test_failures
    )


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if RUN_CPU or RUN_GPU:
        run_tests(needs="filelock")
