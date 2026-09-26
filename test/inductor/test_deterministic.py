# Owner(s): ["module: inductor"]
import contextlib
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

import torch
import torch._inductor.config as inductor_config
from torch._dynamo.utils import counters
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_cache
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    IS_FBCODE,
    parametrize,
    skipIfRocm,
)
from torch.testing._internal.inductor_utils import HAS_TRITON, IS_BIG_GPU


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


@instantiate_parametrized_tests
class DeterministicTestGeneric(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def setUp(self) -> None:
        super().setUp()
        self._exit_stack = contextlib.ExitStack()
        self._exit_stack.enter_context(fresh_cache())

    def tearDown(self) -> None:
        self._exit_stack.close()
        super().tearDown()

    def test_use_deterministic_algorithsm(self):
        old_val = torch.are_deterministic_algorithms_enabled()
        try:
            for new_val in [True, False, True]:
                torch.use_deterministic_algorithms(new_val, warn_only=True)
                self.assertEqual(inductor_config.deterministic, new_val)
        finally:
            torch.use_deterministic_algorithms(old_val, warn_only=True)

    def test_reorder_for_locality_preserves_randint_order(self):
        with inductor_config.patch(fallback_random=True):

            def fn():
                torch.manual_seed(0)
                out = torch.randint(0, 100, (4, 1), dtype=torch.int64)
                _ = torch.randint(0, 100, (2, 1), dtype=torch.int64)
                return out

            compiled = torch.compile(fn, backend="inductor")

            torch.manual_seed(0)
            eager = fn()

            torch.manual_seed(0)
            compiled_out = compiled()

            torch.testing.assert_close(eager, compiled_out)

    @skipIfRocm(msg="https://github.com/pytorch/pytorch/issues/180681")
    @unittest.skipIf(IS_FBCODE, "Skipping run2run determinism test in fbcode")
    @parametrize("model_name", ["GoogleFnet", "BertForMaskedLM", "DistillGPT2"])
    @parametrize("training_or_inference", ["training", "inference"])
    @parametrize("precision", ["float32", "bfloat16", "float16", "amp"])
    def test_run2run_determinism(self, model_name, training_or_inference, precision):
        """
        Test run2run determinism for a few huggingface models.

        The test assumes benchmarks/dynamo/huggingface.py can be found from
        the current working directory.
        """

        def _setup_env(env):
            env["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"  # disable autotune cache
            env["TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE"] = "0"
            env["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "0"
            if enable_determinism:
                env["TORCHINDUCTOR_DETERMINISTIC"] = "1"

        # set to false if you want to check how the test fails without
        # the deterministic mode
        enable_determinism = True
        with tempfile.TemporaryDirectory() as tmpdir:
            saved_pkl = os.path.join(tmpdir, "saved.pkl")
            cmd = (
                f"{sys.executable} {REPO_ROOT}/benchmarks/dynamo/huggingface.py --backend inductor"
                + f" --{precision} --accuracy --only {model_name} --{training_or_inference}"
                + f" --disable-cudagraphs --save-model-outputs-to={saved_pkl}"
            )
            print("Command", cmd)
            env = os.environ.copy()
            _setup_env(env)
            out = subprocess.run(cmd.split(), capture_output=True, env=env)

            # We don't check the accuracy against eager here because some
            # of the combination between model and precision can not
            # pass that accuracy test. But it's still valuable to make
            # sure we generate bitwise equivalent result from run to run.
            # self.assertTrue("pass" in out.stdout.decode())

            cmd = (
                f"{sys.executable} {REPO_ROOT}/benchmarks/dynamo/huggingface.py --backend inductor"
                + f" --{precision} --accuracy --only {model_name} --{training_or_inference}"
                + f" --disable-cudagraphs --compare-model-outputs-with={saved_pkl}"
            )
            print("Command", cmd)

            # distort benchmarking results
            env["TORCHINDUCTOR_DISTORT_BENCHMARKING_RESULT"] = "inverse"
            out = subprocess.run(cmd.split(), capture_output=True, env=env)
            self.assertTrue(
                "The result is bitwise equivalent to the previously saved result"
                in out.stdout.decode(),
                lambda msg: (
                    f"{msg}\nstdout: {out.stdout.decode()}, stderr: {out.stderr.decode()}"
                ),
            )


class DeterministicTestAccelerator(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def setUp(self) -> None:
        super().setUp()
        self._exit_stack = contextlib.ExitStack()
        self._exit_stack.enter_context(fresh_cache())

    def tearDown(self) -> None:
        self._exit_stack.close()
        super().tearDown()

    @parametrize("deterministic", [False, True])
    @unittest.skipIf(not HAS_TRITON, "requires triton")
    def test_mm_padding(self, device, deterministic):
        with inductor_config.patch(deterministic=deterministic):

            @torch.compile()
            def foo(x, y):
                return x @ y

            inps = [torch.rand([2049, 2049], device=device) for _ in range(2)]
            out = foo(*inps)
            self.assertEqual(out, inps[0] @ inps[1])

            if deterministic:
                self.assertTrue(counters["inductor"]["pad_mm_bench"] == 0)
            else:
                self.assertTrue(counters["inductor"]["pad_mm_bench"] > 0)

    @parametrize("deterministic", [False, True])
    @inductor_config.patch(max_autotune=True)
    @unittest.skipIf(not HAS_TRITON, "requires triton")
    @unittest.skipIf(not IS_BIG_GPU, "templates require big gpu")
    def test_max_autotune(self, device, deterministic):
        with inductor_config.patch(deterministic=deterministic):

            @torch.compile()
            def foo(x, y):
                return x @ y

            inps = [torch.rand([2048, 2048], device=device) for _ in range(2)]
            out = foo(*inps)
            self.assertEqual(out, inps[0] @ inps[1])

            if deterministic:
                self.assertTrue(counters["inductor"]["select_algorithm_autotune"] == 0)
            else:
                self.assertTrue(counters["inductor"]["select_algorithm_autotune"] > 0)

    @unittest.skipIf(not HAS_TRITON, "requires triton")
    def test_pointwise_coordesc_tuning(self, device):
        @torch.compile(mode="max-autotune")
        def f(x):
            return x + 1

        x = torch.randn(2048, device=device)
        self.assertEqual(f(x), x + 1)

        self.assertTrue(counters["inductor"]["coordesc_tuning_bench"] > 0)

    @parametrize("deterministic", [False, True])
    @unittest.skipIf(not HAS_TRITON, "requires triton")
    def test_reduction_coordesc_tuning(self, device, deterministic):
        with inductor_config.patch(
            deterministic=deterministic, coordinate_descent_tuning=True
        ):

            @torch.compile()
            def foo(x):
                return x.sum(dim=-1)

            inp = torch.rand([2048, 2048], device=device)

            out = foo(inp)
            self.assertEqual(out, inp.sum(dim=-1))

            if deterministic:
                self.assertTrue(counters["inductor"]["coordesc_tuning_bench"] == 0)
            else:
                self.assertTrue(counters["inductor"]["coordesc_tuning_bench"] > 0)

    @unittest.skipIf(not HAS_TRITON, "requires triton")
    @inductor_config.patch(batch_invariant=True)
    def test_persistent_reduction_batch_invariance(self, device):
        H = 768
        FULL = 1024

        def fn(x, w, b):
            return torch.nn.functional.layer_norm(x, (H,), weight=w, bias=b)

        torch.manual_seed(0)
        w = torch.randn(H, device=device, dtype=torch.bfloat16)
        b = torch.randn(H, device=device, dtype=torch.bfloat16)
        x_full = torch.randn(FULL, H, device=device, dtype=torch.bfloat16)

        compiled = torch.compile(fn)
        torch._dynamo.reset()
        out_full = compiled(x_full, w, b)
        self.assertEqual(out_full, fn(x_full, w, b))

        # Halving sweep, matching what the benchmark harness does.
        size = FULL // 2
        while size >= 1:
            torch._dynamo.reset()
            out = compiled(x_full[:size].contiguous(), w, b)
            ref = out_full[:size].contiguous()
            self.assertTrue(
                torch.equal(ref, out),
                lambda msg: (
                    f"{msg}\npersistent reduction diverged at size={size} (FULL={FULL})"
                ),
            )
            size //= 2

    @unittest.skipIf(not HAS_TRITON, "requires triton")
    def test_cumsum_deterministic(self, device):
        from torch._inductor.utils import run_and_get_code

        x = torch.randn(1_000_003, device=device, dtype=torch.float32)
        compiled = torch.compile(lambda v: torch.cumsum(v, dim=0))

        torch.use_deterministic_algorithms(True, warn_only=False)
        eager = torch.cumsum(x, dim=0)
        result, (_,) = run_and_get_code(compiled, x)
        for _ in range(5):
            self.assertEqual(result.view(torch.int32), compiled(x).view(torch.int32))
        self.assertEqual(result.view(torch.int32), eager.view(torch.int32))


instantiate_device_type_tests(
    DeterministicTestAccelerator,
    globals(),
    except_for="cpu",
    allow_xpu=True,
)

if __name__ == "__main__":
    from torch.utils._triton import has_triton

    if has_triton():
        run_tests()
