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
from torch._inductor.utils import fresh_cache, run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.common_utils import (
    DeterministicGuard,
    instantiate_parametrized_tests,
    IS_FBCODE,
    parametrize,
    skipIfRocm,
)
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_GPU_AND_TRITON,
    IS_BIG_GPU,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


@instantiate_parametrized_tests
class DeterministicTest(TestCase):
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

    def _run_cumsum_and_get_code(
        self,
        shape,
        dim,
        input_dtype=torch.float32,
        output_dtype=None,
    ):
        def fn(x):
            return torch.cumsum(x, dim=dim, dtype=output_dtype)

        if input_dtype.is_floating_point:
            x = torch.randn(shape, device=GPU_TYPE, dtype=input_dtype)
        else:
            x = torch.randint(-8, 9, shape, device=GPU_TYPE, dtype=input_dtype)

        actual, codes = run_and_get_code(torch.compile(fn), x)
        self.assertEqual(actual, fn(x), atol=1e-3, rtol=1e-3)
        self.assertTrue(codes)
        return "\n".join(codes)

    @parametrize("deterministic", [False, True])
    @skipIfRocm
    @unittest.skipIf(GPU_TYPE != "cuda", "requires CUDA")
    def test_cumsum_split_scan_codegen(self, deterministic):
        with DeterministicGuard(deterministic):
            code = self._run_cumsum_and_get_code((1_000_003,), 0)

        if deterministic:
            FileCheck().check("torch.ops.aten.cumsum.default(").run(code)
            FileCheck().check_not("exclusive_scan_decoupled_lookback").run(code)
        else:
            FileCheck().check_not("torch.ops.aten.cumsum.default(").run(code)
            FileCheck().check("exclusive_scan_decoupled_lookback").run(code)

    @parametrize(
        "shape, dim, input_dtype, output_dtype, fallback",
        [
            ((2, 500_003), 1, torch.int32, torch.float32, True),
            ((1_000_003,), 0, torch.float32, torch.int64, False),
        ],
    )
    @skipIfRocm
    @unittest.skipIf(GPU_TYPE != "cuda", "requires CUDA")
    def test_cumsum_split_scan_output_dtype(
        self, shape, dim, input_dtype, output_dtype, fallback
    ):
        with DeterministicGuard(True):
            code = self._run_cumsum_and_get_code(shape, dim, input_dtype, output_dtype)

        if fallback:
            FileCheck().check("torch.ops.aten.cumsum.default(").run(code)
            FileCheck().check_not("exclusive_scan_decoupled_lookback").run(code)
        else:
            FileCheck().check_not("torch.ops.aten.cumsum.default(").run(code)
            FileCheck().check("exclusive_scan_decoupled_lookback").run(code)

    @skipIfRocm
    @unittest.skipIf(GPU_TYPE != "cuda", "requires CUDA")
    def test_cumsum_nonsplit_scan_deterministic(self):
        with DeterministicGuard(True):
            code = self._run_cumsum_and_get_code((32_768, 128), 1)

        FileCheck().check_not("torch.ops.aten.cumsum.default(").run(code)
        FileCheck().check_not("exclusive_scan_decoupled_lookback").run(code)
        FileCheck().check("tl.associative_scan").run(code)

    @parametrize("config_name", ["deterministic", "batch_invariant"])
    @skipIfRocm
    @unittest.skipIf(GPU_TYPE != "cuda", "requires CUDA")
    def test_cumsum_inductor_deterministic_config(self, config_name):
        config_values = {
            "deterministic": config_name == "deterministic",
            "batch_invariant": config_name == "batch_invariant",
        }
        with (
            DeterministicGuard(False),
            inductor_config.patch(**config_values),
        ):
            code = self._run_cumsum_and_get_code((1_000_003,), 0)

        FileCheck().check_not("torch.ops.aten.cumsum.default(").run(code)
        FileCheck().check_not("exclusive_scan_decoupled_lookback").run(code)
        FileCheck().check("tl.associative_scan").run(code)

    @parametrize("deterministic", [False, True])
    def test_mm_padding(self, deterministic):
        with inductor_config.patch(deterministic=deterministic):

            @torch.compile()
            def foo(x, y):
                return x @ y

            inps = [torch.rand([2049, 2049], device=GPU_TYPE) for _ in range(2)]
            out = foo(*inps)
            self.assertEqual(out, inps[0] @ inps[1])

            if deterministic:
                self.assertTrue(counters["inductor"]["pad_mm_bench"] == 0)
            else:
                self.assertTrue(counters["inductor"]["pad_mm_bench"] > 0)

    @parametrize("deterministic", [False, True])
    @inductor_config.patch(max_autotune=True)
    @unittest.skipIf(not IS_BIG_GPU, "templates require big gpu")
    def test_max_autotune(self, deterministic):
        with inductor_config.patch(deterministic=deterministic):

            @torch.compile()
            def foo(x, y):
                return x @ y

            inps = [torch.rand([2048, 2048], device=GPU_TYPE) for _ in range(2)]
            out = foo(*inps)
            self.assertEqual(out, inps[0] @ inps[1])

            if deterministic:
                self.assertTrue(counters["inductor"]["select_algorithm_autotune"] == 0)
            else:
                self.assertTrue(counters["inductor"]["select_algorithm_autotune"] > 0)

    def test_pointwise_coordesc_tuning(self):
        @torch.compile(mode="max-autotune")
        def f(x):
            return x + 1

        x = torch.randn(2048, device=GPU_TYPE)
        self.assertEqual(f(x), x + 1)

        self.assertTrue(counters["inductor"]["coordesc_tuning_bench"] > 0)

    @parametrize("deterministic", [False, True])
    def test_reduction_coordesc_tuning(self, deterministic):
        with inductor_config.patch(
            deterministic=deterministic, coordinate_descent_tuning=True
        ):

            @torch.compile()
            def foo(x):
                return x.sum(dim=-1)

            inp = torch.rand([2048, 2048], device=GPU_TYPE)

            out = foo(inp)
            self.assertEqual(out, inp.sum(dim=-1))

            if deterministic:
                self.assertTrue(counters["inductor"]["coordesc_tuning_bench"] == 0)
            else:
                self.assertTrue(counters["inductor"]["coordesc_tuning_bench"] > 0)

    @unittest.skipIf(not HAS_GPU_AND_TRITON, "requires GPU + Triton")
    @inductor_config.patch(batch_invariant=True)
    def test_persistent_reduction_batch_invariance(self):
        H = 768
        FULL = 1024

        def fn(x, w, b):
            return torch.nn.functional.layer_norm(x, (H,), weight=w, bias=b)

        torch.manual_seed(0)
        w = torch.randn(H, device=GPU_TYPE, dtype=torch.bfloat16)
        b = torch.randn(H, device=GPU_TYPE, dtype=torch.bfloat16)
        x_full = torch.randn(FULL, H, device=GPU_TYPE, dtype=torch.bfloat16)

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
                lambda msg: f"{msg}\npersistent reduction diverged at size={size} (FULL={FULL})",
            )
            size //= 2

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
                lambda msg: f"{msg}\nstdout: {out.stdout.decode()}, stderr: {out.stderr.decode()}",
            )


if __name__ == "__main__":
    if HAS_GPU_AND_TRITON:
        run_tests()
