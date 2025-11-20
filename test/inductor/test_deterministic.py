# Owner(s): ["module: inductor"]
import contextlib
import os
import subprocess
import sys
import tempfile
import unittest
import pathlib

import torch
import torch._inductor.config as inductor_config
from torch._dynamo.utils import counters
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_cache
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CUDA_AND_TRITON,
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

    @parametrize("model_name", ["GoogleFnet", "BertForMaskedLM", "DistillGPT2"])
    @parametrize("training_or_inference", ["training", "inference"])
    @parametrize("precision", ["float32", "bfloat16", "float16", "amp"])
    def test_run2run_determinism(self, model_name, training_or_inference, precision):
        """
        Test run2run determinism for a few huggingface models.

        The test assumes benchmarks/dynamo/huggingface.py can be found from
        the current working directory.
        """
        # XXX log to remove
        # step1: fail since benchmark script not found
        # step2: found the benchmark script but fail numeric check <==
        # step3: run the benchmark script and pass

        # if not os.path.exists("benchmarks/dynamo/huggingface.py"): self.skipTest("Skip due to benchmarks/dynamo/huggingface.py not found.")

        def _setup_env(env):
            env["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"  # disable autotune cache
            env["TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE"] = "0"
            env["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "0"
            if enable_determinism:
                env["TORCHINDUCTOR_DETERMINISTIC"] = "1"

        # set to false if you want to check how the test fails without
        # the deterministic mode
        enable_determinism = False
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
                f"stdout: {out.stdout.decode()}, stderr: {out.stderr.decode()}",
            )


if __name__ == "__main__":
    if HAS_CUDA_AND_TRITON:
        run_tests()
