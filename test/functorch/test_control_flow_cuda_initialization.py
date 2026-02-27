# Owner(s): ["module: functorch"]
import unittest

import torch
from torch.testing._internal.common_utils import (
    run_tests,
    TEST_CUDA_GRAPH_CONDITIONAL_NODES,
    TestCase,
)


@unittest.skipIf(
    not TEST_CUDA_GRAPH_CONDITIONAL_NODES,
    "CUDA 12.4 or greater is required for CUDA Graphs with conditional nodes",
)
class TestControlFlowInCUDAGraphInitialization(TestCase):
    # Duplicated from test_cuda_primary_ctx.py
    CTX_ALREADY_CREATED_ERR_MSG = (
        "Tests defined in TestControlFlowInCUDAGraphInitialization must be run in a process "
        "where CUDA contexts are never created. Use either run_test.py or add "
        "--subprocess to run each test in a different subprocess."
    )

    def setUp(self):
        # Ensure context has not been created beforehand
        self.assertFalse(
            torch._C._cuda_hasPrimaryContext(0),
            TestControlFlowInCUDAGraphInitialization.CTX_ALREADY_CREATED_ERR_MSG,
        )

    def _check_compile_cudagraphs(self, f, pred, *other_args):
        f_compiled = torch.compile(f, backend="cudagraphs")

        outputs = []

        for p in [pred, torch.logical_not(pred)]:
            for i in range(3):
                torch.compiler.cudagraph_mark_step_begin()
                outputs.append(f_compiled(pred, *other_args).clone())

            # We compute the eager output only after running cudagraphs
            # backend compiled function, in order to make sure that
            # cudagraph trees warms up the conditional part of the code
            # properly.
            eager_output = f(pred, *other_args)

            for output in outputs:
                self.assertEqual(output, eager_output)

    def test_cond_cudnn(self):
        # Tests that cublasCreate() does not break stream capture
        def f(pred, x, filters):
            return torch.cond(
                pred,
                lambda y: torch.sum(y),
                lambda y: torch.sum(torch.nn.functional.conv1d(y, filters)),
                [x],
            )

        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))

        pred = torch.tensor(True, device="cuda")
        x = torch.randn(33, 16, 30, device="cuda")
        filters = torch.randn(20, 16, 5, device="cuda")

        self._check_compile_cudagraphs(f, pred, x, filters)

        self.assertTrue(torch._C._cuda_hasPrimaryContext(0))

    def test_cond_stft(self):
        # Tests that cufft plan creation does not break stream capture
        def f(pred, x):
            return torch.cond(
                pred,
                lambda y: torch.sum(y),
                lambda y: torch.sum(torch.stft(y, 512, return_complex=False)),
                [x],
            )

        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))

        pred = torch.tensor(True, device="cuda")
        x = torch.ones(1024 * 1024, device="cuda")

        self._check_compile_cudagraphs(f, pred, x)

        self.assertTrue(torch._C._cuda_hasPrimaryContext(0))


if __name__ == "__main__":
    run_tests()
