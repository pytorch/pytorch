# Owner(s): ["oncall: distributed"]

from unittest import mock

import torch
from torch.distributed.fsdp import fully_shard
from torch.testing._internal.common_fsdp import compiled_fsdp_test
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


class TestCompiledFSDPTest(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @parametrize("failure", ["eager", "compiled", "barrier"])
    @torch._inductor.config.patch(compile_threads=3)
    def test_restores_state_after_failure(self, failure):
        original = torch.distributed.fsdp.fully_shard
        modes = []

        @compiled_fsdp_test()
        def run():
            compiled = fully_shard is not original
            modes.append(compiled)
            if (compiled and failure == "compiled") or (
                not compiled and failure == "eager"
            ):
                raise RuntimeError("test failed")

        barriers = [None, None, None, RuntimeError("barrier failed")]
        with (
            mock.patch(
                "torch.testing._internal.common_fsdp.has_triton", return_value=True
            ),
            mock.patch(
                "torch.distributed.barrier",
                side_effect=barriers if failure == "barrier" else None,
            ),
            self.assertRaisesRegex(RuntimeError, "failed"),
        ):
            run()

        self.assertIs(fully_shard, original)
        self.assertEqual(torch._inductor.config.compile_threads, 3)
        self.assertEqual(modes, [False] if failure == "eager" else [False, True])

    @torch._inductor.config.patch(compile_threads=3)
    def test_runs_both_modes_and_restores_state(self):
        original = torch.distributed.fsdp.fully_shard
        modes = []

        @compiled_fsdp_test()
        def run():
            compiled = fully_shard is not original
            modes.append(compiled)
            self.assertEqual(
                torch._inductor.config.compile_threads, 1 if compiled else 3
            )

        with (
            mock.patch(
                "torch.testing._internal.common_fsdp.has_triton", return_value=True
            ),
            mock.patch("torch.distributed.barrier") as barrier,
        ):
            run()

        self.assertEqual(modes, [False, True])
        self.assertEqual(barrier.call_count, 4)
        self.assertIs(fully_shard, original)
        self.assertEqual(torch._inductor.config.compile_threads, 3)

    @torch._inductor.config.patch(compile_threads=3)
    def test_skips_compiled_mode_without_triton(self):
        calls = []

        @compiled_fsdp_test()
        def run():
            calls.append(fully_shard)

        with (
            mock.patch(
                "torch.testing._internal.common_fsdp.has_triton", return_value=False
            ),
            mock.patch("torch.distributed.barrier") as barrier,
            self.assertWarnsRegex(UserWarning, "Inductor on GPU needs Triton"),
        ):
            run()

        self.assertEqual(calls, [torch.distributed.fsdp.fully_shard])
        self.assertEqual(barrier.call_count, 2)
        self.assertIs(fully_shard, torch.distributed.fsdp.fully_shard)
        self.assertEqual(torch._inductor.config.compile_threads, 3)


instantiate_parametrized_tests(TestCompiledFSDPTest)

if __name__ == "__main__":
    run_tests()
