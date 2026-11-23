# Owner(s): ["module: dynamo"]
from unittest import mock

import torch
import torch._dynamo.test_case


class TestCaseTests(torch._dynamo.test_case.TestCase):
    def _run_leaking_pair(self, leak_fn):
        probe = torch._dynamo.test_case.TestCase(methodName="runTest")
        probe.setUp()
        try:
            leak_fn()
        finally:
            probe.tearDown()

        return torch._dynamo.test_case._snapshot_autocast_state()

    def test_teardown_restores_autocast_enabled(self):
        baseline = torch._dynamo.test_case._snapshot_autocast_state()

        def leak():
            for device in torch._C._autocast_supported_devices():
                torch.set_autocast_enabled(
                    device, not torch.is_autocast_enabled(device)
                )

        self.assertEqual(self._run_leaking_pair(leak), baseline)

    def test_teardown_restores_autocast_dtype(self):
        baseline = torch._dynamo.test_case._snapshot_autocast_state()

        def leak():
            for device in torch._C._autocast_supported_devices():
                dtype = torch.get_autocast_dtype(device)
                other = torch.float16 if dtype != torch.float16 else torch.bfloat16
                torch.set_autocast_dtype(device, other)

        self.assertEqual(self._run_leaking_pair(leak), baseline)

    def test_teardown_restores_autocast_cache_enabled(self):
        baseline = torch._dynamo.test_case._snapshot_autocast_state()

        def leak():
            torch.set_autocast_cache_enabled(not torch.is_autocast_cache_enabled())

        self.assertEqual(self._run_leaking_pair(leak), baseline)

    def test_teardown_restores_autocast_nesting_after_increment(self):
        baseline = torch._dynamo.test_case._snapshot_autocast_state()
        baseline_nesting = torch._dynamo.test_case._autocast_nesting()
        self.assertEqual(baseline_nesting, 0)

        def leak():
            torch.autocast_increment_nesting()
            self.assertEqual(
                torch._dynamo.test_case._autocast_nesting(), baseline_nesting + 1
            )

        with mock.patch.object(
            torch, "clear_autocast_cache", wraps=torch.clear_autocast_cache
        ) as clear_cache:
            self.assertEqual(self._run_leaking_pair(leak), baseline)
        clear_cache.assert_called_once_with()

    def test_teardown_restores_autocast_nesting_after_decrement(self):
        baseline = torch._dynamo.test_case._snapshot_autocast_state()
        baseline_nesting = torch._dynamo.test_case._autocast_nesting()

        def leak():
            torch.autocast_decrement_nesting()
            self.assertEqual(
                torch._dynamo.test_case._autocast_nesting(), baseline_nesting - 1
            )

        self.assertEqual(self._run_leaking_pair(leak), baseline)

    def test_teardown_restores_autocast_after_super_teardown_error(self):
        baseline = torch._dynamo.test_case._snapshot_autocast_state()
        default_dtype = torch.get_default_dtype()
        grad_enabled = torch.is_grad_enabled()
        cpu_enabled = torch.is_autocast_enabled("cpu")
        probe = torch._dynamo.test_case.TestCase(methodName="runTest")
        probe._default_dtype_check_enabled = True
        probe.setUp()
        torch.set_autocast_enabled("cpu", not cpu_enabled)
        torch.set_grad_enabled(not grad_enabled)
        try:
            torch.set_default_dtype(torch.float64)
            with self.assertRaisesRegex(AssertionError, "expected default dtype"):
                probe.tearDown()
        finally:
            torch.set_default_dtype(default_dtype)
        self.assertEqual(torch._dynamo.test_case._snapshot_autocast_state(), baseline)
        self.assertEqual(torch.is_grad_enabled(), grad_enabled)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
