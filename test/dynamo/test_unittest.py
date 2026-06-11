# Owner(s): ["module: dynamo"]
import os
import subprocess
import sys
import textwrap
import unittest

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import make_dynamo_test


class TestUnittest(torch._dynamo.test_case.TestCase):
    def setUp(self):
        super().setUp()
        self._prev = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True

    def tearDown(self):
        super().tearDown()
        torch._dynamo.config.enable_trace_unittest = self._prev

    @make_dynamo_test
    def test_SkipTest(self):
        z = 0
        SkipTest = unittest.SkipTest
        try:
            raise SkipTest("abcd")
        except Exception:
            z = 1
        self.assertEqual(z, 1)

    def _run_python_with_dynamo(self, code):
        env = os.environ.copy()
        env["PYTORCH_TEST_WITH_DYNAMO"] = "1"
        env["PYTORCH_PRINT_REPRO_ON_FAILURE"] = "0"
        torch_root = os.path.dirname(os.path.dirname(torch.__file__))
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, [torch_root, env.get("PYTHONPATH", "")])
        )
        return subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code)],
            cwd=torch_root,
            env=env,
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
            text=True,
        )

    def test_dynamo_run_tests_reports_expected_failure_xpass(self):
        proc = self._run_python_with_dynamo(
            """
            import sys
            from torch._dynamo.test_case import TestCase, run_tests
            from torch.testing._internal import dynamo_test_failures

            class SyntheticDynamoTestCase(TestCase):
                def test_passes(self):
                    pass

            key = "SyntheticDynamoTestCase.test_passes"
            dynamo_test_failures.dynamo_expected_failures.add(key)
            sys.argv = [sys.argv[0], key]
            run_tests()
            """
        )
        self.assertNotEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("Unexpected success, please remove", proc.stdout)
        self.assertIn("SyntheticDynamoTestCase.test_passes", proc.stdout)

    def test_dynamo_expected_failure_ignores_unittest_xfail_marker(self):
        proc = self._run_python_with_dynamo(
            """
            import sys
            import unittest
            from torch.testing._internal import dynamo_test_failures
            from torch.testing._internal.common_utils import TestCase, run_tests

            class SyntheticUnittestExpectedFailure(TestCase):
                @unittest.expectedFailure
                def test_passes(self):
                    pass

            key = "SyntheticUnittestExpectedFailure.test_passes"
            dynamo_test_failures.dynamo_expected_failures.add(key)
            sys.argv = [sys.argv[0], key]
            run_tests()
            """
        )
        self.assertNotEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("Unexpected success, please remove", proc.stdout)
        self.assertIn("SyntheticUnittestExpectedFailure.test_passes", proc.stdout)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
