# Owner(s): ["module: dynamo"]
import logging
import re
import shutil
import unittest

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch.testing._internal.common_utils import skipIfNoDill


class ReplayRecordTests(torch._dynamo.test_case.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._exit_stack.enter_context(
            unittest.mock.patch.object(
                torch._dynamo.config, "replay_record_enabled", True
            )
        )
        torch._logging.set_logs(graph_breaks=True, dynamo=logging.ERROR)
        # These tests require dynamo exceptions to be propagated up to the caller
        cls._exit_stack.enter_context(
            unittest.mock.patch.object(torch._dynamo.config, "suppress_errors", False)
        )
        cls._exit_stack.enter_context(
            unittest.mock.patch.object(
                torch._dynamo.config,
                "debug_dir_root",
                "/tmp/_torchdynamo_debug_/",
            )
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(torch._dynamo.config.debug_dir_root, ignore_errors=True)
        torch._logging.set_logs()
        cls._exit_stack.close()

    def check_replay(self, fn, *args, exp_exc_name=None):
        fn_opt = torch._dynamo.optimize("eager")(fn)
        try:
            fn_opt(*args)
        except Exception as e:
            if exp_exc_name is not None:
                self.assertIn(exp_exc_name, str(e))
            expected_error = str(e)
        else:
            self.fail("opt_fn didn't raise an exception")

        file_name_match = re.search(r"torch._dynamo\.replay\('(.*)'\)", expected_error)

        # Remove replay message from expected error
        expected_error = expected_error.split("\n")
        for i, line in enumerate(expected_error):
            if "torch._dynamo.replay" in line:
                del expected_error[i + 1]  # Empty line
                del expected_error[i]  # Replay message
                break
        expected_error = "\n".join(expected_error)

        self.maxDiff = None
        self.assertTrue(
            file_name_match is not None,
            "No record file name found in generated logs.",
        )
        try:
            torch._dynamo.replay(file_name_match.groups()[0])
        except Exception as e:
            actual_error = str(e)
            if actual_error != expected_error:
                raise e
        else:
            self.fail("Replayed frame didn't raise an exception")

    @skipIfNoDill
    def test_unsuccessful_inline(self):
        def level2():
            a = {10}
            z = a["z"]  # RuntimeError, Illegal to getitem on a set
            return z * torch.ones(1)

        def level1():
            y = torch.ones(1, 1)
            return level2() + y

        def level0():
            x = torch.ones(1, 1)
            return level1() + x

        self.check_replay(level0, exp_exc_name="RuntimeError")

    @skipIfNoDill
    def test_successful_inline(self):
        def test_fn():
            x = torch.ones(2, 2)

            def level1(a):
                return a + torch.ones(2, 2)

            y = level1(x)

            return y + torch.ones(3, 3)  # dimension mismatch

        self.check_replay(test_fn, exp_exc_name="RuntimeError")

    @skipIfNoDill
    def test_nonlocal_fn_call(self):
        def nonlocal_fn(x):
            return x + torch.ones(2, 2)

        def test_fn():
            z = torch.ones(2, 2)
            x = nonlocal_fn(z)
            return x + torch.ones(3, 3)

        self.check_replay(test_fn, exp_exc_name="RuntimeError")

    @skipIfNoDill
    def test_nonlocal_module_fn_call(self):
        # replay when we use a module
        # not defined in the replay env
        try:
            from . import mock_modules
        except ImportError:
            import mock_modules

        def test_fn():
            z = mock_modules.mock_module2.method1([], 2)
            z = torch.ones(2, 2) + z[0]
            return z + torch.zeros(3, 3)

        self.check_replay(test_fn, exp_exc_name="RuntimeError")

    @skipIfNoDill
    def test_nonlocal_module_class(self):
        try:
            from .mock_modules import mock_module2
        except ImportError:
            from mock_modules import mock_module2

        def test_fn():
            z = mock_module2.Class1(1, 2)
            y = z.method2(torch.ones(3, 3))
            return y + torch.zeros(3, 5)

        self.check_replay(test_fn, exp_exc_name="RuntimeError")

    @skipIfNoDill
    def test_local_module(self):
        try:
            from .mock_modules import mock_module3 as _  # noqa: F401

            def test_fn(x):
                from .mock_modules import mock_module3

                z = mock_module3.method1([], torch.ones(5, 1))
                return torch.ones(2, 2) + x + z[0]

        except ImportError:

            def test_fn(x):
                from mock_modules import mock_module3

                z = mock_module3.method1([], torch.ones(5, 1))
                return torch.ones(2, 2) + x + z[0]

        self.check_replay(test_fn, torch.ones(1, 1), exp_exc_name="RuntimeError")

    # Verify that we replay when we have tensor arguments to the frame being replayed
    @skipIfNoDill
    def test_fn_call_args(self):
        def test_fn(x, y):
            return x + y + torch.zeros(2, 2)

        self.check_replay(
            test_fn, torch.ones(3, 3), torch.ones(2, 2), exp_exc_name="RuntimeError"
        )

    # Verify that accessing torch.nn works when frame replaying is enabled
    @skipIfNoDill
    def test_torch_nn(self):
        def fn(x):
            y = torch.nn.functional.pad(x, (10, 10, 10, 10))
            return y + torch.ones(3, 3)  # dimension mismatch

        x = torch.ones(4, 4, 4, 4)
        self.check_replay(fn, x, exp_exc_name="RuntimeError")


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
