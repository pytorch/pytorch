# Owner(s): ["module: unknown"]

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

import logging
from itertools import product


class LoggingTest(TestCase):
    def testApiUsage(self):
        """
        This test verifies that api usage logging is not triggered via static
        initialization. Since it's triggered at first invocation only - we just
        subprocess
        """
        s = TestCase.runWithPytorchAPIUsageStderr("import torch")
        self.assertRegex(s, "PYTORCH_API_USAGE.*import")
        # import the shared library directly - it triggers static init but doesn't call anything
        s = TestCase.runWithPytorchAPIUsageStderr(f"from ctypes import CDLL; CDLL('{torch._C.__file__}')")
        self.assertNotRegex(s, "PYTORCH_API_USAGE")

    def test_log_cpp(self):
        log_level_names = [
            "CRITICAL",
            "ERROR",
            "WARNING",
            "INFO",
            "DEBUG",
        ]
        test_cases = product(
            torch._logging._internal.log_registry.log_alias_to_log_qname.items(),
            log_level_names)

        try:
            for (alias, qname), log_level_name in test_cases:
                log_level = getattr(logging, log_level_name)

                # Test that the log is emitted if the internal C++ log level
                # enables it
                torch._C._enable_log(qname, log_level)
                with self.assertLogs(qname, level=logging.DEBUG) as cm:
                    torch._C._log(qname, log_level, 'test message')
                self.assertEqual(cm.output, [f'{log_level_name}:{qname}:test message'])

                # Test that the log is not emitted if the internal C++ log level
                # disables it
                torch._C._enable_log(qname, log_level + 1)
                with self.assertLogs(qname, level=logging.DEBUG) as cm:
                    torch._C._log(qname, log_level, 'test message')

                    # NOTE: self.assertNoLogs only exists in Python >=3.10
                    # Add this dummy log so we can use self.assertLogs and then
                    # check that only this dummy log is emitted, and not the
                    # previous test log
                    torch._C._log(qname, 100, 'dummy message')

                self.assertEqual(cm.output, [f'Level 100:{qname}:dummy message'])

                # Make sure that `torch._logging.set_logs` sets the internal
                # C++ log level properly as well
                torch._logging.set_logs(all=log_level)
                with self.assertLogs(qname, level=logging.DEBUG) as cm:
                    torch._C._log(qname, log_level, 'test message')
                self.assertEqual(cm.output, [f'{log_level_name}:{qname}:test message'])

                if log_level < logging.CRITICAL:
                    # `set_logs` does not allow anything higher than critical
                    torch._logging.set_logs(all=logging.CRITICAL)
                    with self.assertLogs(qname, level=logging.DEBUG) as cm:
                        torch._C._log(qname, log_level, 'test message')

                        # NOTE: self.assertNoLogs only exists in Python >=3.10
                        # Add this dummy log so we can use self.assertLogs and then
                        # check that only this dummy log is emitted, and not the
                        # previous test log
                        torch._C._log(qname, 100, 'dummy message')

                    self.assertEqual(cm.output, [f'Level 100:{qname}:dummy message'])

        finally:
            torch._logging.set_logs()


if __name__ == '__main__':
    run_tests()
