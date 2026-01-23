# Owner(s): ["oncall: jit"]
# ruff: noqa: F841

import os
import sys

import torch


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


class TestLogging(JitTestCase):
    def test_bump_numeric_counter(self):
        class ModuleThatLogs(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                for _ in range(x.size(0)):
                    x += 1.0
                    torch.jit._logging.add_stat_value("foo", 1)

                if bool(x.sum() > 0.0):
                    torch.jit._logging.add_stat_value("positive", 1)
                else:
                    torch.jit._logging.add_stat_value("negative", 1)
                return x

        logger = torch.jit._logging.LockingLogger()
        old_logger = torch.jit._logging.set_logger(logger)
        try:
            mtl = ModuleThatLogs()
            for _ in range(5):
                mtl(torch.rand(3, 4, 5))

            self.assertEqual(logger.get_counter_val("foo"), 15)
            self.assertEqual(logger.get_counter_val("positive"), 5)
        finally:
            torch.jit._logging.set_logger(old_logger)

    def test_trace_numeric_counter(self):
        def foo(x):
            torch.jit._logging.add_stat_value("foo", 1)
            return x + 1.0

        traced = torch.jit.trace(foo, torch.rand(3, 4))
        logger = torch.jit._logging.LockingLogger()
        old_logger = torch.jit._logging.set_logger(logger)
        try:
            traced(torch.rand(3, 4))

            self.assertEqual(logger.get_counter_val("foo"), 1)
        finally:
            torch.jit._logging.set_logger(old_logger)

    def test_time_measurement_counter(self):
        class ModuleThatTimes(torch.jit.ScriptModule):
            def forward(self, x):
                tp_start = torch.jit._logging.time_point()
                for _ in range(30):
                    x += 1.0
                tp_end = torch.jit._logging.time_point()
                torch.jit._logging.add_stat_value("mytimer", tp_end - tp_start)
                return x

        mtm = ModuleThatTimes()
        logger = torch.jit._logging.LockingLogger()
        old_logger = torch.jit._logging.set_logger(logger)
        try:
            mtm(torch.rand(3, 4))
            self.assertGreater(logger.get_counter_val("mytimer"), 0)
        finally:
            torch.jit._logging.set_logger(old_logger)

    def test_time_measurement_counter_script(self):
        class ModuleThatTimes(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                tp_start = torch.jit._logging.time_point()
                for _ in range(30):
                    x += 1.0
                tp_end = torch.jit._logging.time_point()
                torch.jit._logging.add_stat_value("mytimer", tp_end - tp_start)
                return x

        mtm = ModuleThatTimes()
        logger = torch.jit._logging.LockingLogger()
        old_logger = torch.jit._logging.set_logger(logger)
        try:
            mtm(torch.rand(3, 4))
            self.assertGreater(logger.get_counter_val("mytimer"), 0)
        finally:
            torch.jit._logging.set_logger(old_logger)

    def test_counter_aggregation(self):
        def foo(x):
            for _ in range(3):
                torch.jit._logging.add_stat_value("foo", 1)
            return x + 1.0

        traced = torch.jit.trace(foo, torch.rand(3, 4))
        logger = torch.jit._logging.LockingLogger()
        logger.set_aggregation_type("foo", torch.jit._logging.AggregationType.AVG)
        old_logger = torch.jit._logging.set_logger(logger)
        try:
            traced(torch.rand(3, 4))

            self.assertEqual(logger.get_counter_val("foo"), 1)
        finally:
            torch.jit._logging.set_logger(old_logger)

    def test_logging_levels_set(self):
        torch._C._jit_set_logging_option("foo")
        self.assertEqual("foo", torch._C._jit_get_logging_option())


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
