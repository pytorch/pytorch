import torch._dynamo.test_case
import unittest.mock
import os
import contextlib

import logging

def log_settings(settings):
    return unittest.mock.patch.dict(os.environ, {"TORCH_LOGS": settings})

def make_test(settings, log_names):
    def wrapper(fn):
        def test_fn(self):
            records = []
            with log_settings(settings):
                torch._logging._init_logs()
                with self._handler_watcher([logging.getLogger(n) for n in log_names], records):
                    fn(self, records)

        return test_fn

    return wrapper


class LoggingTestCase(torch._dynamo.test_case.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._exit_stack.enter_context(
            unittest.mock.patch.dict(os.environ, {"___LOG_TESTING": ""})
        )
        cls._exit_stack.enter_context(
            unittest.mock.patch("torch._dynamo.config.suppress_errors", True)
        )

    @classmethod
    def tearDownClass(cls):
        cls._exit_stack.close()
        torch._logging.init_logs()

    def _handler_watcher(self, loggers, record_list, install_hook=None):
        exit_stack = contextlib.ExitStack()

        def patch_emit():
            nonlocal exit_stack

            def emit_post_hook(record):
                nonlocal record_list
                record_list.append(record)

            for logger in loggers:
                num_handlers = len(logger.handlers)
                self.assertLessEqual(
                    len(logger.handlers),
                    2,
                    "All pt2 loggers should only have at most two handlers (debug artifacts and messages above debug level).",
                )

                if num_handlers == 0:
                    continue

                for handler in logger.handlers:
                    old_emit = handler.emit

                    def new_emit(record):
                        old_emit(record)
                        emit_post_hook(record)

                    exit_stack.enter_context(
                        unittest.mock.patch.object(handler, "emit", new_emit)
                    )

        # install_hook allows you to call patch_emit
        # after the logs have been setup if needed
        if install_hook:
            exit_stack.enter_context(install_hook(patch_emit))
        else:
            patch_emit()

        return exit_stack
