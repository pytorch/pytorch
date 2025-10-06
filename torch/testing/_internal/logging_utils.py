# mypy: ignore-errors

import torch._dynamo.test_case
import unittest.mock
import os
import contextlib
import torch._logging
import torch._logging._internal
from contextlib import AbstractContextManager
from collections.abc import Callable
from torch._dynamo.utils import LazyString
from torch._inductor import config as inductor_config
import logging
import io

@contextlib.contextmanager
def preserve_log_state():
    prev_state = torch._logging._internal._get_log_state()
    torch._logging._internal._set_log_state(torch._logging._internal.LogState())
    try:
        yield
    finally:
        torch._logging._internal._set_log_state(prev_state)
        torch._logging._internal._init_logs()

def log_settings(settings):
    exit_stack = contextlib.ExitStack()
    settings_patch = unittest.mock.patch.dict(os.environ, {"TORCH_LOGS": settings})
    exit_stack.enter_context(preserve_log_state())
    exit_stack.enter_context(settings_patch)
    torch._logging._internal._init_logs()
    return exit_stack

def log_api(**kwargs):
    exit_stack = contextlib.ExitStack()
    exit_stack.enter_context(preserve_log_state())
    torch._logging.set_logs(**kwargs)
    return exit_stack


def kwargs_to_settings(**kwargs):
    INT_TO_VERBOSITY = {10: "+", 20: "", 40: "-"}

    settings = []

    def append_setting(name, level):
        if isinstance(name, str) and isinstance(level, int) and level in INT_TO_VERBOSITY:
            settings.append(INT_TO_VERBOSITY[level] + name)
            return
        else:
            raise ValueError("Invalid value for setting")

    for name, val in kwargs.items():
        if isinstance(val, bool):
            settings.append(name)
        elif isinstance(val, int):
            append_setting(name, val)
        elif isinstance(val, dict) and name == "modules":
            for module_qname, level in val.items():
                append_setting(module_qname, level)
        else:
            raise ValueError("Invalid value for setting")

    return ",".join(settings)


# Note on testing strategy:
# This class does two things:
# 1. Runs two versions of a test:
#    1a. patches the env var log settings to some specific value
#    1b. calls torch._logging.set_logs(..)
# 2. patches the emit method of each setup handler to gather records
# that are emitted to each console stream
# 3. passes a ref to the gathered records to each test case for checking
#
# The goal of this testing in general is to ensure that given some settings env var
# that the logs are setup correctly and capturing the correct records.
def make_logging_test(**kwargs):
    def wrapper(fn):
        @inductor_config.patch({"fx_graph_cache": False})
        def test_fn(self):

            torch._dynamo.reset()
            records = []
            # run with env var
            if len(kwargs) == 0:
                with self._handler_watcher(records):
                    fn(self, records)
            else:
                with log_settings(kwargs_to_settings(**kwargs)), self._handler_watcher(records):
                    fn(self, records)

            # run with API
            torch._dynamo.reset()
            records.clear()
            with log_api(**kwargs), self._handler_watcher(records):
                fn(self, records)


        return test_fn

    return wrapper

def make_settings_test(settings):
    def wrapper(fn):
        def test_fn(self):
            torch._dynamo.reset()
            records = []
            # run with env var
            with log_settings(settings), self._handler_watcher(records):
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
        cls._exit_stack.enter_context(
            unittest.mock.patch("torch._dynamo.config.verbose", False)
        )

    @classmethod
    def tearDownClass(cls):
        cls._exit_stack.close()
        torch._logging._internal.log_state.clear()
        torch._logging._init_logs()

    def hasRecord(self, records, m):
        return any(m in r.getMessage() for r in records)

    def getRecord(self, records, m):
        record = None
        for r in records:
            # NB: not r.msg because it looks like 3.11 changed how they
            # structure log records
            if m in r.getMessage():
                self.assertIsNone(
                    record,
                    msg=LazyString(
                        lambda: f"multiple matching records: {record} and {r} among {records}"
                    ),
                )
                record = r
        if record is None:
            self.fail(f"did not find record with {m} among {records}")
        return record

    # This patches the emit method of each handler to gather records
    # as they are emitted
    def _handler_watcher(self, record_list):
        exit_stack = contextlib.ExitStack()

        def emit_post_hook(record):
            nonlocal record_list
            record_list.append(record)

        # registered logs are the only ones with handlers, so patch those
        for log_qname in torch._logging._internal.log_registry.get_log_qnames():
            logger = logging.getLogger(log_qname)
            num_handlers = len(logger.handlers)
            self.assertLessEqual(
                num_handlers,
                2,
                "All pt2 loggers should only have at most two handlers (debug artifacts and messages above debug level).",
            )

            self.assertGreater(num_handlers, 0, "All pt2 loggers should have more than zero handlers")

            for handler in logger.handlers:
                old_emit = handler.emit

                def new_emit(record):
                    old_emit(record)
                    emit_post_hook(record)

                exit_stack.enter_context(
                    unittest.mock.patch.object(handler, "emit", new_emit)
                )

        return exit_stack


def logs_to_string(module, log_option):
    """Example:
    logs_to_string("torch._inductor.compile_fx", "post_grad_graphs")
    returns the output of TORCH_LOGS="post_grad_graphs" from the
    torch._inductor.compile_fx module.
    """
    log_stream = io.StringIO()
    handler = logging.StreamHandler(stream=log_stream)

    @contextlib.contextmanager
    def tmp_redirect_logs():
        try:
            logger = torch._logging.getArtifactLogger(module, log_option)
            logger.addHandler(handler)
            yield
        finally:
            logger.removeHandler(handler)

    def ctx_manager():
        exit_stack = log_settings(log_option)
        exit_stack.enter_context(tmp_redirect_logs())
        return exit_stack

    return log_stream, ctx_manager


def multiple_logs_to_string(module: str, *log_options: str) -> tuple[list[io.StringIO], Callable[[], AbstractContextManager[None]]]:
    """Example:
    multiple_logs_to_string("torch._inductor.compile_fx", "pre_grad_graphs", "post_grad_graphs")
    returns the output of TORCH_LOGS="pre_graph_graphs, post_grad_graphs" from the
    torch._inductor.compile_fx module.
    """
    log_streams = [io.StringIO() for _ in range(len(log_options))]
    handlers = [logging.StreamHandler(stream=log_stream) for log_stream in log_streams]

    @contextlib.contextmanager
    def tmp_redirect_logs():
        loggers = [torch._logging.getArtifactLogger(module, option) for option in log_options]
        try:
            for logger, handler in zip(loggers, handlers):
                logger.addHandler(handler)
            yield
        finally:
            for logger, handler in zip(loggers, handlers):
                logger.removeHandler(handler)

    def ctx_manager() -> AbstractContextManager[None]:
        exit_stack = log_settings(", ".join(log_options))
        exit_stack.enter_context(tmp_redirect_logs())
        return exit_stack  # type: ignore[return-value]

    return log_streams, ctx_manager
