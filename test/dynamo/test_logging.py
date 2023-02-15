import contextlib
import logging
import os
import unittest.mock

import torch
import torch._dynamo.logging as td_logging
import torch._dynamo.test_case
import torch._dynamo.testing


def check_log_result():
    pass


def example_fn(a, b, c):
    a0 = a.add(c)
    b0 = b.add(a0)
    b.copy_(b0)
    a.copy_(a0)
    return a, b


ARGS = (torch.ones(2, 2), torch.ones(2, 2), torch.ones(2, 2))


def log_settings(settings):
    return unittest.mock.patch.dict(os.environ, {"TORCH_COMPILE_LOGS": settings})


# This is needed because we reinit logging each time dynamo is called
def init_logging_post_hook(hook):
    old_init_logging = td_logging.init_logging

    def new_init_logging(log_level, log_file_name=None):
        old_init_logging(log_level, log_file_name)
        hook()

    return unittest.mock.patch.object(td_logging, "init_logging", new_init_logging)


def make_test(settings):
    def wrapper(fn):
        def test_fn(self):
            records = []
            with log_settings(settings), self._handler_watcher(records):
                fn(self, records)

        return test_fn

    return wrapper


def single_record_test(name, ty):
    @make_test(name)
    def fn(self, records):
        fn_opt = torch._dynamo.optimize("eager")(example_fn)
        fn_opt(*ARGS)
        self.assertEqual(len(records), 1)
        self.assertIsInstance(records[0].msg, ty)

    return fn


class LoggingTests(torch._dynamo.test_case.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._exit_stack.enter_context(
            unittest.mock.patch.dict(os.environ, {"___LOG_TESTING": ""})
        )

    @classmethod
    def tearDownClass(cls):
        cls._exit_stack.close()

    def _handler_watcher(self, record_list):
        exit_stack = contextlib.ExitStack()

        def patch_emit():
            nonlocal exit_stack
            td_logger = logging.getLogger(td_logging.TORCHDYNAMO_LOG_NAME)
            aot_logger = logging.getLogger(td_logging.AOT_AUTOGRAD_LOG_NAME)
            inductor_logger = logging.getLogger(td_logging.TORCHINDUCTOR_LOG_NAME)

            def emit_post_hook(record):
                nonlocal record_list
                record_list.append(record)

            for logger in (td_logger, aot_logger, inductor_logger):
                num_handlers = len(logger.handlers)
                self.assertLessEqual(
                    len(logger.handlers),
                    1,
                    "All pt2 loggers should only have at most one handler (right now at least).",
                )

                if num_handlers == 0:
                    continue

                handler = logger.handlers[0]
                old_emit = handler.emit

                def new_emit(record):
                    old_emit(record)
                    emit_post_hook(record)

                exit_stack.enter_context(
                    unittest.mock.patch.object(handler, "emit", new_emit)
                )

        exit_stack.enter_context(init_logging_post_hook(patch_emit))

        return exit_stack

    def test_dynamo(self):
        pass

    def test_dynamo_info(self):
        pass

    def test_aot(self):
        pass

    def test_inductor(self):
        pass

    def test_inductor_info():
        pass


for name, ty in td_logging.LOGGABLE_OBJ_TO_REC_TYPE.items():
    setattr(LoggingTests, f"test_{name}", single_record_test(name, ty))
