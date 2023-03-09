import contextlib
import logging
import os
import unittest.mock

import torch
import torch._dynamo.logging as td_logging
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._logging


def check_log_result():
    pass


def example_fn(a):
    output = a.mul(torch.ones(1000, 1000))
    output = output.add(torch.ones(1000, 1000))
    output.sum().backward()
    return output


def dynamo_error_fn(a):
    output = a.mul(torch.ones(1000, 1000))
    output = output.add(torch.ones(10, 10))
    return output


def inductor_error_fn(a):
    output = torch.round(a)
    return output


ARGS = (torch.ones(1000, 1000, requires_grad=True),)


def log_settings(settings):
    return unittest.mock.patch.dict(os.environ, {"TORCH_LOGS": settings})


# This is needed because we reinit logging each time dynamo is called
def init_logging_post_hook(hook):
    old_init_logging = td_logging.init_logging

    def new_init_logging(log_file_name=None):
        old_init_logging(log_file_name)
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


def multi_record_test(name, ty, num_records):
    @make_test(name)
    def fn(self, records):
        fn_opt = torch._dynamo.optimize("inductor")(example_fn)
        fn_opt(*ARGS)
        self.assertEqual(len(records), num_records)
        self.assertIsInstance(records[0].msg, ty)

    return fn


def within_range_record_test(name, ty, num_records_lower, num_records_higher):
    @make_test(name)
    def fn(self, records):
        fn_opt = torch._dynamo.optimize("inductor")(example_fn)
        fn_opt(*ARGS)
        self.assertGreaterEqual(len(records), num_records_lower)
        self.assertLessEqual(len(records), num_records_higher)
        self.assertIsInstance(records[0].msg, ty)

    return fn


def single_record_test(name, ty):
    return multi_record_test(name, ty, 1)


class LoggingTests(torch._dynamo.test_case.TestCase):
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

        exit_stack.enter_context(init_logging_post_hook(patch_emit))

        return exit_stack

    test_bytecode = multi_record_test("bytecode", td_logging.ByteCodeLogRec, 2)
    test_output_code = multi_record_test("output_code", td_logging.OutputCodeLogRec, 2)

    test_dynamo_debug = within_range_record_test("+dynamo", str, 30, 50)
    test_dynamo_info = within_range_record_test("dynamo", str, 2, 10)

    @make_test("-dynamo")
    def test_dynamo_error(self, records):
        try:
            fn_opt = torch._dynamo.optimize("inductor")(dynamo_error_fn)
            fn_opt(*ARGS)
        except:
            pass
        self.assertEqual(len(records), 1)
        self.assertIsInstance(records[0].msg, str)

    test_aot = multi_record_test("aot", td_logging.AOTJointGraphLogRec, 3)
    test_inductor_debug = within_range_record_test("+inductor", str, 5, 15)
    test_inductor_info = within_range_record_test("inductor", str, 2, 4)

    @make_test("-inductor")
    def test_inductor_error(self, records):
        import torch._inductor.lowering

        def throw(x):
            assert False

        # inject an error in the lowerings
        for x in list(torch._inductor.lowering.lowerings.keys()):
            if "round" in x.__name__:
                torch._inductor.lowering.lowerings[x] = throw

        try:
            fn_opt = torch._dynamo.optimize("inductor")(inductor_error_fn)
            fn_opt(*ARGS)
        except:
            pass
        self.assertEqual(len(records), 1)
        self.assertIsInstance(records[0].msg, str)


# single record tests
exclusions = {"bytecode", "output_code"}
for name, ty in torch._logging.NAME_TO_RECORD_TYPE.items():
    if name not in exclusions:
        setattr(LoggingTests, f"test_{name}", single_record_test(name, ty))

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
