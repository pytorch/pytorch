import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._dynamo.logging as td_logging
import logging
import unittest.mock
import contextlib
import os

def check_log_result():
    pass

def fn(a, b, c):
    a0 = a.add(c)
    b0 = b.add(a0)
    b.copy_(b0)
    a.copy_(a0)
    return a,b

ARGS = (torch.ones(2, 2), torch.ones(2, 2), torch.ones(2, 2))

def log_settings(settings):
    return unittest.mock.patch.dict(os.environ, {"TORCH_COMPILE_LOGS": settings})


def init_logging_post_hook(hook):
    old_init_logging = td_logging.init_logging

    def new_init_logging(log_level, log_file_name=None):
        old_init_logging(log_level, log_file_name)
        hook()

    return unittest.mock.patch.object(td_logging, "init_logging", new_init_logging)


class LoggingTests(torch._dynamo.test_case.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._exit_stack.enter_context(
            unittest.mock.patch.dict(
                os.environ, {"___LOG_TESTING": ""}
            )
        )

    @classmethod
    def tearDownClass(cls):
        cls._exit_stack.close()

    def test_guards(self):

        with log_settings("guards"), init_logging_post_hook(lambda: print("hi")):
            fn_opt = torch._dynamo.optimize("eager")(fn)
            fn_opt(*ARGS)

            print("hi")


    def test_bytecode(self):
        pass

    def test_graph(self):
        pass

    def test_graph_code(self):
        pass

    def test_aot_forward(self):
        pass

    def test_aot_backward(self):
        pass

    def test_aot_joint(self):
        pass

    def test_output_code(self):
        pass

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
