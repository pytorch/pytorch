import contextlib

from torch._dynamo.test_case import (
    run_tests as dynamo_run_tests,
    TestCase as DynamoTestCase,
)

from torch._inductor import config


def run_tests(needs=()):
    dynamo_run_tests(needs)


class TestCase(DynamoTestCase):
    """
    A base TestCase for inductor tests. Enables FX graph caching and isolates
    the cache directory for each test.
    """

    _inductor_stack: contextlib.ExitStack

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._inductor_stack = contextlib.ExitStack()
        cls._inductor_stack.enter_context(config.patch({"fx_graph_cache": True}))

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        cls._inductor_stack.close()
