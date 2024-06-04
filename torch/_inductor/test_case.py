# mypy: allow-untyped-defs
import contextlib
import os

from torch._dynamo.test_case import (
    run_tests as dynamo_run_tests,
    TestCase as DynamoTestCase,
)

from torch._inductor import config
from torch._inductor.utils import fresh_inductor_cache


def run_tests(needs=()):
    dynamo_run_tests(needs)


class TestCase(DynamoTestCase):
    """
    A base TestCase for inductor tests. Enables FX graph caching and isolates
    the cache directory for each test.
    """

    def setUp(self):
        super().setUp()
        self._inductor_test_stack = contextlib.ExitStack()
        self._inductor_test_stack.enter_context(config.patch({"fx_graph_cache": True}))
        if (
            os.environ.get("INDUCTOR_TEST_DISABLE_FRESH_CACHE") != "1"
            and os.environ.get("TORCH_COMPILE_DEBUG") != "1"
        ):
            self._inductor_test_stack.enter_context(fresh_inductor_cache())

    def tearDown(self):
        super().tearDown()
        self._inductor_test_stack.close()
