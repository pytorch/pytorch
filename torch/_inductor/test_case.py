import contextlib
import tempfile
import unittest

import torch

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

    _inductor_test_stack: contextlib.ExitStack

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._inductor_test_stack = contextlib.ExitStack()
        cls._inductor_test_stack.enter_context(config.patch({"fx_graph_cache": True}))

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        cls._inductor_test_stack.close()

    def setUp(self):
        super().setUp()

        # For all tests, mock the tmp directory populated by the inductor
        # caches, both for test isolation and to avoid filling disk.
        self._inductor_cache_tmp_dir = tempfile.TemporaryDirectory()
        self._inductor_cache_get_tmp_dir_patch = unittest.mock.patch(
            "torch._inductor.codecache.cache_dir"
        )
        mock_get_dir = self._inductor_cache_get_tmp_dir_patch.start()
        mock_get_dir.return_value = self._inductor_cache_tmp_dir.name

        # This function uses a functools.lru_cache to keep a reference
        # to the tmp dir, so clear it.
        torch._inductor.codecache.cpp_prefix_path.cache_clear()

    def tearDown(self):
        super().tearDown()

        # Clean up the FxGraphCache tmp dir.
        self._inductor_cache_get_tmp_dir_patch.stop()
        self._inductor_cache_tmp_dir.cleanup()
