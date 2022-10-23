# Owner(s): ["module: dynamo"]
import shutil
import unittest

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.utils import DebugDir, get_debug_dir


class DebugDirTests(torch._dynamo.test_case.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._exit_stack.enter_context(
            unittest.mock.patch.object(
                torch._dynamo.config,
                "debug_dir_root",
                "/tmp/torch._dynamo_debug_dirs/",
            )
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(torch._dynamo.config.debug_dir_root, ignore_errors=True)
        cls._exit_stack.close()

    def setUp(self):
        super().setUp()
        torch._dynamo.utils.debug_dir = DebugDir()

    def tearDown(self):
        torch._dynamo.utils.debug_dir = DebugDir()
        super().tearDown()

    def _setup(self):
        debug_dir = torch._dynamo.utils.debug_dir
        debug_dir.setup()
        self.assertIsNotNone(debug_dir.debug_path)
        self.assertEqual(debug_dir.num_setup_calls, 1)
        return debug_dir

    def test_setup(self):
        self._setup()

    def test_clear(self):
        debug_dir = self._setup()
        debug_dir.clear()
        self.assertIsNone(debug_dir.debug_path)
        self.assertEqual(debug_dir.num_setup_calls, 0)

    def test_multi_setup_single_clear(self):
        debug_dir = self._setup()
        prev = get_debug_dir()

        debug_dir.setup()
        self.assertEqual(prev, get_debug_dir())
        self.assertEqual(debug_dir.num_setup_calls, 2)

        debug_dir.clear()
        self.assertEqual(prev, get_debug_dir())
        self.assertEqual(debug_dir.num_setup_calls, 1)

    def test_multi_setup_multi_clear(self):
        debug_dir = self._setup()
        prev = get_debug_dir()

        debug_dir.setup()
        self.assertEqual(prev, get_debug_dir())
        self.assertEqual(debug_dir.num_setup_calls, 2)

        debug_dir.clear()
        self.assertEqual(prev, get_debug_dir())
        self.assertEqual(debug_dir.num_setup_calls, 1)

        debug_dir.clear()
        self.assertIsNone(debug_dir.debug_path)
        self.assertEqual(debug_dir.num_setup_calls, 0)

    def test_single_setup_single_clear(self):
        debug_dir = self._setup()
        debug_dir.clear()
        self.assertIsNone(debug_dir.debug_path)
        self.assertEqual(debug_dir.num_setup_calls, 0)

    def test_multi_get(self):
        self._setup()
        prev = get_debug_dir()
        next = get_debug_dir()
        self.assertEqual(prev, next)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
