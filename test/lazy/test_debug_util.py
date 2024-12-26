# Owner(s): ["oncall: jit"]

import os
import re
import tempfile
import unittest

import torch._lazy
import torch._lazy.ts_backend
import torch.nn as nn
from torch.testing._internal.common_utils import IS_WINDOWS, run_tests, TestCase


torch._lazy.ts_backend.init()


@unittest.skipIf(IS_WINDOWS, "To be fixed")
class DebugUtilTest(TestCase):
    def _run_linear(self):
        device = "lazy"
        model = nn.Linear(5, 5).to(device)
        output = model(torch.randn(1, 5).to(device))  # noqa: F841
        torch._lazy.mark_step()

    def test_get_python_frames(self):
        # We only care about the first "Python Stacktrace" part of the saved
        # graph. However, we cannot save the whole stack for comparison given
        # it depends on a lot of things.
        partial_graph = (
            r"Python Stacktrace:.*"
            r"mark_step \(.*/_lazy/__init__.py:[0-9]+\).*"
            r"_run_linear \(.*lazy/test_debug_util.py:[0-9]+\).*"
            r"test_get_python_frames \(.*lazy/test_debug_util.py:[0-9]+\)"
        )

        with tempfile.NamedTemporaryFile(mode="r+", encoding="utf-8") as graph_file:
            os.environ["LTC_SAVE_TENSORS_FILE"] = graph_file.name
            self._run_linear()
            file = graph_file.read()
            if re.search(partial_graph, file, re.DOTALL) is None:
                print(file)
                self.assertTrue(False)


if __name__ == "__main__":
    run_tests()
