# Owner(s): ["oncall: mobile"]

import torch
from test.jit.fixtures_srcs.fixtures_src import generate_models
from unittest import TestCase


if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

class TestTyping(TestCase):
    def test_all_modules(self):
        for a_module, expect_operator in generate_models.ALL_MODULES.items():
            self.assertEqual(isinstance(a_module, torch.nn.Module))
