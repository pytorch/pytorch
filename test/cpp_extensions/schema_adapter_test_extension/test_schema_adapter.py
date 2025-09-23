#!/usr/bin/env python3

# Owner(s): ["module: cpp"]

from pathlib import Path

import torch
from torch.testing._internal.common_utils import (
    install_cpp_extension,
    IS_WINDOWS,
    run_tests,
    TestCase,
)


# TODO: Fix this error in Windows:
# LINK : error LNK2001: unresolved external symbol PyInit__C
if not IS_WINDOWS:

    class TestSchemaAdapter(TestCase):
        @classmethod
        def setUpClass(cls):
            try:
                import schema_adapter_test  # noqa: F401
            except Exception:
                install_cpp_extension(extension_root=Path(__file__).parent)

        def test_v2_op(self):
            import schema_adapter_test

            t = torch.zeros(2, 3)
            out = schema_adapter_test.ops.test_dummy_op_v2(t)
            self.assertEqual(out, torch.fill_(torch.zeros_like(t), 2))
            out = schema_adapter_test.ops.test_dummy_op_v2(t, 1)
            self.assertEqual(out, torch.ones_like(t))

        def test_v1_op_fails_without_adapter(self):
            import schema_adapter_test

            t = torch.zeros(2, 3)
            out = schema_adapter_test.ops.test_dummy_op_v1(t)
            # tensor will be filled with garbage values as an out of bounds read happens
            # on the StableIValue stack
            self.assertNotEqual(out, torch.fill_(torch.zeros_like(t), 2))

        def test_v1_op_with_adapter(self):
            import schema_adapter_test

            schema_adapter_test.ops.register_adapter()

            t = torch.zeros(2, 3)
            out = schema_adapter_test.ops.test_dummy_op_v1(t)
            self.assertEqual(out, torch.fill_(torch.zeros_like(t), 2))


if __name__ == "__main__":
    run_tests()
