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

        def test_a_schema_upgrader_failures_without_adapter(self):
            """Test that V1 schema calls fail without adapter registration.

            Note: The 'a_' prefix ensures this test runs first alphabetically,
            before any adapters are registered by other tests.
            """
            import schema_adapter_test

            t = torch.zeros(2, 3)
            out = schema_adapter_test.ops.test_schema_upgrader_v1(t)
            out2 = schema_adapter_test.ops.test_schema_upgrader_v2(t)
            out3 = schema_adapter_test.ops.test_schema_upgrader_v3(t)
            out4 = schema_adapter_test.ops.test_schema_upgrader_v4(t)

            # For out and out2 tensor will be filled with garbage values
            # as an out of bounds read happens on the StableIValue stack
            self.assertNotEqual(out, torch.fill_(torch.zeros_like(t), 2))
            self.assertNotEqual(out2, torch.fill_(torch.zeros_like(t), 2))

            # v3 and v4 should behave as expected without adapters
            self.assertEqual(out3, torch.fill_(torch.zeros_like(t), 2))
            self.assertEqual(out4, torch.fill_(torch.zeros_like(t), 3))

        def test_schema_upgrader_with_adapter(self):
            import schema_adapter_test

            # Register the adapters
            schema_adapter_test.ops.register_test_adapters()

            t = torch.zeros(2, 3)
            out = schema_adapter_test.ops.test_schema_upgrader_v1(t)
            out2 = schema_adapter_test.ops.test_schema_upgrader_v2(t)
            out3 = schema_adapter_test.ops.test_schema_upgrader_v3(t)
            out4 = schema_adapter_test.ops.test_schema_upgrader_v4(t)

            self.assertEqual(out, torch.fill_(torch.zeros_like(t), 2))
            self.assertEqual(out2, torch.fill_(torch.zeros_like(t), 2))
            self.assertEqual(out3, torch.fill_(torch.zeros_like(t), 2))
            self.assertEqual(out4, torch.fill_(torch.zeros_like(t), 3))


if __name__ == "__main__":
    run_tests()
