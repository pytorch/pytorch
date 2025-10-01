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

        def test_schema_upgrader_with_adapter(self):
            import schema_adapter_test

            # without adapters UndefinedBehaviorSanitizer in CI will fail

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
