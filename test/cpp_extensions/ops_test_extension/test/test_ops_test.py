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

    class TestOpsExtension(TestCase):
        @classmethod
        def setUpClass(cls):
            try:
                import ops_test  # noqa: F401
            except Exception:
                install_cpp_extension(extension_root=Path(__file__).parent.parent)

        def test_op_with_dummy(self):
            """Test that test_op_with_dummy runs and returns a tensor."""
            import ops_test

            # Create a test tensor
            input_tensor = torch.ones(1, 3)
            expected_result = torch.empty_like(input_tensor).fill_(42)
            result = ops_test.ops.test_op_with_dummy(input_tensor)
            self.assertEqual(result, expected_result)


if __name__ == "__main__":
    run_tests()
