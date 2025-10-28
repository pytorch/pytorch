# Owner(s): ["module: cpp"]

from pathlib import Path

import torch
from torch.testing._internal.common_utils import (
    install_cpp_extension,
    IS_WINDOWS,
    run_tests,
    TestCase,
)

import os


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
        
        def test_op_with_dummy_scale3(self):
            """Test that test_op_with_dummy_scale3 runs and returns a tensor with scale=3."""
            import ops_test

            # Create a test tensor
            input_tensor = torch.ones(1, 3)
            expected_result = torch.empty_like(input_tensor).fill_(
                42 * 3
            )  # scale=3 should multiply by 3
            if os.environ.get("TARGET", '0') == 'V1' and os.environ.get("RUN", "0") == "V1":
                with self.assertRaisesRegex(RuntimeError, "scale argument not supported in version <= 2.8.0"):
                    result = ops_test.ops.test_op_with_dummy_scale3(input_tensor)
            else:
                result = ops_test.ops.test_op_with_dummy_scale3(input_tensor)
                self.assertEqual(result, expected_result)



if __name__ == "__main__":
    run_tests()
