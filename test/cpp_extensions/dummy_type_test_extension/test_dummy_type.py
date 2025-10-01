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

    class TestDummyType(TestCase):
        @classmethod
        def setUpClass(cls):
            try:
                import dummy_type_test  # noqa: F401
            except Exception:
                install_cpp_extension(extension_root=Path(__file__).parent)

        def test_dummy_conversion_from_libtorch_to_extension(self):
            """
            Creates a Dummy object in libtorch and passes it to an extension.
            Makes sure extension can read `id` field properly.
            """
            import dummy_type_test

            inp = torch.empty((2, 3))
            out = dummy_type_test.ops.test_fn(inp, torch._C._Dummy(42))
            self.assertEqual(out, torch.empty((2, 3)).fill_(42))

        def test_dummy_conversion_to_libtorch_from_extension(self):
            """
            Creates a Dummy object in an extension and passes it to libtorch.
            Makes sure libtorch has `foo` field populated properly.
            """
            import dummy_type_test

            inp = torch.empty((2, 3))
            dummy = dummy_type_test.ops.create_dummy(inp)
            self.assertTrue(dummy.id == 42)
            self.assertFalse(hasattr(dummy, "foo"))


if __name__ == "__main__":
    run_tests()
