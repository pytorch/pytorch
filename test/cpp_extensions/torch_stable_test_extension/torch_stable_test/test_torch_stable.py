# Owner(s): ["module: cpp"]

from pathlib import Path

from torch.testing._internal.common_utils import (
    install_cpp_extension,
    IS_WINDOWS,
    run_tests,
    TestCase,
)


if not IS_WINDOWS:

    class TestTorchStable(TestCase):
        def test_setup_fails(self):
            with self.assertRaisesRegex(RuntimeError, "build failed for cpp extension"):
                install_cpp_extension(extension_root=Path(__file__).parent.parent)


if __name__ == "__main__":
    run_tests()
