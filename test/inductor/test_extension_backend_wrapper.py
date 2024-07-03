# Owner(s): ["module: inductor"]
import os
import shutil
import sys
import unittest

import torch
import torch._dynamo
import torch.utils.cpp_extension

try:
    from extension_backends.cpp.extension_codegen_backend import (
        ExtensionScheduling,
        ExtensionWrapperCodegen,
    )
except ImportError:
    from .extension_backends.cpp.extension_codegen_backend import (
        ExtensionScheduling,
        ExtensionWrapperCodegen,
    )

from torch._inductor.codegen.common import register_backend_for_device
from torch.testing._internal.common_utils import IS_FBCODE, IS_MACOS

try:
    try:
        from . import test_torchinductor
    except ImportError:
        import test_torchinductor
except unittest.SkipTest:
    if __name__ == "__main__":
        sys.exit(0)
    raise


TestCase = test_torchinductor.TestCase


def remove_build_path():
    if sys.platform == "win32":
        return
    default_build_root = torch.utils.cpp_extension.get_default_build_root()
    if os.path.exists(default_build_root):
        shutil.rmtree(default_build_root, ignore_errors=True)


@unittest.skipIf(IS_FBCODE, "cpp_extension doesn't work in fbcode right now")
class ExtensionBackendDeviceImportTests(TestCase):
    """ Tests the custom_empty_strided_device and device_imports parameters of
        WrapperCodeGen """
    module = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # Build Extension
        remove_build_path()
        source_file_path = os.path.dirname(os.path.abspath(__file__))
        source_file = os.path.join(
            source_file_path, "extension_backends/cpp/extension_device_import.cpp"
        )
        cls.module = torch.utils.cpp_extension.load(
            name="extension_device_import",
            sources=[
                str(source_file),
            ],
            extra_cflags=["-g"],
            verbose=True,
        )
        torch.utils.rename_privateuse1_backend("extension_device_import")
        torch._register_device_module("extension_device_import", cls.module)

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()
        super().tearDownClass()

        remove_build_path()

    def setUp(self):
        torch._dynamo.reset()
        super().setUp()

        # cpp extensions use relative paths. Those paths are relative to
        # this file, so we'll change the working directory temporarily
        self.old_working_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        assert self.module is not None

    def tearDown(self):
        super().tearDown()
        torch._dynamo.reset()

        # return the working directory (see setUp)
        os.chdir(self.old_working_dir)

    def compile_test_fn(cls):
        device = cls.module.get_custom_device()
        return torch.empty(2, 16, device=device)

    def test_without_import(self):
        """Register a new cpu backend to mimic an out-of-tree import which
        does so. Do not add an import."""

        class CustomEmptyWrapperCodeGen(ExtensionWrapperCodegen):
            def __init__(self) -> None:
                pytorch_test_dir = os.path.dirname(os.path.realpath(__file__))
                super().__init__(custom_empty_strided_device="extension_device_import")

        register_backend_for_device(
            "extension_device_import", ExtensionScheduling, CustomEmptyWrapperCodeGen
        )

        compiled_fn = torch.compile(self.compile_test_fn)
        # Check that this fails due to missing import
        with self.assertRaises(NameError):
            compiled_fn()

    def test_with_import(self):
        """Register a new cpu backend to mimic an out-of-tree import which
        does so. Add an import for empty_strided for the extension to test
        importing"""

        class ImportingWrapperCodeGen(ExtensionWrapperCodegen):
            def __init__(self) -> None:

                pytorch_test_dir = os.path.dirname(os.path.realpath(__file__))
                device_imports = [
                    "import sys",
                    'sys.path.append("' + pytorch_test_dir + '")',
                    "from extension_backends.cpp import extension_device_import_wrapper",
                    "empty_strided_extension_device_import = "
                    + "extension_device_import_wrapper."
                    + "empty_strided_extension_device_import",
                ]

                super().__init__(
                    device_imports=device_imports,
                    custom_empty_strided_device="extension_device_import",
                )

        register_backend_for_device(
            "extension_device_import", ExtensionScheduling, ImportingWrapperCodeGen
        )

        compiled_fn = torch.compile(self.compile_test_fn)
        compiled_fn()


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests
    from torch.testing._internal.inductor_utils import HAS_CPU

    # cpp_extension doesn't work in fbcode right now
    if HAS_CPU and not IS_MACOS and not IS_FBCODE:
        run_tests(needs="filelock")
