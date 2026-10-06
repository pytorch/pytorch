# Owner(s): ["module: windows"]

import json
import subprocess
import sys
import textwrap
import unittest

from torch.testing._internal.common_utils import run_tests, TestCase


@unittest.skipUnless(sys.platform == "win32", "Windows DLL loading test")
class TestWindowsDllLoading(TestCase):
    def test_import_skips_test_only_dlls(self):
        script = textwrap.dedent(
            """
            import ctypes
            import json
            from pathlib import Path
            import torch

            dll_dir = Path(torch._C.__file__).parent / "lib"
            kernel32 = ctypes.WinDLL("kernel32.dll")
            kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
            kernel32.GetModuleHandleW.restype = ctypes.c_void_p
            test_dlls = (
                "aoti_custom_ops.dll",
                "torchbind_test.dll",
                "jitbackend_test.dll",
                "backend_with_compiler.dll",
            )
            print(json.dumps({
                "core_present": (dll_dir / "c10.dll").exists(),
                "core_loaded": bool(kernel32.GetModuleHandleW("c10.dll")),
                "test_dlls": {
                    name: bool(kernel32.GetModuleHandleW(name))
                    for name in test_dlls if (dll_dir / name).exists()
                },
            }))
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        loaded = json.loads(result.stdout.splitlines()[-1])
        if not loaded["test_dlls"]:
            self.skipTest("test DLLs are not installed")
        self.assertTrue(loaded["core_present"])
        self.assertTrue(loaded["core_loaded"])
        self.assertFalse(any(loaded["test_dlls"].values()), loaded["test_dlls"])


if __name__ == "__main__":
    run_tests()
