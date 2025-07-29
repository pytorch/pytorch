import ctypes
import glob
import os
import platform
import sys
import textwrap
import sysconfig

import torch

if sys.platform == "win32":

    def _load_dll_libraries() -> None:

        py_dll_path = os.path.join(sys.exec_prefix, "Library", "bin")
        th_dll_path = os.path.join(os.path.dirname(__file__), "lib")
        usebase_path = os.path.join(sysconfig.get_config_var("userbase"), "Library", "bin")
        py_root_bin_path = os.path.join(sys.exec_prefix, "bin")

        # When users create a virtualenv that inherits the base environment,
        # we will need to add the corresponding library directory into
        # DLL search directories. Otherwise, it will rely on `PATH` which
        # is dependent on user settings.
        if sys.exec_prefix != sys.base_exec_prefix:
            base_py_dll_path = os.path.join(sys.base_exec_prefix, "Library", "bin")
        else:
            base_py_dll_path = ""

        dll_paths = [
            p
            for p in (
                th_dll_path,
                py_dll_path,
                base_py_dll_path,
                usebase_path,
                py_root_bin_path,
            )
            if os.path.exists(p)
        ]


        kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
        with_load_library_flags = hasattr(kernel32, "AddDllDirectory")
        prev_error_mode = kernel32.SetErrorMode(0x0001)

        kernel32.LoadLibraryW.restype = ctypes.c_void_p
        if with_load_library_flags:
            kernel32.LoadLibraryExW.restype = ctypes.c_void_p

        for dll_path in dll_paths:
            os.add_dll_directory(dll_path)

        try:
            ctypes.CDLL("vcruntime140.dll")
            ctypes.CDLL("msvcp140.dll")
            if platform.machine() != "ARM64":
                ctypes.CDLL("vcruntime140_1.dll")
        except OSError:
            print(
                textwrap.dedent(
                    """
                    Microsoft Visual C++ Redistributable is not installed, this may lead to the DLL load failure.
                    It can be downloaded at https://aka.ms/vs/16/release/vc_redist.x64.exe
                    """
                ).strip()
            )

        dlls = glob.glob(os.path.join(th_dll_path, "*.dll"))
        path_patched = False
        for dll in dlls:
            is_loaded = False
            if with_load_library_flags:
                res = kernel32.LoadLibraryExW(dll, None, 0x00001100)
                last_error = ctypes.get_last_error()
                if res is None and last_error != 126:
                    err = ctypes.WinError(last_error)
                    err.strerror += (
                        f' Error loading "{dll}" or one of its dependencies.'
                    )
                    raise err
                elif res is not None:
                    is_loaded = True
            if not is_loaded:
                if not path_patched:
                    os.environ["PATH"] = ";".join(dll_paths + [os.environ["PATH"]])
                    path_patched = True
                res = kernel32.LoadLibraryW(dll)
                if res is None:
                    err = ctypes.WinError(ctypes.get_last_error())
                    err.strerror += (
                        f' Error loading "{dll}" or one of its dependencies.'
                    )
                    raise err

        kernel32.SetErrorMode(prev_error_mode)

    _load_dll_libraries()
    del _load_dll_libraries

import torch_openreg._C  # type: ignore[misc]
import torch_openreg.openreg


torch.utils.rename_privateuse1_backend("openreg")
torch._register_device_module("openreg", torch_openreg.openreg)
torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)
