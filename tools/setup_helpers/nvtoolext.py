import os
import platform
import ctypes.util
from subprocess import Popen, PIPE

from .cuda import WITH_CUDA

WINDOWS_HOME = 'C:/Program Files/NVIDIA Corporation/NvToolsExt'

if not WITH_CUDA:
    NVTOOLEXT_HOME = None
else:
    # We use nvcc path on Linux and cudart path on macOS
    osname = platform.system()
    if osname != 'Windows':
        NVTOOLEXT_HOME = None
    else:
        NVTOOLEXT_HOME = os.getenv('NVTOOLSEXT_PATH', WINDOWS_HOME).replace('\\', '/')
        if not os.path.exists(NVTOOLEXT_HOME):
            NVTOOLEXT_HOME = ctypes.util.find_library('nvToolsExt64_1')
            if NVTOOLEXT_HOME is not None:
                NVTOOLEXT_HOME = os.path.dirname(NVTOOLEXT_HOME)
            else:
                NVTOOLEXT_HOME = None
        if NVTOOLEXT_HOME is None:
            from _winreg import ConnectRegistry, OpenKey, QueryValueEx

            reg = ConnectRegistry(None, HKEY_LOCAL_MACHINE)

            try:
                # query in the registry for the path of NvToolsExt: %NVTOOLEXT%\lib\x64\nvToolsExt64_1.lib
                lib_path_key = OpenKey(aReg, r"""SOFTWARE\Microsoft\Windows\CurrentVersion\Installer\UserData\
                                       S-1-5-18\Components\2221AD5AF46D01746B10434547D0B4F7""")
                lib_path = QueryValueEx(aKey, "88F2D65B5688DF047BCA0F47EED402D1")[0]
            except WindowsError:
                NVTOOLEXT_HOME = None

            if os.path.exists(lib_path):
                lib_path = os.path.dirname(lib_path).replace('\\', '/')
                NVTOOLEXT_HOME = os.path.dirname(os.path.dirname(lib_path))
            else:
                NVTOOLEXT_HOME = None
