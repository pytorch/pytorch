

import ctypes
import os

if 'OSS_ONNXIFI_LIB' in os.environ:
    lib = os.environ['OSS_ONNXIFI_LIB']
    print("Loading ONNXIFI lib: ".format(lib))  # type: ignore[str-format]
    ctypes.CDLL(lib, ctypes.RTLD_GLOBAL)
