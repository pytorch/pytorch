from __future__ import absolute_import, division, print_function, unicode_literals
from caffe2.proto import caffe2_pb2
import os
import sys
import platform
# TODO: refactor & remove the following alias
caffe2_pb2.CPU = caffe2_pb2.PROTO_CPU
caffe2_pb2.CUDA = caffe2_pb2.PROTO_CUDA
caffe2_pb2.MKLDNN = caffe2_pb2.PROTO_MKLDNN
caffe2_pb2.OPENGL = caffe2_pb2.PROTO_OPENGL
caffe2_pb2.OPENCL = caffe2_pb2.PROTO_OPENCL
caffe2_pb2.IDEEP = caffe2_pb2.PROTO_IDEEP
caffe2_pb2.HIP = caffe2_pb2.PROTO_HIP
caffe2_pb2.COMPILE_TIME_MAX_DEVICE_TYPES = caffe2_pb2.PROTO_COMPILE_TIME_MAX_DEVICE_TYPES
caffe2_pb2.ONLY_FOR_TEST = caffe2_pb2.PROTO_ONLY_FOR_TEST

if platform.system() == 'Windows':
    IS_CONDA = 'conda' in sys.version or 'Continuum' in sys.version or any([x.startswith('CONDA') for x in os.environ])

    if IS_CONDA:
        from ctypes import windll, c_wchar_p
        from ctypes.wintypes import DWORD, HMODULE

        AddDllDirectory = windll.kernel32.AddDllDirectory
        AddDllDirectory.restype = DWORD
        AddDllDirectory.argtypes = [c_wchar_p]

    def add_extra_dll_dir(extra_dll_dir):
        if os.path.isdir(extra_dll_dir):
            os.environ['PATH'] = extra_dll_dir + os.pathsep + os.environ['PATH']

            if IS_CONDA:
                AddDllDirectory(extra_dll_dir)

    # first get nvToolsExt PATH
    def get_nvToolsExt_path():
        NVTOOLEXT_HOME = os.getenv('NVTOOLSEXT_PATH', 'C:\\Program Files\\NVIDIA Corporation\\NvToolsExt')

        if os.path.exists(NVTOOLEXT_HOME):
            return os.path.join(NVTOOLEXT_HOME, 'bin', 'x64')
        else:
            return ''

    py_dll_path = os.path.join(os.path.dirname(sys.executable), 'Library', 'bin')
    th_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'torch')
    th_dll_path = os.path.join(th_root, 'lib')

    dll_paths = [th_dll_path, py_dll_path, get_nvToolsExt_path()]

    # then add the path to env
    for p in dll_paths:
        add_extra_dll_dir(p)
