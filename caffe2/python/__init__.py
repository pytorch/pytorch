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
    is_conda = os.path.exists(os.path.join(sys.prefix, 'conda-meta'))
    py_dll_path = os.path.join(os.path.dirname(sys.executable), 'Library', 'bin')
    th_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'torch')
    th_dll_path = os.path.join(th_root, 'lib')

    if not os.path.exists(os.path.join(th_dll_path, 'nvToolsExt64_1.dll')) and \
            not os.path.exists(os.path.join(py_dll_path, 'nvToolsExt64_1.dll')):
        nvtoolsext_dll_path = os.path.join(
            os.getenv('NVTOOLSEXT_PATH', 'C:\\Program Files\\NVIDIA Corporation\\NvToolsExt'), 'bin', 'x64')
    else:
        nvtoolsext_dll_path = ''

    import importlib.util
    import glob
    spec = importlib.util.spec_from_file_location('torch_version', os.path.join(th_root, 'version.py'))
    torch_version = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(torch_version)
    if torch_version.cuda and len(glob.glob(os.path.join(th_dll_path, 'cudart64*.dll'))) == 0 and \
            len(glob.glob(os.path.join(py_dll_path, 'cudart64*.dll'))) == 0:
        cuda_version = torch_version.cuda
        cuda_version_1 = cuda_version.replace('.', '_')
        cuda_path_var = 'CUDA_PATH_V' + cuda_version_1
        default_path = 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v' + cuda_version
        cuda_path = os.path.join(os.getenv(cuda_path_var, default_path), 'bin')
    else:
        cuda_path = ''

    if not is_conda and sys.version_info >= (3, 8):
        dll_paths = list(filter(os.path.exists, [th_dll_path, py_dll_path, nvtoolsext_dll_path, cuda_path]))

        for dll_path in dll_paths:
            os.add_dll_directory(dll_path)
    else:
        dll_paths = [th_dll_path, py_dll_path, nvtoolsext_dll_path, cuda_path]
        dll_paths = list(filter(os.path.exists, dll_paths)) + [os.environ['PATH']]

        os.environ['PATH'] = ';'.join(dll_paths)
