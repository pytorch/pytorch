import os
import sys
import warnings


try:
    from caffe2.proto import caffe2_pb2
except ImportError:
    warnings.warn('Caffe2 support is not enabled in this PyTorch build. '
                  'Please enable Caffe2 by building PyTorch from source with `BUILD_CAFFE2=1` flag.', stacklevel=TO_BE_DETERMINED)
    raise

# TODO: refactor & remove the following alias
caffe2_pb2.CPU = caffe2_pb2.PROTO_CPU
caffe2_pb2.CUDA = caffe2_pb2.PROTO_CUDA
caffe2_pb2.MKLDNN = caffe2_pb2.PROTO_MKLDNN
caffe2_pb2.OPENGL = caffe2_pb2.PROTO_OPENGL
caffe2_pb2.OPENCL = caffe2_pb2.PROTO_OPENCL
caffe2_pb2.IDEEP = caffe2_pb2.PROTO_IDEEP
caffe2_pb2.HIP = caffe2_pb2.PROTO_HIP
caffe2_pb2.COMPILE_TIME_MAX_DEVICE_TYPES = caffe2_pb2.PROTO_COMPILE_TIME_MAX_DEVICE_TYPES

if sys.platform == "win32":
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

    import ctypes
    kernel32 = ctypes.WinDLL('kernel32.dll', use_last_error=True)
    dll_paths = list(filter(os.path.exists, [th_dll_path, py_dll_path, nvtoolsext_dll_path, cuda_path]))
    with_load_library_flags = hasattr(kernel32, 'AddDllDirectory')
    prev_error_mode = kernel32.SetErrorMode(0x0001)

    kernel32.LoadLibraryW.restype = ctypes.c_void_p
    if with_load_library_flags:
        kernel32.LoadLibraryExW.restype = ctypes.c_void_p

    for dll_path in dll_paths:
        os.add_dll_directory(dll_path)

    dlls = glob.glob(os.path.join(th_dll_path, '*.dll'))
    path_patched = False
    for dll in dlls:
        is_loaded = False
        if with_load_library_flags:
            res = kernel32.LoadLibraryExW(dll, None, 0x00001100)
            last_error = ctypes.get_last_error()
            if res is None and last_error != 126:
                err = ctypes.WinError(last_error)
                err.strerror += ' Error loading "{}" or one of its dependencies.'.format(dll)
                raise err
            elif res is not None:
                is_loaded = True
        if not is_loaded:
            if not path_patched:
                os.environ['PATH'] = ';'.join(dll_paths + [os.environ['PATH']])
                path_patched = True
            res = kernel32.LoadLibraryW(dll)
            if res is None:
                err = ctypes.WinError(ctypes.get_last_error())
                err.strerror += ' Error loading "{}" or one of its dependencies.'.format(dll)
                raise err

    kernel32.SetErrorMode(prev_error_mode)
