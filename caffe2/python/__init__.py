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
    # first get nvToolsExt PATH
    def get_nvToolsExt_path():
        NVTOOLEXT_HOME = os.getenv('NVTOOLSEXT_PATH', 'C:\\Program Files\\NVIDIA Corporation\\NvToolsExt')

        if os.path.exists(NVTOOLEXT_HOME):
            return NVTOOLEXT_HOME + '\\bin\\x64\\'
        else:
            return ''

    py_dll_path = os.path.join(os.path.dirname(sys.executable), 'Library\\bin')
    th_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'torch')
    th_dll_path = th_root + '\\lib\\'

    dll_paths = [th_dll_path, py_dll_path, get_nvToolsExt_path(), os.environ['PATH']]

    # then add the path to env
    os.environ['PATH'] = ';'.join(dll_paths)
