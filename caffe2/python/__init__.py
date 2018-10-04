from __future__ import absolute_import, division, print_function, unicode_literals
from caffe2.proto import caffe2_pb2
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
