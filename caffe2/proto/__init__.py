# NOTE: we have to import python protobuf here **before** we load cpp extension.
# Otherwise it breaks under certain build conditions if cpp implementation of
# protobuf is used. Presumably there's some registry in protobuf library and
# python side has to initialize the dictionary first, before static
# initialization in python extension does so. Otherwise, duplicated protobuf
# descriptors will be created and it can lead to obscure errors like
#   "Parameter to MergeFrom() must be instance of same class:
#    expected caffe2.NetDef got caffe2.NetDef."
#
# This has to be done for all python targets, so listing them here
from caffe2.proto import caffe2_pb2, metanet_pb2, torch_pb2
try:
    from caffe2.caffe2.fb.session.proto import session_pb2
except ImportError:
    pass
