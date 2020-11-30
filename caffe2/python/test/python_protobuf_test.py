



# make sure we use cpp implementation of protobuf
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"

# import cpp extension first
from caffe2.python import core
# then import protobuf
from caffe2.proto import caffe2_pb2, metanet_pb2

import unittest


class TestCrossProtoCalls(unittest.TestCase):
    def testSimple(self):
        net = caffe2_pb2.NetDef()
        meta = metanet_pb2.MetaNetDef()
        # if metanet_pb2 wasn't initialized properly the following fails with a
        # cryptic message: "Parameter to MergeFrom() must be instance of same
        # class: expected caffe2.NetDef got caffe2.NetDef."
        meta.nets.add(key="foo", value=net)
