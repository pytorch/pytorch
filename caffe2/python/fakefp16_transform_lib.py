#!/usr/bin/env python3




import caffe2.python._import_c_extension as C
from caffe2.proto.caffe2_pb2 import NetDef

def fakeFp16FuseOps(net : NetDef) -> NetDef:
    net_str = net.SerializeToString()

    out_str = C.fakeFp16FuseOps(net_str)
    out_net = NetDef()
    out_net.ParseFromString(out_str)

    return out_net
