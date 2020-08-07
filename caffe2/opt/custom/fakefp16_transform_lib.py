#!/usr/bin/env python3
from __future__ import division
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import caffe2.opt.custom.fakefp16_transform_pybind11 as fakefp16Transform
from caffe2.proto.caffe2_pb2 import NetDef


def fakeFp16FuseOps(net : NetDef) -> NetDef:
    net_str = net.SerializeToString()
    out_str = fakefp16Transform.fakeFp16FuseOps(net_str)
    out_net = NetDef()
    out_net.ParseFromString(out_str)

    return out_net
