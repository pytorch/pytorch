



import unittest
from caffe2.python.fakefp16_transform_lib import fakeFp16FuseOps
from caffe2.python import core

class Transformer(unittest.TestCase):
    def test_fuse(self):
        net_swish = core.Net("test_swish")
        net_swish_init = core.Net("test_swish_init")

        deq = core.CreateOperator("Int8DequantizeNNPI", ["Xq"], ["X"])
        swish = core.CreateOperator("SwishFakeFp16NNPI", ["X"], ["Y"])
        quant = core.CreateOperator("Int8QuantizeNNPI", ["Y"], ["Y_q"])
        net_swish.Proto().op.extend(
            [
                deq, swish, quant
            ]
        )
        print(net_swish.Proto())
        out_net = fakeFp16FuseOps(net_swish.Proto())
        assert(len(out_net.op) == 1)
