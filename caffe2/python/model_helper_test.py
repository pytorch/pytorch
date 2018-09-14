"""unittest for ModelHelper class"""

from __future__ import absolute_import, division, print_function

import unittest

from caffe2.python import brew, model_helper


class ModelHelperTest(unittest.TestCase):
    def test_get_complete_net_type(self):
        model = model_helper.ModelHelper("test_orig")
        brew.conv(
            model,
            "input",
            "conv",
            dim_in=3,
            dim_out=16,
            weight_init=("MSRAFill", {}),
            kernel=3,
            stride=1,
            pad=0,
        )
        model.net.Proto().type = "async_scheduling"
        net = model.GetCompleteNet()
        model2 = model_helper.ModelHelper("test_new")
        model2.ConstructInitTrainNetfromNet(net)
        self.assertTrue(model2.net.Proto().type, "async_scheduling")
        self.assertTrue(model2.param_init_net.Proto().type, "async_scheduling")

    def test_get_complete_net(self):
        model = model_helper.ModelHelper("test_orig")
        conv = brew.conv(
            model,
            "input",
            "conv",
            dim_in=3,
            dim_out=16,
            weight_init=("MSRAFill", {}),
            kernel=3,
            stride=1,
            pad=0,
        )
        conv = brew.spatial_bn(model, conv, "conv_bn", 16, epsilon=1e-3, is_test=False)
        conv = brew.relu(model, conv, "conv_relu")
        pred = brew.fc(model, conv, "pred", dim_in=16 * 3 * 3, dim_out=10)
        brew.softmax(model, pred, "softmax")
        net = model.GetCompleteNet()
        model2 = model_helper.ModelHelper("test_new")
        model2.ConstructInitTrainNetfromNet(net)

        net = model.param_init_net
        net2 = model2.param_init_net
        for op1, op2 in zip(net.Proto().op, net2.Proto().op):
            op1.debug_info = op1.debug_info + "/param_init_net"
            self.assertEqual(
                op1, op2, "op mismatch between {}\n and {}\n".format(op1, op2)
            )
        net = model.net
        net2 = model2.net
        for op1, op2 in zip(net.Proto().op, net2.Proto().op):
            self.assertEqual(
                op1, op2, "op mismatch between {}\n and {}\n".format(op1, op2)
            )
        # this is not guaranteed in other situations where user define own net
        self.assertEqual(
            sorted(map(str, net.external_inputs)),
            sorted(map(str, net2.external_inputs)),
        )
