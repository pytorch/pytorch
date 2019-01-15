from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, test_util, workspace
from caffe2.python.control_ops_grad import disambiguate_grad_if_op_output
from caffe2.python.model_helper import ModelHelper
import numpy as np


class TestControl(test_util.TestCase):
    def test_disambiguate_grad_if_op_output(self):
        workspace.FeedBlob("cond", np.array(True))
        workspace.FeedBlob("then_grad", np.array(1))
        workspace.FeedBlob("else_grad", np.array(2))

        then_model = ModelHelper(name="then_test_model")
        then_model.net.Copy("then_grad", "input_grad")

        else_model = ModelHelper(name="else_test_model")
        else_model.net.Copy("else_grad", "else_temp_grad")
        else_model.net.Copy("else_temp", "input_grad")

        # to BuildGradientGenerators, in forward pass, we need else temp
        # as one of the output. Which later on results in a grad op like this:
        grad_op = core.CreateOperator(
            "If",
            ["cond", "then_grad", "else_grad"],
            ["input_grad", "else_temp_grad"],
            then_net=then_model.net.Proto(),
            else_net=else_model.net.Proto(),
        )

        # in certain cases, another branch of the net also generates input_grad
        # and we call _DisambiguateGradOpOutput in core.py
        new_grad_output = "input_grad" + "_autosplit_" + "0"
        disambiguate_grad_if_op_output(grad_op, 0, new_grad_output)
        self.assertEqual(grad_op.output[0], new_grad_output)
        self.assertEqual(grad_op.arg[1].n.op[1].output[0], new_grad_output)
