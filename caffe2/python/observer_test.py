from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import unittest

from caffe2.python import model_helper, brew
import caffe2.python.workspace as ws


class TestObservers(unittest.TestCase):
    def setUp(self):
        ws.ResetWorkspace()
        self.model = model_helper.ModelHelper()
        brew.fc(self.model, "data", "y",
                    dim_in=4, dim_out=2,
                    weight_init=('ConstantFill', dict(value=1.0)),
                    bias_init=('ConstantFill', dict(value=0.0)),
                    axis=0)
        ws.FeedBlob("data", np.zeros([4], dtype='float32'))

        ws.RunNetOnce(self.model.param_init_net)
        ws.CreateNet(self.model.net)

    def testObserver(self):
        ob = self.model.net.AddObserver("TimeObserver")
        ws.RunNet(self.model.net)
        print(ob.average_time())
        num = self.model.net.NumObservers()
        self.model.net.RemoveObserver(ob)
        assert(self.model.net.NumObservers() + 1 == num)
