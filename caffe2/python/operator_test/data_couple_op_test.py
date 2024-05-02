




from caffe2.python import core, workspace
from caffe2.python.test_util import TestCase
import numpy as np


class TestDataCoupleOp(TestCase):

    def test_data_couple_op(self):
        param_array = np.random.rand(10, 10)
        gradient_array = np.random.rand(10, 10)
        extra_array = np.random.rand(10, 10)
        workspace.FeedBlob("param", param_array)
        workspace.FeedBlob("gradient", gradient_array)
        workspace.FeedBlob("extraBlob", extra_array)

        workspace.RunOperatorOnce(core.CreateOperator(
            "DataCouple",
            ["param", "gradient", "extraBlob"],
            ["param", "gradient"]))

        result1 = workspace.FetchBlob('param')
        result2 = workspace.FetchBlob('gradient')

        self.assertFalse((result1 - param_array).any())
        self.assertFalse((result2 - gradient_array).any())
