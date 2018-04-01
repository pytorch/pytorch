from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python.models.seq2seq import seq2seq_model_helper
from caffe2.python import scope, test_util


class Seq2SeqModelHelperTest(test_util.TestCase):
    def testConstuctor(self):
        model_name = 'TestModel'
        m = seq2seq_model_helper.Seq2SeqModelHelper(name=model_name)

        self.assertEqual(m.name, model_name)
        self.assertEqual(m.init_params, True)

        self.assertEqual(m.arg_scope, {
            'use_cudnn': True,
            'cudnn_exhaustive_search': False,
            'order': 'NHWC'
        })

    def testAddParam(self):
        m = seq2seq_model_helper.Seq2SeqModelHelper()

        param_name = 'test_param'
        param = m.AddParam(param_name, init_value=1)
        self.assertEqual(str(param), param_name)

    def testGetNonTrainableParams(self):
        m = seq2seq_model_helper.Seq2SeqModelHelper()

        m.AddParam('test_param1', init_value=1, trainable=True)
        p2 = m.AddParam('test_param2', init_value=2, trainable=False)

        self.assertEqual(
            m.GetNonTrainableParams(),
            [p2]
        )

        with scope.NameScope('A', reset=True):
            p3 = m.AddParam('test_param3', init_value=3, trainable=False)
            self.assertEqual(
                m.GetNonTrainableParams(),
                [p3]
            )

        self.assertEqual(
            m.GetNonTrainableParams(),
            [p2, p3]
        )

    def testGetAllParams(self):
        m = seq2seq_model_helper.Seq2SeqModelHelper()

        p1 = m.AddParam('test_param1', init_value=1, trainable=True)
        p2 = m.AddParam('test_param2', init_value=2, trainable=False)

        self.assertEqual(
            m.GetAllParams(),
            [p1, p2]
        )


if __name__ == "__main__":
    import unittest
    import random
    random.seed(2221)
    unittest.main()
