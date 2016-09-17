from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import schema
import numpy as np

import unittest
import pickle


class TestDB(unittest.TestCase):
    def testPicklable(self):
        s = schema.Struct(
            ('field1', schema.Scalar(dtype=np.int32)),
            ('field2', schema.List(
                schema.Scalar(dtype=str))))
        s2 = pickle.loads(pickle.dumps(s))
        for r in (s, s2):
            self.assertTrue(isinstance(r.field1, schema.Scalar))
            self.assertTrue(isinstance(r.field2, schema.List))
            self.assertTrue(getattr(r, 'non_existent', None) is None)
