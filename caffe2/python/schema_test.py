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
                schema.Scalar(dtype=str)))
        )
        s2 = pickle.loads(pickle.dumps(s))
        for r in (s, s2):
            self.assertTrue(isinstance(r.field1, schema.Scalar))
            self.assertTrue(isinstance(r.field2, schema.List))
            self.assertTrue(getattr(r, 'non_existent', None) is None)

    def testNormalizeField(self):
        s = schema.Struct(('field1', np.int32), ('field2', str))
        self.assertEquals(
            s,
            schema.Struct(
                ('field1', schema.Scalar(dtype=np.int32)),
                ('field2', schema.Scalar(dtype=str))
            )
        )

    def testTuple(self):
        s = schema.Tuple(np.int32, str, np.float32)
        s2 = schema.Struct(
            ('field_0', schema.Scalar(dtype=np.int32)),
            ('field_1', schema.Scalar(dtype=np.str)),
            ('field_2', schema.Scalar(dtype=np.float32))
        )
        self.assertEquals(s, s2)
        self.assertEquals(s[0], schema.Scalar(dtype=np.int32))
        self.assertEquals(s[1], schema.Scalar(dtype=np.str))
        self.assertEquals(s[2], schema.Scalar(dtype=np.float32))
        self.assertEquals(
            s[2, 0],
            schema.Struct(
                ('field_2', schema.Scalar(dtype=np.float32)),
                ('field_0', schema.Scalar(dtype=np.int32)),
            )
        )
        # test iterator behavior
        for i, (v1, v2) in enumerate(zip(s, s2)):
            self.assertEquals(v1, v2)
            self.assertEquals(s[i], v1)
            self.assertEquals(s2[i], v1)

    def testRawTuple(self):
        s = schema.RawTuple(2)
        self.assertEquals(
            s,
            schema.Struct(
                ('field_0', schema.Scalar()),
                ('field_1', schema.Scalar())))
        self.assertEquals(s[0], schema.Scalar())
        self.assertEquals(s[1], schema.Scalar())

    def testStructIndexing(self):
        s = schema.Struct(
            ('field1', schema.Scalar(dtype=np.int32)),
            ('field2', schema.List(schema.Scalar(dtype=str)))
        )
        self.assertEquals(s['field2'], s.field2)
        self.assertEquals(s['field2'], schema.List(schema.Scalar(dtype=str)))
        self.assertEquals(
            s['field2', 'field1'],
            schema.Struct(
                ('field2', schema.List(schema.Scalar(dtype=str))),
                ('field1', schema.Scalar(dtype=np.int32)),
            )
        )
