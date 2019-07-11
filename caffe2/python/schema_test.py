from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, schema
import numpy as np

import unittest
import pickle
import random


class TestDB(unittest.TestCase):
    def testPicklable(self):
        s = schema.Struct(
            ('field1', schema.Scalar(dtype=np.int32)),
            ('field2', schema.List(schema.Scalar(dtype=str)))
        )
        s2 = pickle.loads(pickle.dumps(s))
        for r in (s, s2):
            self.assertTrue(isinstance(r.field1, schema.Scalar))
            self.assertTrue(isinstance(r.field2, schema.List))
            self.assertTrue(getattr(r, 'non_existent', None) is None)

    def testListSubclassClone(self):
        class Subclass(schema.List):
            pass

        s = Subclass(schema.Scalar())
        clone = s.clone()
        self.assertIsInstance(clone, Subclass)
        self.assertEqual(s, clone)
        self.assertIsNot(clone, s)

    def testListWithEvictedSubclassClone(self):
        class Subclass(schema.ListWithEvicted):
            pass

        s = Subclass(schema.Scalar())
        clone = s.clone()
        self.assertIsInstance(clone, Subclass)
        self.assertEqual(s, clone)
        self.assertIsNot(clone, s)

    def testStructSubclassClone(self):
        class Subclass(schema.Struct):
            pass

        s = Subclass(
            ('a', schema.Scalar()),
        )
        clone = s.clone()
        self.assertIsInstance(clone, Subclass)
        self.assertEqual(s, clone)
        self.assertIsNot(clone, s)

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
            s, schema.Struct(
                ('field_0', schema.Scalar()), ('field_1', schema.Scalar())
            )
        )
        self.assertEquals(s[0], schema.Scalar())
        self.assertEquals(s[1], schema.Scalar())

    def testStructIndexing(self):
        s = schema.Struct(
            ('field1', schema.Scalar(dtype=np.int32)),
            ('field2', schema.List(schema.Scalar(dtype=str))),
            ('field3', schema.Struct()),
        )
        self.assertEquals(s['field2'], s.field2)
        self.assertEquals(s['field2'], schema.List(schema.Scalar(dtype=str)))
        self.assertEquals(s['field3'], schema.Struct())
        self.assertEquals(
            s['field2', 'field1'],
            schema.Struct(
                ('field2', schema.List(schema.Scalar(dtype=str))),
                ('field1', schema.Scalar(dtype=np.int32)),
            )
        )

    def testListInStructIndexing(self):
        a = schema.List(schema.Scalar(dtype=str))
        s = schema.Struct(
            ('field1', schema.Scalar(dtype=np.int32)),
            ('field2', a)
        )
        self.assertEquals(s['field2:lengths'], a.lengths)
        self.assertEquals(s['field2:values'], a.items)
        with self.assertRaises(KeyError):
            s['fields2:items:non_existent']
        with self.assertRaises(KeyError):
            s['fields2:non_existent']

    def testListWithEvictedInStructIndexing(self):
        a = schema.ListWithEvicted(schema.Scalar(dtype=str))
        s = schema.Struct(
            ('field1', schema.Scalar(dtype=np.int32)),
            ('field2', a)
        )
        self.assertEquals(s['field2:lengths'], a.lengths)
        self.assertEquals(s['field2:values'], a.items)
        self.assertEquals(s['field2:_evicted_values'], a._evicted_values)
        with self.assertRaises(KeyError):
            s['fields2:items:non_existent']
        with self.assertRaises(KeyError):
            s['fields2:non_existent']

    def testMapInStructIndexing(self):
        a = schema.Map(
            schema.Scalar(dtype=np.int32),
            schema.Scalar(dtype=np.float32),
        )
        s = schema.Struct(
            ('field1', schema.Scalar(dtype=np.int32)),
            ('field2', a)
        )
        self.assertEquals(s['field2:values:keys'], a.keys)
        self.assertEquals(s['field2:values:values'], a.values)
        with self.assertRaises(KeyError):
            s['fields2:keys:non_existent']

    def testPreservesMetadata(self):
        s = schema.Struct(
            ('a', schema.Scalar(np.float32)), (
                'b', schema.Scalar(
                    np.int32,
                    metadata=schema.Metadata(categorical_limit=5)
                )
            ), (
                'c', schema.List(
                    schema.Scalar(
                        np.int32,
                        metadata=schema.Metadata(categorical_limit=6)
                    )
                )
            )
        )
        # attach metadata to lengths field
        s.c.lengths.set_metadata(schema.Metadata(categorical_limit=7))

        self.assertEqual(None, s.a.metadata)
        self.assertEqual(5, s.b.metadata.categorical_limit)
        self.assertEqual(6, s.c.value.metadata.categorical_limit)
        self.assertEqual(7, s.c.lengths.metadata.categorical_limit)
        sc = s.clone()
        self.assertEqual(None, sc.a.metadata)
        self.assertEqual(5, sc.b.metadata.categorical_limit)
        self.assertEqual(6, sc.c.value.metadata.categorical_limit)
        self.assertEqual(7, sc.c.lengths.metadata.categorical_limit)
        sv = schema.from_blob_list(
            s, [
                np.array([3.4]), np.array([2]), np.array([3]),
                np.array([1, 2, 3])
            ]
        )
        self.assertEqual(None, sv.a.metadata)
        self.assertEqual(5, sv.b.metadata.categorical_limit)
        self.assertEqual(6, sv.c.value.metadata.categorical_limit)
        self.assertEqual(7, sv.c.lengths.metadata.categorical_limit)

    def testDupField(self):
        with self.assertRaises(ValueError):
            schema.Struct(
                ('a', schema.Scalar()),
                ('a', schema.Scalar()))

    def testAssignToField(self):
        with self.assertRaises(TypeError):
            s = schema.Struct(('a', schema.Scalar()))
            s.a = schema.Scalar()

    def testPreservesEmptyFields(self):
        s = schema.Struct(
            ('a', schema.Scalar(np.float32)),
            ('b', schema.Struct()),
        )
        sc = s.clone()
        self.assertIn("a", sc.fields)
        self.assertIn("b", sc.fields)
        sv = schema.from_blob_list(s, [np.array([3.4])])
        self.assertIn("a", sv.fields)
        self.assertIn("b", sv.fields)
        self.assertEqual(0, len(sv.b.fields))

    def testStructSubstraction(self):
        s1 = schema.Struct(
            ('a', schema.Scalar()),
            ('b', schema.Scalar()),
            ('c', schema.Scalar()),
        )
        s2 = schema.Struct(
            ('b', schema.Scalar())
        )
        s = s1 - s2
        self.assertEqual(['a', 'c'], s.field_names())

        s3 = schema.Struct(
            ('a', schema.Scalar())
        )
        s = s1 - s3
        self.assertEqual(['b', 'c'], s.field_names())

        with self.assertRaises(TypeError):
            s1 - schema.Scalar()

    def testStructNestedSubstraction(self):
        s1 = schema.Struct(
            ('a', schema.Scalar()),
            ('b', schema.Struct(
                ('c', schema.Scalar()),
                ('d', schema.Scalar()),
                ('e', schema.Scalar()),
                ('f', schema.Scalar()),
            )),
        )
        s2 = schema.Struct(
            ('b', schema.Struct(
                ('d', schema.Scalar()),
                ('e', schema.Scalar()),
            )),
        )
        s = s1 - s2
        self.assertEqual(['a', 'b:c', 'b:f'], s.field_names())

    def testStructAddition(self):
        s1 = schema.Struct(
            ('a', schema.Scalar())
        )
        s2 = schema.Struct(
            ('b', schema.Scalar())
        )
        s = s1 + s2
        self.assertIn("a", s.fields)
        self.assertIn("b", s.fields)
        with self.assertRaises(TypeError):
            s1 + s1
        with self.assertRaises(TypeError):
            s1 + schema.Scalar()

    def testStructNestedAddition(self):
        s1 = schema.Struct(
            ('a', schema.Scalar()),
            ('b', schema.Struct(
                ('c', schema.Scalar())
            )),
        )
        s2 = schema.Struct(
            ('b', schema.Struct(
                ('d', schema.Scalar())
            ))
        )
        s = s1 + s2
        self.assertEqual(['a', 'b:c', 'b:d'], s.field_names())

        s3 = schema.Struct(
            ('b', schema.Scalar()),
        )
        with self.assertRaises(TypeError):
            s = s1 + s3

    def testGetFieldByNestedName(self):
        st = schema.Struct(
            ('a', schema.Scalar()),
            ('b', schema.Struct(
                ('c', schema.Struct(
                    ('d', schema.Scalar()),
                )),
            )),
        )
        self.assertRaises(KeyError, st.__getitem__, '')
        self.assertRaises(KeyError, st.__getitem__, 'x')
        self.assertRaises(KeyError, st.__getitem__, 'x:y')
        self.assertRaises(KeyError, st.__getitem__, 'b:c:x')
        a = st['a']
        self.assertTrue(isinstance(a, schema.Scalar))
        bc = st['b:c']
        self.assertIn('d', bc.fields)
        bcd = st['b:c:d']
        self.assertTrue(isinstance(bcd, schema.Scalar))

    def testAddFieldByNestedName(self):
        f_a = schema.Scalar(blob=core.BlobReference('blob1'))
        f_b = schema.Struct(
            ('c', schema.Struct(
                ('d', schema.Scalar(blob=core.BlobReference('blob2'))),
            )),
        )
        f_x = schema.Struct(
            ('x', schema.Scalar(blob=core.BlobReference('blob3'))),
        )

        with self.assertRaises(TypeError):
            st = schema.Struct(
                ('a', f_a),
                ('b', f_b),
                ('b:c:d', f_x),
            )
        with self.assertRaises(TypeError):
            st = schema.Struct(
                ('a', f_a),
                ('b', f_b),
                ('b:c:d:e', f_x),
            )

        st = schema.Struct(
            ('a', f_a),
            ('b', f_b),
            ('e:f', f_x),
        )
        self.assertEqual(['a', 'b:c:d', 'e:f:x'], st.field_names())
        self.assertEqual(['blob1', 'blob2', 'blob3'], st.field_blobs())

        st = schema.Struct(
            ('a', f_a),
            ('b:c:e', f_x),
            ('b', f_b),
        )
        self.assertEqual(['a', 'b:c:e:x', 'b:c:d'], st.field_names())
        self.assertEqual(['blob1', 'blob3', 'blob2'], st.field_blobs())

        st = schema.Struct(
            ('a:a1', f_a),
            ('b:b1', f_b),
            ('a', f_x),
        )
        self.assertEqual(['a:a1', 'a:x', 'b:b1:c:d'], st.field_names())
        self.assertEqual(['blob1', 'blob3', 'blob2'], st.field_blobs())

    def testContains(self):
        st = schema.Struct(
            ('a', schema.Scalar()),
            ('b', schema.Struct(
                ('c', schema.Struct(
                    ('d', schema.Scalar()),
                )),
            )),
        )
        self.assertTrue('a' in st)
        self.assertTrue('b:c' in st)
        self.assertTrue('b:c:d' in st)
        self.assertFalse('' in st)
        self.assertFalse('x' in st)
        self.assertFalse('b:c:x' in st)
        self.assertFalse('b:c:d:x' in st)

    def testFromEmptyColumnList(self):
        st = schema.Struct()
        columns = st.field_names()
        rec = schema.from_column_list(col_names=columns)
        self.assertEqual(rec, schema.Struct())

    def testFromColumnList(self):
        st = schema.Struct(
            ('a', schema.Scalar()),
            ('b', schema.List(schema.Scalar())),
            ('c', schema.Map(schema.Scalar(), schema.Scalar()))
        )
        columns = st.field_names()
        # test that recovery works for arbitrary order
        for _ in range(10):
            some_blobs = [core.BlobReference('blob:' + x) for x in columns]
            rec = schema.from_column_list(columns, col_blobs=some_blobs)
            self.assertTrue(rec.has_blobs())
            self.assertEqual(sorted(st.field_names()), sorted(rec.field_names()))
            self.assertEqual([str(blob) for blob in rec.field_blobs()],
                             [str('blob:' + name) for name in rec.field_names()])
            random.shuffle(columns)

    def testStructGet(self):
        net = core.Net('test_net')
        s1 = schema.NewRecord(net, schema.Scalar(np.float32))
        s2 = schema.NewRecord(net, schema.Scalar(np.float32))
        t = schema.Tuple(s1, s2)
        assert t.get('field_0', None) == s1
        assert t.get('field_1', None) == s2
        assert t.get('field_2', None) is None

    def testScalarForVoidType(self):
        s0_good = schema.Scalar((None, (2, )))
        with self.assertRaises(TypeError):
            s0_bad = schema.Scalar((np.void, (2, )))

        s1_good = schema.Scalar(np.void)
        s2_good = schema.Scalar(None)
        assert s1_good == s2_good

    def testScalarShape(self):
        s0 = schema.Scalar(np.int32)
        self.assertEqual(s0.field_type().shape, ())

        s1_good = schema.Scalar((np.int32, 5))
        self.assertEqual(s1_good.field_type().shape, (5, ))

        with self.assertRaises(ValueError):
            s1_bad = schema.Scalar((np.int32, -1))

        s1_hard = schema.Scalar((np.int32, 1))
        self.assertEqual(s1_hard.field_type().shape, (1, ))

        s2 = schema.Scalar((np.int32, (2, 3)))
        self.assertEqual(s2.field_type().shape, (2, 3))

    def testDtypeForCoreType(self):
        dtype = schema.dtype_for_core_type(core.DataType.FLOAT16)
        self.assertEqual(dtype, np.float16)

        with self.assertRaises(TypeError):
            schema.dtype_for_core_type(100)
