from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from hypothesis import given

from caffe2.python import core, workspace
from caffe2.proto import caffe2_pb2
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st

import numpy as np


def feed_inputs(inputs):
    for name, value in inputs.items():
        workspace.FeedBlob(name, value)


def assert_proto_equals(proto, expected):
    proto_lines = proto.strip().split('\n')
    expected_lines = expected.strip().split('\n')
    assert len(proto_lines) == len(expected_lines), \
        '{} != {}'.format(proto, expected)
    for left, right in zip(proto_lines, expected_lines):
        assert left.strip() == right.strip(), \
            '{} != {}'.format(proto, expected)


class TestCaffe2Script(hu.HypothesisTestCase):
    test_program = """
          def foo(a,b,X,W) -> (c):
              t = a + b*b
              c = FC(X,W,t)
          def testIf(c0,c1,t,f) -> (r):
              if c0 < c1:
                  r = t
              else:
                  r = f
              r = Add(r,3f,broadcast=1)
          def testWhile(r) -> (r):
              m = 0
              while m < 4:
                  # Plus operator automatically broadcasts, and we cannot
                  # do in-place B and C arguments when we broadcast, so use
                  # an explicit Add op.
                  r = Add(r, r)
                  m = m + 1
      """

    @given(firstdim=st.integers(min_value=1, max_value=4096),
           seconddim=st.integers(min_value=1, max_value=4096),
           seed=st.integers(min_value=0, max_value=65536),
           **hu.gcs)
    def test_foo(self, firstdim, seconddim, seed, gc, dc):
        np.random.seed(int(seed))
        inputs = {}
        a = inputs['a'] = np.random.rand(seconddim).astype(np.float32)
        b = inputs['b'] = np.random.rand(seconddim).astype(np.float32)
        X = inputs['X'] = np.random.rand(firstdim, firstdim).astype(np.float32)
        W = inputs['W'] = np.random.rand(seconddim, firstdim).astype(np.float32)

        feed_inputs(inputs)

        CU = core.C.CompilationUnit()
        CU.define(self.test_program)
        CU.create_net('foo').run()

        ref_t = a + b * b
        ref_c = np.matmul(X, W.transpose()) + ref_t
        actual_c = workspace.FetchBlob('c')

        np.testing.assert_allclose(actual_c, ref_c, rtol=1e-05)

    def test_trinary(self):
        CU = core.C.CompilationUnit()
        CU.define("""
            def foo(c) -> (d):
                d = 1 + (2 if c else 4)
        """)
        workspace.FeedBlob('c', np.ones((1), dtype=bool))
        net = CU.create_net('foo')
        net.run()
        assert(3 == workspace.FetchBlob('d'))
        workspace.FeedBlob('c', np.zeros((1), dtype=bool))
        net.run()
        assert(5 == workspace.FetchBlob('d'))

    def test_bool_literal(self):
        CU = core.C.CompilationUnit()
        CU.define("""
            def foo() -> (a,b):
                a = True
                b = False
        """)
        net = CU.create_net('foo')
        net.run()
        assert(workspace.FetchBlob('a'))
        assert(not workspace.FetchBlob('b'))

    def test_bool_operators(self):
        CU = core.C.CompilationUnit()
        CU.define("""
            def foo() -> (a, b, c, d, e):
                a = True and False
                b = True or False
                c = not b
                d = not False or True
                e = not (1 if a else 0) == (1 if b else 0)
        """)
        net = CU.create_net('foo')
        net.run()
        assert(not workspace.FetchBlob('a'))
        assert(workspace.FetchBlob('b'))
        assert(not workspace.FetchBlob('c'))
        assert(workspace.FetchBlob('d'))
        assert(workspace.FetchBlob('e'))

    def expect_fail(self, fn, msg):
        try:
            fn()
        except RuntimeError as r:
            if msg not in str(r):
                raise RuntimeError(
                    "Failed wrong: expected string '{}' ".format(msg) +
                    "in error message but found\n{}".format(str(r)))

    def test_fails(self):
        def fail_inputs():
            CU = core.C.CompilationUnit()
            CU.define("""
                def foo() -> ():
                    Print(1,4)
            """)
        self.expect_fail(fail_inputs, "expects 1 inputs but found 2")

        def fail_undef():
            CU = core.C.CompilationUnit()
            CU.define("""
                def foo(a) -> (b):
                    a = what()
            """)
        self.expect_fail(fail_undef, "attempting to call unknown operation")

        def fail_schema():
            CU = core.C.CompilationUnit()
            CU.define("""
                def foo(a) -> (b):
                    a = FC(a,a,a)
            """)
        self.expect_fail(fail_schema, "failed schema checking")

    def test_print(self):
        CU = core.C.CompilationUnit()
        CU.define("""
            def foo() -> ():
                a = 1
                Print(a)
                Print(a+1)
                _ = 4
                Print(_) # verify in print this isn't _ but some temorary
                Print(1)
                Print(1.f)
                Print(3.0)
        """)
        net = CU.create_net('foo')
        net.run()

    def test_method(self):
        CU = core.C.CompilationUnit()
        CU.define("""
            def foo() -> (a):
                a = (3+1).Add(4).Add(1)
        """)
        net = CU.create_net('foo')
        net.run()
        assert(9 == workspace.FetchBlob('a'))

    def test_plus_eq(self):
        CU = core.C.CompilationUnit()
        CU.define("""
            def foo() -> (a):
                a = 4
                a += 1
        """)
        net = CU.create_net('foo')
        net.run()
        assert(5 == workspace.FetchBlob('a'))

    def test_cast(self):
        CU = core.C.CompilationUnit()
        CU.define("""
            def foo() -> (a):
                a = int(4.5f)
        """)
        net = CU.create_net('foo')
        net.run()
        assert(4 == workspace.FetchBlob('a'))

    def test_global(self):
        CU = core.C.CompilationUnit()
        CU.define("""
            def foo() -> (a):
                global m
                m.a = 4
                m.b = 5
                a = m.a + m.b
        """)
        net = CU.create_net('foo')
        net.run()
        assert(9 == workspace.FetchBlob('a'))

    def test_module_as_arg_ret(self):
        CU = core.C.CompilationUnit()
        CU.define("""
            def bar(a,c) -> (b):
                b = Module()
                temp = a.second
                b.first = temp
                b.second = a.first + c
            def foo() -> (a,b):
                x = Module()
                x.first = 1
                x.second = 2
                x.y = bar(x,4)
                a = x.y.first
                b = x.y.second
        """)
        net = CU.create_net('foo')
        net.run()
        assert(2 == workspace.FetchBlob('a'))
        assert(5 == workspace.FetchBlob('b'))

    def test_call_extern(self):
        CU = core.C.CompilationUnit()
        net = caffe2_pb2.NetDef()
        net.op.extend([
            core.CreateOperator(
                'Mul',
                ['i', 'i'],
                ['o'],
            )
        ])
        net.external_input.append('i')
        net.external_output.append('o')

        CU.extern("myActualExtern", net)
        CU.define("""
            def myExtern(x) -> (y):
                t = x
                if t > 1:
                    y = t * t
                else:
                    y = 5
            def foo() -> (b):
                a = 4
                a += 1
                b = 2 + myExtern(a) + myExtern(a, rename=False) + myActualExtern(a)
        """)
        net = CU.create_net('foo')
        net.run()
        assert(77 == workspace.FetchBlob('b'))

    @given(seed=st.integers(min_value=0, max_value=65536), **hu.gcs)
    def test_if(self, seed, gc, dc):
        np.random.seed(int(seed))
        inputs = {}
        c0 = inputs['c0'] = np.random.rand(1).astype(np.float32)
        c1 = inputs['c1'] = np.random.rand(1).astype(np.float32)
        t = inputs['t'] = np.random.rand(3, 3).astype(np.float32)
        f = inputs['f'] = np.random.rand(3, 3).astype(np.float32)

        feed_inputs(inputs)

        CU = core.C.CompilationUnit()
        CU.define(self.test_program)
        CU.create_net('testIf').run()

        if c0 < c1:
            ref_r = t + 3
        else:
            ref_r = f + 3
        actual_r = workspace.FetchBlob('r')

        np.testing.assert_allclose(actual_r, ref_r)

    @given(seed=st.integers(min_value=0, max_value=65536), **hu.gcs)
    def test_while(self, seed, gc, dc):
        np.random.seed(int(seed))
        inputs = {}
        r = inputs['r'] = np.ones([3, 3]).astype(np.float32)

        feed_inputs(inputs)

        CU = core.C.CompilationUnit()
        CU.define(self.test_program)
        CU.create_net('testWhile').run()

        m = 0
        while m < 4:
            r = r + r
            m = m + 1

        actual_r = workspace.FetchBlob('r')

        np.testing.assert_allclose(actual_r, r)

    @given(seed=st.integers(min_value=0, max_value=65536), **hu.gcs)
    def test_gather(self, seed, gc, dc):
        CU = core.C.CompilationUnit()
        CU.define("""
        def easy(tensor, indices) -> (output):
            output = tensor[indices]
        def hard(tensor, i, j, k) -> (output):
            output = tensor[i][j][k]
        """)

        # First check that the generated proto is as expected. This tests that
        # we desugar the gather syntax correctly and emit the right code.
        proto = CU.get_proto('easy')
        assert_proto_equals(proto, """
            name: "easy"
            op {
              input: "tensor"
              input: "indices"
              output: "output"
              type: "Gather"
            }""")

        proto = CU.get_proto('hard')
        assert_proto_equals(proto, """
            name: "hard"
            op {
              input: "tensor"
              input: "i"
              output: "$t1"
              type: "Gather"
            }
            op {
              input: "$t1"
              input: "j"
              output: "$t0"
              type: "Gather"
            }
            op {
              input: "$t0"
              input: "k"
              output: "output"
              type: "Gather"
            }""")

        # Now just test that the effect of the generated code is as expected.
        np.random.seed(int(seed))
        tensor = np.random.rand(5, 4, 3).astype(np.float32)
        indices = np.random.randint(len(tensor), size=(5, 5))

        feed_inputs(dict(tensor=tensor, indices=indices))

        net = CU.create_net('easy')
        net.run()

        output = workspace.FetchBlob('output')
        expected_output = [tensor[sample] for sample in indices]
        np.testing.assert_allclose(output, expected_output)

    @given(seed=st.integers(min_value=0, max_value=65536), **hu.gcs)
    def test_slice(self, seed, gc, dc):
        CU = core.C.CompilationUnit()
        CU.define("""
        def slice_from_tensor(tensor, start, end) -> (output):
            output = tensor[start:end]
        def slice_from_vector(vector, start, end) -> (a, b, c, d):
            a = vector[start:end]
            b = vector[start:]
            c = vector[:end]
            d = vector[:]
        """)

        # slice_from_tensor
        proto = CU.get_proto('slice_from_tensor')
        assert_proto_equals(proto, """
            name: "slice_from_tensor"
            op {
              input: "tensor"
              input: "start"
              input: "end"
              output: "output"
              type: "Slice"
            }""")

        np.random.seed(int(seed))
        tensor = np.random.rand(5, 4, 3).astype(np.float32)
        start = np.array([0, 1, 0], dtype=np.int32)
        end = np.array([-1, 2, -1], dtype=np.int32)

        feed_inputs(dict(tensor=tensor, start=start, end=end))

        net = CU.create_net('slice_from_tensor')
        net.run()

        output = workspace.FetchBlob('output')
        np.testing.assert_allclose(output, tensor[:, 1:2])

        # slice_from_vector
        proto = CU.get_proto('slice_from_vector')
        assert_proto_equals(proto, """
            name: "slice_from_vector"
            op {
              input: "vector"
              input: "start"
              input: "end"
              output: "a"
              type: "Slice"
            }
            op {
              output: "$t0"
              type: "ConstantFill"
              arg {
                name: "dtype"
                i: 2
              }
              arg {
                name: "value"
                i: -1
              }
              arg {
                name: "shape"
                ints: 1
              }
            }
            op {
              input: "vector"
              input: "start"
              input: "$t0"
              output: "b"
              type: "Slice"
            }
            op {
              output: "$t1"
              type: "ConstantFill"
              arg {
                name: "dtype"
                i: 2
              }
             arg {
                name: "value"
                i: 0
              }
              arg {
                name: "shape"
                ints: 1
              }
            }
            op {
              input: "vector"
              input: "$t1"
              input: "end"
              output: "c"
              type: "Slice"
            }
            op {
              output: "$t2"
              type: "ConstantFill"
              arg {
                name: "dtype"
                i: 2
              }
             arg {
                name: "value"
                i: 0
              }
              arg {
                name: "shape"
                ints: 1
              }
            }
            op {
              output: "$t3"
              type: "ConstantFill"
              arg {
                name: "dtype"
                i: 2
              }
             arg {
                name: "value"
                i: -1
              }
              arg {
                name: "shape"
                ints: 1
              }
            }
            op {
              input: "vector"
              input: "$t2"
              input: "$t3"
              output: "d"
              type: "Slice"
            }""")

        vector = np.random.rand(10).astype(np.float32)
        start = np.array([2], dtype=np.int32)
        end = np.array([6], dtype=np.int32)
        feed_inputs(dict(vector=vector, start=start, end=end))

        net = CU.create_net('slice_from_vector')
        net.run()

        output = workspace.FetchBlob('a')
        np.testing.assert_allclose(output, vector[2:6])

        output = workspace.FetchBlob('b')
        np.testing.assert_allclose(output, vector[2:])

        output = workspace.FetchBlob('c')
        np.testing.assert_allclose(output, vector[:6])

        output = workspace.FetchBlob('d')
        np.testing.assert_allclose(output, vector)
