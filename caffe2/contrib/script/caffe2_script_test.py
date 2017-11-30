from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from hypothesis import given

from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st

import numpy as np


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
              r = Add(r,3,broadcast=1i)
          def testWhile(m,r) -> (r):
              while m < 4:
                  r = r + r
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

        for name, inp in inputs.items():
            workspace.FeedBlob(name, inp)

        CU = core.C.CompilationUnit()
        CU.define(self.test_program)
        CU.create_net('foo').run()

        ref_t = a + b * b
        ref_c = np.matmul(X, W.transpose()) + ref_t
        actual_c = workspace.FetchBlob('c')

        np.testing.assert_allclose(actual_c, ref_c)

    @given(seed=st.integers(min_value=0, max_value=65536), **hu.gcs)
    def test_if(self, seed, gc, dc):
        np.random.seed(int(seed))
        inputs = {}
        c0 = inputs['c0'] = np.random.rand(1).astype(np.float32)
        c1 = inputs['c1'] = np.random.rand(1).astype(np.float32)
        t = inputs['t'] = np.random.rand(3, 3).astype(np.float32)
        f = inputs['f'] = np.random.rand(3, 3).astype(np.float32)

        for name, inp in inputs.items():
            workspace.FeedBlob(name, inp)

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
        m = inputs['m'] = np.zeros([]).astype(np.float32)

        for name, inp in inputs.items():
            workspace.FeedBlob(name, inp)

        CU = core.C.CompilationUnit()
        CU.define(self.test_program)
        CU.create_net('testWhile').run()

        while m < 4:
            r = r + r
            m = m + 1

        actual_r = workspace.FetchBlob('r')

        np.testing.assert_allclose(actual_r, r)
