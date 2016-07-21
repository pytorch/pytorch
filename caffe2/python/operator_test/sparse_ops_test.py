from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
from caffe2.python import core, workspace
from caffe2.python.test_util import TestCase, rand_array


class TestScatterOps(TestCase):
    def test_configs(self):
        return [
            # first_dim, index_num, data_dims
            (1, 2, []),
            (5, 5, []),
            (2, 5, []),
            (1, 1, []),
            (13, 7, []),
            (13, 7, [2]),
            (1, 5, [3, 4]),
            (17, 8, [2, 2, 2]),
        ]
        # TODO(dzhulgakov): add test cases for failure scenarios

    def testScatterWeightedSum(self):
        for num_args in [1, 2]:
            ins = ['data', 'w0', 'indices']
            for i in range(1, num_args + 1):
                ins.extend(['x' + str(i), 'w' + str(i)])
            op = core.CreateOperator('ScatterWeightedSum', ins, ['data'])
            for first_dim, index_dim, extra_dims in self.test_configs():
                for dtype in [np.int32, np.int64]:
                    d = rand_array(first_dim, *extra_dims)
                    ind = np.random.randint(0, first_dim,
                                            index_dim).astype(dtype)
                    w0 = rand_array()
                    r = d.copy()
                    for i in ind:
                        r[i] *= w0

                    # forward
                    workspace.FeedBlob('data', d)
                    workspace.FeedBlob('w0', w0)
                    workspace.FeedBlob('indices', ind)
                    for inp in range(1, num_args + 1):
                        w = rand_array()
                        x = rand_array(index_dim, *extra_dims)
                        workspace.FeedBlob('x' + str(inp), x)
                        workspace.FeedBlob('w' + str(inp), w)
                        for i, j in enumerate(ind):
                            r[j] += w * x[i]
                    workspace.RunOperatorOnce(op)
                    out = workspace.FetchBlob('data')
                    np.testing.assert_allclose(out, r, rtol=1e-3)

    def testScatterAssign(self):
        op = core.CreateOperator('ScatterAssign',
                                 ['data', 'indices', 'slices'], ['data'])
        for first_dim, index_dim, extra_dims in self.test_configs():
            # let's have indices unique
            if first_dim < index_dim:
                first_dim, index_dim = index_dim, first_dim
            for dtype in [np.int32, np.int64]:
                d = rand_array(first_dim, *extra_dims)
                ind = np.random.choice(first_dim, index_dim,
                                       replace=False).astype(dtype)
                x = rand_array(index_dim, *extra_dims)

                r = d.copy()
                r[ind] = x

                # forward
                workspace.FeedBlob('data', d)
                workspace.FeedBlob('indices', ind)
                workspace.FeedBlob('slices', x)
                workspace.RunOperatorOnce(op)
                out = workspace.FetchBlob('data')
                np.testing.assert_allclose(out, r, rtol=1e-3)
