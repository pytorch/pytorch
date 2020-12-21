




from caffe2.python import core
from caffe2.python.test_util import rand_array
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
from hypothesis import given, settings
import hypothesis.strategies as st
import numpy as np

class TestScatterOps(serial.SerializedTestCase):
    # TODO(dzhulgakov): add test cases for failure scenarios
    @given(num_args=st.integers(1, 5),
           first_dim=st.integers(1, 20),
           index_dim=st.integers(1, 10),
           extra_dims=st.lists(st.integers(1, 4), min_size=0, max_size=3),
           ind_type=st.sampled_from([np.int32, np.int64]),
           **hu.gcs)
    @settings(deadline=10000)
    def testScatterWeightedSum(
        self, num_args, first_dim, index_dim, extra_dims, ind_type, gc, dc):
        ins = ['data', 'w0', 'indices']
        for i in range(1, num_args + 1):
            ins.extend(['x' + str(i), 'w' + str(i)])
        op = core.CreateOperator(
            'ScatterWeightedSum',
            ins,
            ['data'],
            device_option=gc)
        def ref(d, w0, ind, *args):
            r = d.copy()
            for i in ind:
                r[i] *= w0
            for i in range(0, len(args), 2):
                x = args[i]
                w = args[i+1]
                for i, j in enumerate(ind):
                    r[j] += w * x[i]
            return [r]

        d = rand_array(first_dim, *extra_dims)
        ind = np.random.randint(0, first_dim, index_dim).astype(ind_type)
        # ScatterWeightedSumOp only supports w0=1.0 in CUDAContext
        if(gc == hu.gpu_do or gc == hu.hip_do):
            w0 = np.array(1.0).astype(np.float32)
        else:
            w0 = rand_array()
        inputs = [d, w0, ind]
        for _ in range(1, num_args + 1):
            x = rand_array(index_dim, *extra_dims)
            w = rand_array()
            inputs.extend([x,w])
        self.assertReferenceChecks(gc, op, inputs, ref, threshold=1e-3)

    @given(first_dim=st.integers(1, 20),
           index_dim=st.integers(1, 10),
           extra_dims=st.lists(st.integers(1, 4), min_size=0, max_size=3),
           data_type=st.sampled_from([np.float16, np.float32, np.int32, np.int64]),
           ind_type=st.sampled_from([np.int32, np.int64]),
           **hu.gcs)
    @settings(deadline=10000)
    def testScatterAssign(
            self, first_dim, index_dim, extra_dims, data_type, ind_type, gc, dc):
        op = core.CreateOperator('ScatterAssign',
                                 ['data', 'indices', 'slices'], ['data'])
        def ref(d, ind, x):
            r = d.copy()
            r[ind] = x
            return [r]

        # let's have indices unique
        if first_dim < index_dim:
            first_dim, index_dim = index_dim, first_dim
        d = (rand_array(first_dim, *extra_dims) * 10).astype(data_type)
        ind = np.random.choice(first_dim, index_dim,
                               replace=False).astype(ind_type)
        x = (rand_array(index_dim, *extra_dims) * 10).astype(data_type)
        self.assertReferenceChecks(gc, op, [d, ind, x], ref, threshold=1e-3, ensure_outputs_are_inferred=True)

if __name__ == "__main__":
    import unittest
    unittest.main()
