




from caffe2.proto import caffe2_pb2
from caffe2.python import core
from hypothesis import assume, given, settings, HealthCheck
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import hypothesis.strategies as st
import numpy as np
import unittest


class TestFcOperator(serial.SerializedTestCase):
    def _run_test(self, n, m, k, transposed, multi_dim, dtype, engine, gc, dc):
        if dtype == np.float16:
            # fp16 only supported with CUDA/HIP
            assume(core.IsGPUDeviceType(gc.device_type))
            dc = [d for d in dc if core.IsGPUDeviceType(d.device_type)]

        if engine == 'TENSORCORE':
            # TensorCore only makes sense with CUDA
            assume(gc.device_type == caffe2_pb2.CUDA)
            # ensures TensorCore kernels can be called
            m *= 8
            k *= 8
            n *= 8

        X = np.random.rand(m, k).astype(dtype) - 0.5
        if multi_dim:
            if transposed:
                W = np.random.rand(k, n, 1, 1).astype(dtype) - 0.5
            else:
                W = np.random.rand(n, k, 1, 1).astype(dtype) - 0.5
        else:
            if transposed:
                W = np.random.rand(k, n).astype(dtype) - 0.5
            else:
                W = np.random.rand(n, k).astype(dtype) - 0.5
        b = np.random.rand(n).astype(dtype) - 0.5

        def fc_op(X, W, b):
            return [np.dot(X, W.reshape(n, k).transpose()) + b.reshape(n)]

        def fc_transposed_op(X, W, b):
            return [np.dot(X, W.reshape(k, n)) + b.reshape(n)]

        op = core.CreateOperator(
            'FCTransposed' if transposed else 'FC',
            ['X', 'W', 'b'],
            'out',
            engine=engine,
        )

        if dtype == np.float16 and core.IsGPUDeviceType(gc.device_type):
            a = caffe2_pb2.Argument()
            a.i = 1
            a.name = "float16_compute"
            op.arg.extend([a])

        # Check against numpy reference
        # ReferenceChecks is flaky on rocm with threshold of 1e-4 for fp16. Relaxing to 1e-3.
        threshold = 1e-3 if (gc.device_type == caffe2_pb2.HIP and dtype == np.float16) else 1e-4
        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X, W, b],
            reference=fc_transposed_op if transposed else fc_op,
            threshold=threshold
        )
        # Check over multiple devices
        self.assertDeviceChecks(dc, op, [X, W, b], [0])

        # Gradient checks
        threshold = 0.5 if dtype == np.float16 else 0.005
        stepsize = 0.5 if dtype == np.float16 else 0.05
        for i in range(3):
            self.assertGradientChecks(gc, op, [X, W, b], i, [0],
                                      threshold=threshold, stepsize=stepsize)

    @settings(max_examples=50, suppress_health_check=[HealthCheck.filter_too_much])
    @serial.given(n=st.integers(1, 5),
           m=st.integers(0, 5),
           k=st.integers(1, 5),
           multi_dim=st.sampled_from([True, False]),
           dtype=st.sampled_from([np.float32, np.float16]),
           engine=st.sampled_from(['', 'TENSORCORE']),
           **hu.gcs)
    def test_fc(self, **kwargs):
        self._run_test(transposed=False, **kwargs)

    @settings(max_examples=50, suppress_health_check=[HealthCheck.filter_too_much])
    @given(n=st.integers(1, 5),
           m=st.integers(0, 5),
           k=st.integers(1, 5),
           multi_dim=st.sampled_from([True, False]),
           dtype=st.sampled_from([np.float32, np.float16]),
           engine=st.sampled_from(['', 'TENSORCORE']),
           **hu.gcs)
    def test_fc_transposed(self, **kwargs):
        self._run_test(transposed=True, **kwargs)


if __name__ == "__main__":
    import unittest
    unittest.main()
