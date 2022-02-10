




from hypothesis import assume, given, settings
import hypothesis.strategies as st
import numpy as np

from caffe2.proto import caffe2_pb2
from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial


class TestDropout(serial.SerializedTestCase):

    @serial.given(X=hu.tensor(),
           in_place=st.booleans(),
           ratio=st.floats(0, 0.999),
           engine=st.sampled_from(["", "CUDNN"]),
           **hu.gcs)
    def test_dropout_is_test(self, X, in_place, ratio, engine, gc, dc):
        """Test with is_test=True for a deterministic reference impl."""
        # TODO(lukeyeager): enable this path when the GPU path is fixed
        if in_place:
            # Skip if trying in-place on GPU
            assume(not (gc.device_type in {caffe2_pb2.CUDA, caffe2_pb2.HIP} and engine == ''))
            # If in-place on CPU, don't compare with GPU
            dc = dc[:1]

        op = core.CreateOperator("Dropout", ["X"],
                                 ["X" if in_place else "Y"],
                                 ratio=ratio, engine=engine, is_test=True)

        self.assertDeviceChecks(dc, op, [X], [0])
        # No sense in checking gradients for test phase

        def reference_dropout_test(x):
            return x, np.ones(x.shape, dtype=np.bool)
        self.assertReferenceChecks(
            gc, op, [X], reference_dropout_test,
            # The 'mask' output may be uninitialized
            outputs_to_check=[0])

    @given(X=hu.tensor(),
           in_place=st.booleans(),
           output_mask=st.booleans(),
           engine=st.sampled_from(["", "CUDNN"]),
           **hu.gcs)
    @settings(deadline=10000)
    def test_dropout_ratio0(self, X, in_place, output_mask, engine, gc, dc):
        """Test with ratio=0 for a deterministic reference impl."""
        # TODO(lukeyeager): enable this path when the op is fixed
        if in_place:
            # Skip if trying in-place on GPU
            assume(gc.device_type not in {caffe2_pb2.CUDA, caffe2_pb2.HIP})
            # If in-place on CPU, don't compare with GPU
            dc = dc[:1]
        is_test = not output_mask
        op = core.CreateOperator("Dropout", ["X"],
                                 ["X" if in_place else "Y"] +
                                 (["mask"] if output_mask else []),
                                 ratio=0.0, engine=engine,
                                 is_test=is_test)

        self.assertDeviceChecks(dc, op, [X], [0])
        if not is_test:
            self.assertGradientChecks(gc, op, [X], 0, [0])

        def reference_dropout_ratio0(x):
            return (x,) if is_test else (x, np.ones(x.shape, dtype=np.bool))
        self.assertReferenceChecks(
            gc, op, [X], reference_dropout_ratio0,
            # Don't check the mask with cuDNN because it's packed data
            outputs_to_check=None if engine != 'CUDNN' else [0])
