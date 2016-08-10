from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.proto import caffe2_pb2
from caffe2.python import (
    workspace, device_checker, gradient_checker, test_util, core)
import contextlib
import copy
import hypothesis
import hypothesis.extra.numpy
import hypothesis.strategies as st
import numpy as np
import os


def is_sandcastle():
    if os.getenv('SANDCASTLE') == '1':
        return True
    elif os.getenv('TW_JOB_USER') == 'sandcastle':
        return True
    return False


hypothesis.settings.register_profile(
    "sandcastle",
    hypothesis.settings(
        max_examples=100,
        verbosity=hypothesis.Verbosity.verbose))

hypothesis.settings.register_profile(
    "dev",
    hypothesis.settings(
        max_examples=10,
        verbosity=hypothesis.Verbosity.verbose))
hypothesis.settings.register_profile(
    "debug",
    hypothesis.settings(
        max_examples=1000,
        verbosity=hypothesis.Verbosity.verbose))
hypothesis.settings.load_profile(
    'sandcastle' if is_sandcastle() else os.getenv('CAFFE2_HYPOTHESIS_PROFILE',
                                                   'dev')
)


def dims(min_value=1, max_value=5):
    return st.integers(min_value=min_value, max_value=max_value)


def elements_of_type(dtype=np.float32, filter_=None):
    elems = None
    if dtype in (np.float32, np.float64):
        elems = st.floats(min_value=-1.0, max_value=1.0)
    elif dtype is np.int32:
        elems = st.integers(min_value=0, max_value=2 ** 31 - 1)
    elif dtype is np.int64:
        elems = st.integers(min_value=0, max_value=2 ** 63 - 1)
    elif dtype is np.bool:
        elems = st.booleans()
    else:
        raise ValueError("Unexpected dtype without elements provided")
    return elems if filter_ is None else elems.filter(filter_)


def arrays(dims, dtype=np.float32, elements=None):
    if elements is None:
        elements = elements_of_type(dtype)
    return hypothesis.extra.numpy.arrays(dtype, dims, elements=elements)


def tensor(min_dim=1, max_dim=4, dtype=np.float32, elements=None, **kwargs):
    dims_ = st.lists(dims(**kwargs), min_size=min_dim, max_size=max_dim)
    return dims_.flatmap(lambda dims: arrays(dims, dtype, elements))


def segment_ids(size, is_sorted):
    if is_sorted:
        return arrays(
            [size],
            dtype=np.int32,
            elements=st.booleans()).map(
                lambda x: np.cumsum(x, dtype=np.int32) - x[0])
    else:
        return arrays(
            [size],
            dtype=np.int32,
            elements=st.integers(min_value=0, max_value=2 * size))


def segmented_tensor(min_dim=1, max_dim=4, dtype=np.float32, is_sorted=True,
                     elements=None, **kwargs):
    data_dims_ = st.lists(dims(**kwargs), min_size=min_dim, max_size=max_dim)
    return data_dims_.flatmap(lambda data_dims: st.tuples(
        arrays(data_dims, dtype, elements),
        segment_ids(data_dims[0], is_sorted=is_sorted),
    ))


def sparse_segmented_tensor(min_dim=1, max_dim=4, dtype=np.float32,
                            is_sorted=True, elements=None, **kwargs):
    data_dims_ = st.lists(dims(**kwargs), min_size=min_dim, max_size=max_dim)
    all_dims_ = data_dims_.flatmap(lambda data_dims: st.tuples(
        st.just(data_dims),
        st.integers(min_value=1, max_value=data_dims[0]),
    ))
    return all_dims_.flatmap(lambda dims: st.tuples(
        arrays(dims[0], dtype, elements),
        arrays(dims[1], dtype=np.int64, elements=st.integers(
            min_value=0, max_value=dims[0][0] - 1)),
        segment_ids(dims[1], is_sorted=is_sorted),
    ))


def tensors(n, min_dim=1, max_dim=4, dtype=np.float32, elements=None, **kwargs):
    dims_ = st.lists(dims(**kwargs), min_size=min_dim, max_size=max_dim)
    return dims_.flatmap(
        lambda dims: st.lists(arrays(dims, dtype, elements),
                              min_size=n, max_size=n))

cpu_do = caffe2_pb2.DeviceOption()
gpu_do = caffe2_pb2.DeviceOption(device_type=caffe2_pb2.CUDA)
device_options = [cpu_do] + ([gpu_do] if workspace.has_gpu_support else [])
# Include device option for each GPU
expanded_device_options = [cpu_do] + (
    [caffe2_pb2.DeviceOption(device_type=caffe2_pb2.CUDA, cuda_gpu_id=i)
     for i in range(workspace.NumCudaDevices())]
    if workspace.has_gpu_support else [])


def device_checker_device_options():
    return st.just(device_options)


def gradient_checker_device_option():
    return st.sampled_from(device_options)


gcs = dict(
    gc=gradient_checker_device_option(),
    dc=device_checker_device_options()
)

gcs_cpu_only = dict(gc=st.sampled_from([cpu_do]), dc=st.just([cpu_do]))


@contextlib.contextmanager
def temp_workspace(name=b"temp_ws"):
    old_ws_name = workspace.CurrentWorkspace()
    workspace.SwitchWorkspace(name, True)
    yield
    workspace.ResetWorkspace()
    workspace.SwitchWorkspace(old_ws_name)


class HypothesisTestCase(test_util.TestCase):
    def assertDeviceChecks(
        self,
        device_options,
        op,
        inputs,
        outputs_to_check,
        input_device_options=None,
        threshold=0.01
    ):
        dc = device_checker.DeviceChecker(
            threshold,
            device_options=device_options
        )
        self.assertTrue(
            dc.CheckSimple(op, inputs, outputs_to_check, input_device_options)
        )

    def assertGradientChecks(
        self,
        device_option,
        op,
        inputs,
        outputs_to_check,
        outputs_with_grads,
        grad_ops=None,
        threshold=0.005,
        stepsize=0.05,
        input_device_options=None,
    ):
        gc = gradient_checker.GradientChecker(
            stepsize=stepsize,
            threshold=threshold,
            device_option=device_option,
            workspace_name=str(device_option),
        )
        res, grad, grad_estimated = gc.CheckSimple(
            op, inputs, outputs_to_check, outputs_with_grads,
            grad_ops=grad_ops,
            input_device_options=input_device_options
        )
        self.assertEqual(grad.shape, grad_estimated.shape)
        self.assertTrue(res, "Gradient checks failed")

    def _assertGradReferenceChecks(
        self,
        op,
        inputs,
        ref_outputs,
        output_to_grad,
        grad_reference,
        threshold=1e-4,
    ):
        grad_blob_name = output_to_grad + '_grad'
        grad_ops, grad_map = core.GradientRegistry.GetBackwardPass(
            [op], {output_to_grad: grad_blob_name})
        output_grad = workspace.FetchBlob(output_to_grad)
        grad_ref_outputs = grad_reference(output_grad, ref_outputs, inputs)
        workspace.FeedBlob(grad_blob_name, workspace.FetchBlob(output_to_grad))
        workspace.RunOperatorsOnce(grad_ops)

        self.assertEqual(len(grad_ref_outputs), len(inputs))
        for (n, ref) in zip(op.input, grad_ref_outputs):
            grad_names = grad_map.get(n)
            if not grad_names:
                # no grad for this input
                self.assertIsNone(ref)
            else:
                if isinstance(grad_names, core.BlobReference):
                    # dense gradient
                    ref_vals = ref
                    ref_indices = None
                    val_name = grad_names
                else:
                    # sparse gradient
                    ref_vals, ref_indices = ref
                    val_name = grad_names.values
                vals = workspace.FetchBlob(str(val_name))
                np.testing.assert_allclose(vals, ref_vals,
                                           atol=threshold, rtol=threshold)
                if ref_indices is not None:
                    indices = workspace.FetchBlob(str(grad_names.indices))
                    np.testing.assert_allclose(indices, ref_indices,
                                               atol=1e-4, rtol=1e-4)

    def assertReferenceChecks(
        self,
        device_option,
        op,
        inputs,
        reference,
        input_device_options={},
        threshold=1e-4,
        output_to_grad=None,
        grad_reference=None,
    ):
        op = copy.deepcopy(op)
        op.device_option.CopyFrom(device_option)

        with temp_workspace():
            for (n, b) in zip(op.input, inputs):
                workspace.FeedBlob(
                    n,
                    b,
                    device_option=input_device_options.get(n, device_option)
                )
            workspace.RunOperatorOnce(op)
            reference_outputs = reference(*inputs)
            if not (isinstance(reference_outputs, tuple) or
                    isinstance(reference_outputs, list)):
                raise RuntimeError(
                    "You are providing a wrong reference implementation. A "
                    "proper one should return a tuple/list of numpy arrays.")
            self.assertEqual(len(reference_outputs), len(op.output))
            outs = []
            for (n, ref) in zip(op.output, reference_outputs):
                output = workspace.FetchBlob(n)
                if output.dtype.kind in ('S', 'O'):
                    np.testing.assert_array_equal(output, ref)
                else:
                    np.testing.assert_allclose(
                        output, ref, atol=threshold, rtol=threshold)
                outs.append(output)
            if grad_reference and output_to_grad:
                self._assertGradReferenceChecks(
                    op, inputs, reference_outputs,
                    output_to_grad, grad_reference)
            return outs

    def assertValidationChecks(
        self,
        device_option,
        op,
        inputs,
        validator,
        input_device_options={},
        as_kwargs=True
    ):
        if as_kwargs:
            assert len(set(list(op.input) + list(op.output))) == \
                len(op.input) + len(op.output), \
                "in-place ops are not supported in as_kwargs mode"
        op = copy.deepcopy(op)
        op.device_option.CopyFrom(device_option)

        with temp_workspace():
            for (n, b) in zip(op.input, inputs):
                workspace.FeedBlob(
                    n,
                    b,
                    device_option=input_device_options.get(n, device_option)
                )
            workspace.RunOperatorOnce(op)
            outputs = [workspace.FetchBlob(n) for n in op.output]
            if as_kwargs:
                validator(**dict(zip(
                    list(op.input) + list(op.output), inputs + outputs)))
            else:
                validator(inputs=inputs, outputs=outputs)
