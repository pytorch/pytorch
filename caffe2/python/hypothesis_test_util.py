## @package hypothesis_test_util
# Module caffe2.python.hypothesis_test_util
"""
The Hypothesis library uses *property-based testing* to check
invariants about the code under test under a variety of random inputs.

 The key idea here is to express properties of the code under test
(e.g. that it passes a gradient check, that it implements a reference
function, etc), and then generate random instances and verify they
satisfy these properties.

The main functions of interest are exposed on `HypothesisTestCase`.
You can usually just add a short function in this to generate an
arbitrary number of test cases for your operator.

The key functions are:

- `assertDeviceChecks(devices, op, inputs, outputs)`. This asserts that the
  operator computes the same outputs, regardless of which device it is executed
  on.
- `assertGradientChecks(device, op, inputs, output_,
  outputs_with_grads)`. This implements a standard numerical gradient checker
  for the operator in question.
- `assertReferenceChecks(device, op, inputs, reference)`. This runs the
  reference function (effectively calling `reference(*inputs)`, and comparing
  that to the output of output.

`hypothesis_test_util.py` exposes some useful pre-built samplers.

- `hu.gcs` - a gradient checker device (`gc`) and device checker devices (`dc`)

- `hu.gcs_cpu_only` - a CPU-only gradient checker device (`gc`) and
  device checker devices (`dc`). Used for when your operator is only
  implemented on the CPU.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.proto import caffe2_pb2
from caffe2.python import (
    workspace, device_checker, gradient_checker, test_util, core)
import contextlib
import copy
import functools
import hypothesis
import hypothesis.extra.numpy
import hypothesis.strategies as st
import logging
import numpy as np
import os
import six


def is_sandcastle():
    if os.getenv('SANDCASTLE') == '1':
        return True
    elif os.getenv('TW_JOB_USER') == 'sandcastle':
        return True
    return False


def is_travis():
    return 'TRAVIS' in os.environ


#  "min_satisfying_examples" setting has been deprecated in hypythesis
#  3.56.0 and removed in hypothesis 4.x
if hypothesis.version.__version_info__ >= (3, 56, 0):
    hypothesis.settings.register_profile(
        "sandcastle",
        hypothesis.settings(
            derandomize=True,
            suppress_health_check=[hypothesis.HealthCheck.too_slow],
            database=None,
            max_examples=100,
            verbosity=hypothesis.Verbosity.verbose))
    hypothesis.settings.register_profile(
        "dev",
        hypothesis.settings(
            suppress_health_check=[hypothesis.HealthCheck.too_slow],
            database=None,
            max_examples=10,
            verbosity=hypothesis.Verbosity.verbose))
    hypothesis.settings.register_profile(
        "debug",
        hypothesis.settings(
            suppress_health_check=[hypothesis.HealthCheck.too_slow],
            database=None,
            max_examples=1000,
            verbosity=hypothesis.Verbosity.verbose))
else:
    hypothesis.settings.register_profile(
        "sandcastle",
        hypothesis.settings(
            derandomize=True,
            suppress_health_check=[hypothesis.HealthCheck.too_slow],
            database=None,
            max_examples=100,
            min_satisfying_examples=1,
            verbosity=hypothesis.Verbosity.verbose))
    hypothesis.settings.register_profile(
        "dev",
        hypothesis.settings(
            suppress_health_check=[hypothesis.HealthCheck.too_slow],
            database=None,
            max_examples=10,
            min_satisfying_examples=1,
            verbosity=hypothesis.Verbosity.verbose))
    hypothesis.settings.register_profile(
        "debug",
        hypothesis.settings(
            suppress_health_check=[hypothesis.HealthCheck.too_slow],
            database=None,
            max_examples=1000,
            min_satisfying_examples=1,
            verbosity=hypothesis.Verbosity.verbose))

hypothesis.settings.load_profile(
    'sandcastle' if is_sandcastle() else os.getenv('CAFFE2_HYPOTHESIS_PROFILE',
                                                   'dev')
)


def dims(min_value=1, max_value=5):
    return st.integers(min_value=min_value, max_value=max_value)


def elements_of_type(dtype=np.float32, filter_=None):
    elems = None
    if dtype in (np.float16, np.float32, np.float64):
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
    return hypothesis.extra.numpy.arrays(
        dtype,
        dims,
        elements=elements,
    )


def tensor(min_dim=1,
           max_dim=4,
           dtype=np.float32,
           elements=None,
           **kwargs):
    dims_ = st.lists(dims(**kwargs), min_size=min_dim, max_size=max_dim)
    return dims_.flatmap(
        lambda dims: arrays(dims, dtype, elements))


def tensor1d(min_len=1, max_len=64, dtype=np.float32, elements=None):
    return tensor(1, 1, dtype, elements, min_value=min_len, max_value=max_len)


def segment_ids(size, is_sorted):
    if size == 0:
        return st.just(np.empty(shape=[0], dtype=np.int32))
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


def lengths(size, min_segments=None, max_segments=None, **kwargs):
    # First generate number of boarders between segments
    # Then create boarder values and add 0 and size
    # By sorting and computing diff we convert them to lengths of
    # possible 0 value
    if min_segments is None:
        min_segments = 0
    if max_segments is None:
        max_segments = size
    assert min_segments >= 0
    assert min_segments <= max_segments
    if size == 0 and max_segments == 0:
        return st.just(np.empty(shape=[0], dtype=np.int32))
    assert max_segments > 0, "size is not 0, need at least one segment"
    return st.integers(
        min_value=max(min_segments - 1, 0), max_value=max_segments - 1
    ).flatmap(
        lambda num_borders:
        hypothesis.extra.numpy.arrays(
            np.int32, num_borders, elements=st.integers(
                min_value=0, max_value=size
            )
        )
    ).map(
        lambda x: np.append(x, np.array([0, size], dtype=np.int32))
    ).map(sorted).map(np.diff)


def segmented_tensor(
    min_dim=1,
    max_dim=4,
    dtype=np.float32,
    is_sorted=True,
    elements=None,
    segment_generator=segment_ids,
    allow_empty=False,
    **kwargs
):
    gen_empty = st.booleans() if allow_empty else st.just(False)
    data_dims_ = st.lists(dims(**kwargs), min_size=min_dim, max_size=max_dim)
    data_dims_ = st.tuples(
        gen_empty, data_dims_
    ).map(lambda pair: ([0] if pair[0] else []) + pair[1])
    return data_dims_.flatmap(lambda data_dims: st.tuples(
        arrays(data_dims, dtype, elements),
        segment_generator(data_dims[0], is_sorted=is_sorted),
    ))


def lengths_tensor(min_segments=None, max_segments=None, *args, **kwargs):
    gen = functools.partial(
        lengths, min_segments=min_segments, max_segments=max_segments)
    return segmented_tensor(*args, segment_generator=gen, **kwargs)


def sparse_segmented_tensor(min_dim=1, max_dim=4, dtype=np.float32,
                            is_sorted=True, elements=None, allow_empty=False,
                            segment_generator=segment_ids, itype=np.int64,
                            **kwargs):
    gen_empty = st.booleans() if allow_empty else st.just(False)
    data_dims_ = st.lists(dims(**kwargs), min_size=min_dim, max_size=max_dim)
    all_dims_ = st.tuples(gen_empty, data_dims_).flatmap(
        lambda pair: st.tuples(
            st.just(pair[1]),
            (st.integers(min_value=1, max_value=pair[1][0]) if not pair[0]
             else st.just(0)),
        ))
    return all_dims_.flatmap(lambda dims: st.tuples(
        arrays(dims[0], dtype, elements),
        arrays(dims[1], dtype=itype, elements=st.integers(
            min_value=0, max_value=dims[0][0] - 1)),
        segment_generator(dims[1], is_sorted=is_sorted),
    ))


def sparse_lengths_tensor(**kwargs):
    return sparse_segmented_tensor(segment_generator=lengths, **kwargs)


def tensors(n, min_dim=1, max_dim=4, dtype=np.float32, elements=None, **kwargs):
    dims_ = st.lists(dims(**kwargs), min_size=min_dim, max_size=max_dim)
    return dims_.flatmap(
        lambda dims: st.lists(
            arrays(dims, dtype, elements),
            min_size=n,
            max_size=n))


def tensors1d(n, min_len=1, max_len=64, dtype=np.float32, elements=None):
    return tensors(
        n, 1, 1, dtype, elements, min_value=min_len, max_value=max_len
    )


cpu_do = caffe2_pb2.DeviceOption()
cuda_do = caffe2_pb2.DeviceOption(device_type=caffe2_pb2.CUDA)
hip_do = caffe2_pb2.DeviceOption(device_type=caffe2_pb2.HIP)
gpu_do =  caffe2_pb2.DeviceOption(device_type=workspace.GpuDeviceType) # CUDA or ROCm
# (bddppq) Do not rely on this no_hip option! It's just used to
# temporarily skip some flaky tests on ROCM before it's getting more mature.
_device_options_no_hip = [cpu_do] + ([cuda_do] if workspace.has_cuda_support else [])
device_options = _device_options_no_hip + ([hip_do] if workspace.has_hip_support else [])

# Include device option for each GPU
expanded_device_options = [cpu_do] + [
    caffe2_pb2.DeviceOption(device_type=workspace.GpuDeviceType, device_id=i)
    for i in range(workspace.NumGpuDevices())]


def device_checker_device_options():
    return st.just(device_options)


def gradient_checker_device_option():
    return st.sampled_from(device_options)


gcs = dict(
    gc=gradient_checker_device_option(),
    dc=device_checker_device_options()
)

gcs_cpu_only = dict(gc=st.sampled_from([cpu_do]), dc=st.just([cpu_do]))
gcs_cuda_only = dict(gc=st.sampled_from([cuda_do]), dc=st.just([cuda_do]))
gcs_gpu_only = dict(gc=st.sampled_from([gpu_do]), dc=st.just([gpu_do])) # CUDA or ROCm
gcs_no_hip = dict(gc=st.sampled_from(_device_options_no_hip), dc=st.just(_device_options_no_hip))


@contextlib.contextmanager
def temp_workspace(name=b"temp_ws"):
    old_ws_name = workspace.CurrentWorkspace()
    workspace.SwitchWorkspace(name, True)
    yield
    workspace.ResetWorkspace()
    workspace.SwitchWorkspace(old_ws_name)


def runOpBenchmark(
    device_option,
    op,
    inputs,
    input_device_options=None,
    iterations=10,
):
    op = copy.deepcopy(op)
    op.device_option.CopyFrom(device_option)
    net = caffe2_pb2.NetDef()
    net.op.extend([op])
    net.name = op.name if op.name else "test"

    with temp_workspace():
        _input_device_options = input_device_options or \
            core.InferOpBlobDevicesAsDict(op)[0]
        for (n, b) in zip(op.input, inputs):
            workspace.FeedBlob(
                n,
                b,
                device_option=_input_device_options.get(n, device_option)
            )
        workspace.CreateNet(net)
        ret = workspace.BenchmarkNet(net.name, 1, iterations, True)
    return ret


def runOpOnInput(
    device_option,
    op,
    inputs,
    input_device_options=None,
):
    op = copy.deepcopy(op)
    op.device_option.CopyFrom(device_option)

    with temp_workspace():
        if (len(op.input) > len(inputs)):
            raise ValueError(
                'must supply an input for each input on the op: %s vs %s' %
                (op.input, inputs))
        _input_device_options = input_device_options or \
            core.InferOpBlobDevicesAsDict(op)[0]
        for (n, b) in zip(op.input, inputs):
            workspace.FeedBlob(
                n,
                b,
                device_option=_input_device_options.get(n, device_option)
            )
        workspace.RunOperatorOnce(op)
        outputs_to_check = list(range(len(op.output)))
        outs = []
        for output_index in outputs_to_check:
            output_blob_name = op.output[output_index]
            output = workspace.FetchBlob(output_blob_name)
            outs.append(output)
        return outs


class HypothesisTestCase(test_util.TestCase):
    """
    A unittest.TestCase subclass with some helper functions for
    utilizing the `hypothesis` (hypothesis.readthedocs.io) library.
    """

    def assertDeviceChecks(
        self,
        device_options,
        op,
        inputs,
        outputs_to_check,
        input_device_options=None,
        threshold=0.01
    ):
        """
        Asserts that the operator computes the same outputs, regardless of
        which device it is executed on.

        Useful for checking the consistency of GPU and CPU
        implementations of operators.

        Usage example:

            @given(inputs=hu.tensors(n=2), in_place=st.booleans(), **hu.gcs)
            def test_sum(self, inputs, in_place, gc, dc):
                op = core.CreateOperator("Sum", ["X1", "X2"],
                                                ["Y" if not in_place else "X1"])
                X1, X2 = inputs
                self.assertDeviceChecks(dc, op, [X1, X2], [0])
        """
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
        """
        Implements a standard numerical gradient checker for the operator
        in question.

        Useful for checking the consistency of the forward and
        backward implementations of operators.

        Usage example:

            @given(inputs=hu.tensors(n=2), in_place=st.booleans(), **hu.gcs)
            def test_sum(self, inputs, in_place, gc, dc):
                op = core.CreateOperator("Sum", ["X1", "X2"],
                                                ["Y" if not in_place else "X1"])
                X1, X2 = inputs
                self.assertGradientChecks(gc, op, [X1, X2], 0, [0])
        """
        gc = gradient_checker.GradientChecker(
            stepsize=stepsize,
            threshold=threshold,
            device_option=device_option,
            workspace_name=str(device_option),
            input_device_options=input_device_options,
        )
        res, grad, grad_estimated = gc.CheckSimple(
            op, inputs, outputs_to_check, outputs_with_grads,
            grad_ops=grad_ops,
            input_device_options=input_device_options
        )
        self.assertEqual(grad.shape, grad_estimated.shape)
        self.assertTrue(
            res,
            "Gradient check failed for input " + str(op.input[outputs_to_check])
        )

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
                np.testing.assert_allclose(
                    vals,
                    ref_vals,
                    atol=threshold,
                    rtol=threshold,
                    err_msg='Gradient {0} (x) is not matching the reference (y)'
                    .format(val_name),
                )
                if ref_indices is not None:
                    indices = workspace.FetchBlob(str(grad_names.indices))
                    np.testing.assert_allclose(indices, ref_indices,
                                               atol=1e-4, rtol=1e-4)

    def _assertInferTensorChecks(self, name, shapes, types, output,
                                 ensure_output_is_inferred=False):
        self.assertTrue(
            not ensure_output_is_inferred or (name in shapes),
            'Shape for {0} was not inferred'.format(name))

        if name not in shapes:
            # No inferred shape or type available
            return
        output = workspace.FetchBlob(name)
        if type(output) is np.ndarray:
            if output.dtype == np.dtype('float64'):
                correct_type = caffe2_pb2.TensorProto.DOUBLE
            elif output.dtype == np.dtype('float32'):
                correct_type = caffe2_pb2.TensorProto.FLOAT
            elif output.dtype == np.dtype('int32'):
                correct_type = caffe2_pb2.TensorProto.INT32
            elif output.dtype == np.dtype('int64'):
                correct_type = caffe2_pb2.TensorProto.INT64
            else:
                correct_type = "unknown {}".format(np.dtype)
        else:
            correct_type = str(type(output))
        try:
            np.testing.assert_array_equal(
                np.array(shapes[name]).astype(np.int32),
                np.array(output.shape).astype(np.int32),
                err_msg='Shape {} mismatch: {} vs. {}'.format(
                    name,
                    shapes[name],
                    output.shape))
            # BUG: Workspace blob type not being set correctly T16121392
            if correct_type != caffe2_pb2.TensorProto.INT32:
                return
            np.testing.assert_equal(
                types[name],
                correct_type,
                err_msg='Type {} mismatch: {} vs. {}'.format(
                    name, types[name], correct_type,
                )
            )
        except AssertionError as e:
            # Temporarily catch these assertion errors when validating
            # inferred shape and type info
            logging.warning(str(e))
            if os.getenv('CAFFE2_ASSERT_SHAPEINFERENCE') == '1':
                raise e

    def assertReferenceChecks(
        self,
        device_option,
        op,
        inputs,
        reference,
        input_device_options=None,
        threshold=1e-4,
        output_to_grad=None,
        grad_reference=None,
        atol=None,
        outputs_to_check=None,
        ensure_outputs_are_inferred=False,
    ):
        """
        This runs the reference Python function implementation
        (effectively calling `reference(*inputs)`, and compares that
        to the output of output, with an absolute/relative tolerance
        given by the `threshold` parameter.

        Useful for checking the implementation matches the Python
        (typically NumPy) implementation of the same functionality.

        Usage example:

            @given(X=hu.tensor(), inplace=st.booleans(), **hu.gcs)
            def test_softsign(self, X, inplace, gc, dc):
                op = core.CreateOperator(
                    "Softsign", ["X"], ["X" if inplace else "Y"])

                def softsign(X):
                    return (X / (1 + np.abs(X)),)

                self.assertReferenceChecks(gc, op, [X], softsign)
        """
        op = copy.deepcopy(op)
        op.device_option.CopyFrom(device_option)

        with temp_workspace():
            if (len(op.input) > len(inputs)):
                raise ValueError(
                    'must supply an input for each input on the op: %s vs %s' %
                    (op.input, inputs))
            _input_device_options = input_device_options or \
                core.InferOpBlobDevicesAsDict(op)[0]
            for (n, b) in zip(op.input, inputs):
                workspace.FeedBlob(
                    n,
                    b,
                    device_option=_input_device_options.get(n, device_option)
                )
            net = core.Net("opnet")
            net.Proto().op.extend([op])
            test_shape_inference = False
            try:
                (shapes, types) = workspace.InferShapesAndTypes([net])
                test_shape_inference = True
            except RuntimeError as e:
                # Temporarily catch runtime errors when inferring shape
                # and type info
                logging.warning(str(e))
                if os.getenv('CAFFE2_ASSERT_SHAPEINFERENCE') == '1':
                    raise e
            workspace.RunNetOnce(net)
            reference_outputs = reference(*inputs)
            if not (isinstance(reference_outputs, tuple) or
                    isinstance(reference_outputs, list)):
                raise RuntimeError(
                    "You are providing a wrong reference implementation. A "
                    "proper one should return a tuple/list of numpy arrays.")
            if not outputs_to_check:
                self.assertEqual(len(reference_outputs), len(op.output))
                outputs_to_check = list(range(len(op.output)))
            outs = []
            for (output_index, ref) in zip(outputs_to_check, reference_outputs):
                output_blob_name = op.output[output_index]
                output = workspace.FetchBlob(output_blob_name)
                if output.dtype.kind in ('S', 'O'):
                    np.testing.assert_array_equal(output, ref)
                else:
                    if atol is None:
                        atol = threshold
                    np.testing.assert_allclose(
                        output, ref, atol=atol, rtol=threshold,
                        err_msg=(
                            'Output {0} is not matching the reference'.format(
                                output_blob_name,
                            )),
                    )
                if test_shape_inference:
                    self._assertInferTensorChecks(
                        output_blob_name, shapes, types, output,
                        ensure_output_is_inferred=ensure_outputs_are_inferred)
                outs.append(output)
            if grad_reference is not None:
                assert output_to_grad is not None, \
                    "If grad_reference is set," \
                    "output_to_grad has to be set as well"

                with core.DeviceScope(device_option):
                    self._assertGradReferenceChecks(
                        op, inputs, reference_outputs,
                        output_to_grad, grad_reference,
                        threshold=threshold)

            return outs

    def assertValidationChecks(
            self,
            device_option,
            op,
            inputs,
            validator,
            input_device_options=None,
            as_kwargs=True,
            init_net=None,
    ):
        if as_kwargs:
            assert len(set(list(op.input) + list(op.output))) == \
                len(op.input) + len(op.output), \
                "in-place ops are not supported in as_kwargs mode"
        op = copy.deepcopy(op)
        op.device_option.CopyFrom(device_option)

        with temp_workspace():
            _input_device_options = input_device_options or \
                core.InferOpBlobDevicesAsDict(op)[0]
            for (n, b) in zip(op.input, inputs):
                workspace.FeedBlob(
                    n,
                    b,
                    device_option=_input_device_options.get(n, device_option)
                )
            if init_net:
                workspace.RunNetOnce(init_net)
            workspace.RunOperatorOnce(op)
            outputs = [workspace.FetchBlob(n) for n in op.output]
            if as_kwargs:
                validator(**dict(zip(
                    list(op.input) + list(op.output), inputs + outputs)))
            else:
                validator(inputs=inputs, outputs=outputs)

    def assertRunOpRaises(
        self,
        device_option,
        op,
        inputs,
        input_device_options=None,
        exception=(Exception,),
        regexp=None,
    ):
        op = copy.deepcopy(op)
        op.device_option.CopyFrom(device_option)

        with temp_workspace():
            _input_device_options = input_device_options or \
                core.InferOpBlobDevicesAsDict(op)[0]
            for (n, b) in zip(op.input, inputs):
                workspace.FeedBlob(
                    n,
                    b,
                    device_option=_input_device_options.get(n, device_option)
                )
            if regexp is None:
                self.assertRaises(exception, workspace.RunOperatorOnce, op)
            else:
                six.assertRaisesRegex(
                    self, exception, regexp, workspace.RunOperatorOnce, op)
