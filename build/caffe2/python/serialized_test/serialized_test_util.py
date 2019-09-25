from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
from caffe2.proto import caffe2_pb2
from caffe2.python import gradient_checker
import caffe2.python.hypothesis_test_util as hu
from caffe2.python.serialized_test import coverage
import hypothesis as hy
import inspect
import numpy as np
import os
import shutil
import sys
import tempfile
import threading
from zipfile import ZipFile

operator_test_type = 'operator_test'
TOP_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_SUFFIX = 'data'
DATA_DIR = os.path.join(TOP_DIR, DATA_SUFFIX)
_output_context = threading.local()


def given(*given_args, **given_kwargs):
    def wrapper(f):
        hyp_func = hy.given(*given_args, **given_kwargs)(f)
        fixed_seed_func = hy.seed(0)(hy.settings(max_examples=1)(hy.given(
            *given_args, **given_kwargs)(f)))

        def func(self, *args, **kwargs):
            self.should_serialize = True
            fixed_seed_func(self, *args, **kwargs)
            self.should_serialize = False
            hyp_func(self, *args, **kwargs)
        return func
    return wrapper


def _getGradientOrNone(op_proto):
    try:
        grad_ops, _ = gradient_checker.getGradientForOp(op_proto)
        return grad_ops
    except Exception:
        return []


# necessary to support converting jagged lists into numpy arrays
def _transformList(l):
    ret = np.empty(len(l), dtype=np.object)
    for (i, arr) in enumerate(l):
        ret[i] = arr
    return ret


def _prepare_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


class SerializedTestCase(hu.HypothesisTestCase):

    should_serialize = False

    def get_output_dir(self):
        output_dir_arg = getattr(_output_context, 'output_dir', DATA_DIR)
        output_dir = os.path.join(
            output_dir_arg, operator_test_type)

        if os.path.exists(output_dir):
            return output_dir

        # fall back to pwd
        cwd = os.getcwd()
        serialized_util_module_components = __name__.split('.')
        serialized_util_module_components.pop()
        serialized_dir = '/'.join(serialized_util_module_components)
        output_dir_fallback = os.path.join(cwd, serialized_dir, DATA_SUFFIX)
        output_dir = os.path.join(
            output_dir_fallback,
            operator_test_type)

        return output_dir

    def get_output_filename(self):
        class_path = inspect.getfile(self.__class__)
        file_name_components = os.path.basename(class_path).split('.')
        test_file = file_name_components[0]

        function_name_components = self.id().split('.')
        test_function = function_name_components[-1]

        return test_file + '.' + test_function

    def serialize_test(self, inputs, outputs, grad_ops, op, device_option):
        output_dir = self.get_output_dir()
        test_name = self.get_output_filename()
        full_dir = os.path.join(output_dir, test_name)
        _prepare_dir(full_dir)

        inputs = _transformList(inputs)
        outputs = _transformList(outputs)
        device_type = int(device_option.device_type)

        op_path = os.path.join(full_dir, 'op.pb')
        grad_paths = []
        inout_path = os.path.join(full_dir, 'inout')

        with open(op_path, 'wb') as f:
            f.write(op.SerializeToString())
        for (i, grad) in enumerate(grad_ops):
            grad_path = os.path.join(full_dir, 'grad_{}.pb'.format(i))
            grad_paths.append(grad_path)
            with open(grad_path, 'wb') as f:
                f.write(grad.SerializeToString())

        np.savez_compressed(
            inout_path,
            inputs=inputs,
            outputs=outputs,
            device_type=device_type)

        with ZipFile(os.path.join(output_dir, test_name + '.zip'), 'w') as z:
            z.write(op_path, 'op.pb')
            z.write(inout_path + '.npz', 'inout.npz')
            for path in grad_paths:
                z.write(path, os.path.basename(path))

        shutil.rmtree(full_dir)

    def compare_test(self, inputs, outputs, grad_ops, atol=1e-7, rtol=1e-7):

        def parse_proto(x):
            proto = caffe2_pb2.OperatorDef()
            proto.ParseFromString(x)
            return proto

        source_dir = self.get_output_dir()
        test_name = self.get_output_filename()
        temp_dir = tempfile.mkdtemp()
        with ZipFile(os.path.join(source_dir, test_name + '.zip')) as z:
            z.extractall(temp_dir)

        op_path = os.path.join(temp_dir, 'op.pb')
        inout_path = os.path.join(temp_dir, 'inout.npz')

        # load serialized input and output
        loaded = np.load(inout_path, encoding='bytes', allow_pickle=True)
        loaded_inputs = loaded['inputs'].tolist()
        inputs_equal = True
        for (x, y) in zip(inputs, loaded_inputs):
            if not np.array_equal(x, y):
                inputs_equal = False
        loaded_outputs = loaded['outputs'].tolist()

        # if inputs are not the same, run serialized input through serialized op
        if not inputs_equal:
            # load operator
            with open(op_path, 'rb') as f:
                loaded_op = f.read()

            op_proto = parse_proto(loaded_op)
            device_type = loaded['device_type']
            device_option = caffe2_pb2.DeviceOption(
                device_type=int(device_type))

            outputs = hu.runOpOnInput(device_option, op_proto, loaded_inputs)
            grad_ops = _getGradientOrNone(op_proto)

        # assert outputs are equal
        for (x, y) in zip(outputs, loaded_outputs):
            np.testing.assert_allclose(x, y, atol=atol, rtol=rtol)

        # assert gradient op is equal
        for i in range(len(grad_ops)):
            grad_path = os.path.join(temp_dir, 'grad_{}.pb'.format(i))
            with open(grad_path, 'rb') as f:
                loaded_grad = f.read()
            grad_proto = parse_proto(loaded_grad)
            self._assertSameOps(grad_proto, grad_ops[i])

        shutil.rmtree(temp_dir)

    def _assertSameOps(self, op1, op2):
        op1_ = caffe2_pb2.OperatorDef()
        op1_.CopyFrom(op1)
        op1_.arg.sort(key=lambda arg: arg.name)

        op2_ = caffe2_pb2.OperatorDef()
        op2_.CopyFrom(op2)
        op2_.arg.sort(key=lambda arg: arg.name)

        self.assertEqual(op1_, op2_)

    def assertSerializedOperatorChecks(
            self,
            inputs,
            outputs,
            gradient_operator,
            op,
            device_option,
            atol=1e-7,
            rtol=1e-7,
    ):
        if self.should_serialize:
            if getattr(_output_context, 'should_generate_output', False):
                self.serialize_test(
                    inputs, outputs, gradient_operator, op, device_option)
                if not getattr(_output_context, 'disable_gen_coverage', False):
                    coverage.gen_serialized_test_coverage(
                        self.get_output_dir(), TOP_DIR)
            else:
                self.compare_test(
                    inputs, outputs, gradient_operator, atol, rtol)

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
    ):
        outs = super(SerializedTestCase, self).assertReferenceChecks(
            device_option,
            op,
            inputs,
            reference,
            input_device_options,
            threshold,
            output_to_grad,
            grad_reference,
            atol,
            outputs_to_check,
        )
        if not getattr(_output_context, 'disable_serialized_check', False):
            grad_ops = _getGradientOrNone(op)
            rtol = threshold
            if atol is None:
                atol = threshold
            self.assertSerializedOperatorChecks(
                inputs,
                outs,
                grad_ops,
                op,
                device_option,
                atol,
                rtol,
            )


def testWithArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-G', '--generate-serialized', action='store_true', dest='generate',
        help='generate output files (default=false, compares to current files)')
    parser.add_argument(
        '-O', '--output', default=DATA_DIR,
        help='output directory (default: %(default)s)')
    parser.add_argument(
        '-D', '--disable-serialized_check', action='store_true', dest='disable',
        help='disable checking serialized tests')
    parser.add_argument(
        '-C', '--disable-gen-coverage', action='store_true',
        dest='disable_coverage',
        help='disable generating coverage markdown file')
    parser.add_argument('unittest_args', nargs='*')
    args = parser.parse_args()
    sys.argv[1:] = args.unittest_args
    _output_context.__setattr__('should_generate_output', args.generate)
    _output_context.__setattr__('output_dir', args.output)
    _output_context.__setattr__('disable_serialized_check', args.disable)
    _output_context.__setattr__('disable_gen_coverage', args.disable_coverage)

    import unittest
    unittest.main()
