## @package onnx
# Module caffe2.python.onnx.tests.ssa_test






import copy
import numpy as np
from caffe2.proto import caffe2_pb2
from caffe2.python import core
from onnx import TensorProto

import caffe2.python.onnx.frontend as c2_onnx
from caffe2.python.onnx.helper import c2_native_run_net
from caffe2.python.onnx.tests.test_utils import TestCase


class TestFrontendSSAConversion(TestCase):
    def test_ssa(self):
        X = np.random.randn(4, 2).astype(np.float32)
        W = np.random.randn(3, 2).astype(np.float32)
        b = np.random.randn(3).astype(np.float32)
        s = np.random.randn(1).astype(np.float32)
        np_result = X.dot(W.transpose()) + b + s

        net = caffe2_pb2.NetDef()
        net.name = 'test-ssa'
        net.external_input[:] = ['W', 'X', 'b', 's']
        net.op.extend([
            core.CreateOperator(
                'FC',
                ['X', 'W', 'b'],
                ['Y']
            ),
            core.CreateOperator(
                'Add',
                ['Y', 's'],
                ['Y'],
                broadcast=True,
            )
        ])
        net.external_output[:] = ['Y']

        init_net = caffe2_pb2.NetDef()
        init_net.name = 'test-ssa-init'
        init_net.op.extend([
            core.CreateOperator(
                'GivenTensorFill',
                [],
                ['W'],
                values=W,
                shape=W.shape,
            ),
            core.CreateOperator(
                'GivenTensorFill',
                [],
                ['b'],
                values=b,
                shape=b.shape,
            ),
            core.CreateOperator(
                'GivenTensorFill',
                [],
                ['s'],
                values=s,
                shape=s.shape,
            )
        ])
        init_net.external_output[:] = ['W', 'b', 's']

        _, orig_output = c2_native_run_net(
            predict_net=net,
            init_net=init_net,
            inputs=[X])

        value_info = {'X': (TensorProto.FLOAT, X.shape)}
        c2_onnx.Caffe2Frontend._ssa_rewrite(
            net,
            init_net,
            value_info)

        self.assertEqual(net.external_input, ['W', 'X', 'b', 's'])
        self.assertEqual(net.op[0].input, ['X', 'W', 'b'])
        self.assertEqual(net.op[0].output, ['Y_1'])
        self.assertEqual(net.op[1].input, ['Y_1', 's'])
        self.assertEqual(net.op[1].output, ['Y_2'])
        self.assertEqual(net.external_output, ['Y_2'])

        self.assertEqual(init_net.external_input, [])
        self.assertEqual(init_net.op[0].input, [])
        self.assertEqual(init_net.op[0].output, ['W'])
        self.assertEqual(init_net.op[1].input, [])
        self.assertEqual(init_net.op[1].output, ['b'])
        self.assertEqual(init_net.op[2].input, [])
        self.assertEqual(init_net.op[2].output, ['s'])
        self.assertEqual(init_net.external_output, ['W', 'b', 's'])
        self.assertEqual(value_info, {'X': (TensorProto.FLOAT, X.shape)})

        _, ssa_output = c2_native_run_net(
            predict_net=net,
            init_net=init_net,
            inputs=[X])

        self.assertSameOutputs(ssa_output, orig_output)
        self.assertSameOutputs(ssa_output, [np_result])

    def test_idempotence(self):
        net = caffe2_pb2.NetDef()
        net.name = 'test-idempotence'
        net.external_input[:] = ['W', 'X', 'b', 's']
        net.op.extend([
            core.CreateOperator(
                'FC',
                ['X', 'W', 'b'],
                ['Y']
            ),
            core.CreateOperator(
                'Add',
                ['Y', 's'],
                ['Z'],
                broadcast=True,
            )
        ])
        net.external_output[:] = ['Z']

        value_info = {'X': (TensorProto.FLOAT, [4, 2])}
        net_copy = copy.deepcopy(net)
        c2_onnx.Caffe2Frontend._ssa_rewrite(
            net_copy,
            None,
            value_info)
        self.assertEqual(net, net_copy)
