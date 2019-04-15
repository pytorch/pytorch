## @package onnx
# Module caffe2.python.onnx.tests.conversion_test

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import six
import tempfile
import textwrap
import traceback
import unittest
import zipfile

from caffe2.proto import caffe2_pb2
from caffe2.python import brew, core
from caffe2.python.model_helper import ModelHelper
from click.testing import CliRunner
import numpy as np
from onnx import helper, ModelProto, TensorProto
from caffe2.python.onnx.helper import c2_native_run_net

from caffe2.python.onnx.bin.conversion import caffe2_to_onnx, onnx_to_caffe2
import caffe2.python.onnx.backend as c2
from caffe2.python.onnx.tests.test_utils import TestCase


class TestConversion(TestCase):
    def _run_command(self, cmd, *args, **kwargs):
        runner = CliRunner()
        result = runner.invoke(cmd, *args, **kwargs)
        self.assertEqual(result.exit_code, 0, textwrap.dedent('''
        Command exited with non-zero exit code:
        output: {}
        exception: {}
        exc_info: {}
        '''.format(result.output,
                   result.exception,
                   traceback.format_exception(*result.exc_info))))
        return result

    def test_caffe2_to_onnx(self):
        caffe2_net = tempfile.NamedTemporaryFile()
        caffe2_init_net = tempfile.NamedTemporaryFile()
        output = tempfile.NamedTemporaryFile()

        model = ModelHelper(name='caffe2-to-onnx-test')
        brew.relu(model, ["X"], "Y")
        caffe2_net.write(model.net.Proto().SerializeToString())
        caffe2_net.flush()

        init_model = ModelHelper(name='caffe2-to-onnx-init-test')
        init_model.net.GivenTensorFill([], 'X', shape=[2, 2],
                                       values=np.zeros((2, 2)).flatten().astype(float))
        caffe2_init_net.write(init_model.net.Proto().SerializeToString())
        caffe2_init_net.flush()

        self._run_command(
            caffe2_to_onnx, [
                caffe2_net.name,
                '--caffe2-init-net', caffe2_init_net.name,
                '--output', output.name,
            ],
            catch_exceptions=False,
        )

        onnx_model = ModelProto()
        onnx_model.ParseFromString(output.read())
        self.assertEqual(len(onnx_model.graph.node), 1)
        self.assertEqual(onnx_model.graph.node[0].op_type, 'Relu')
        self.assertEqual(len(onnx_model.graph.initializer), 1)
        self.assertEqual(onnx_model.graph.initializer[0].name, onnx_model.graph.input[0].name)

    def test_caffe2_to_onnx_value_info(self):
        caffe2_net = tempfile.NamedTemporaryFile()
        output = tempfile.NamedTemporaryFile()

        model = ModelHelper(name='caffe2-to-onnx-test')
        brew.relu(model, ["X"], "Y")
        caffe2_net.write(model.net.Proto().SerializeToString())
        caffe2_net.flush()

        args = [caffe2_net.name, '--output', output.name]
        six.assertRaisesRegex(self, Exception,
                              'value info',
                              self._run_command, caffe2_to_onnx, args)

        args.extend([
            '--value-info',
            json.dumps({
                'X': (TensorProto.FLOAT, (2, 2)),
            })])
        self._run_command(caffe2_to_onnx, args)

        onnx_model = ModelProto()
        onnx_model.ParseFromString(output.read())
        self.assertEqual(len(onnx_model.graph.node), 1)
        self.assertEqual(onnx_model.graph.node[0].op_type, 'Relu')
        self.assertEqual(len(onnx_model.graph.initializer), 0)

    def test_onnx_to_caffe2(self):
        onnx_model = tempfile.NamedTemporaryFile()
        output = tempfile.NamedTemporaryFile()
        init_net_output = tempfile.NamedTemporaryFile()

        node_def = helper.make_node(
            "Mul", ["X", "W"], ["Y"])
        graph_def = helper.make_graph(
            [node_def],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3)),
             helper.make_tensor_value_info("W", TensorProto.FLOAT, (3, 2))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (2, 2))],
            initializer=[helper.make_tensor("W",
                                            TensorProto.FLOAT,
                                            [3, 2],
                                            np.zeros((3, 2)).flatten().astype(float))])
        model_def = helper.make_model(graph_def, producer_name='onnx-to-caffe2-test')
        onnx_model.write(model_def.SerializeToString())
        onnx_model.flush()

        self._run_command(
            onnx_to_caffe2, [
                onnx_model.name,
                '--output', output.name,
                '--init-net-output', init_net_output.name,
            ])

        caffe2_net = caffe2_pb2.NetDef()
        caffe2_net.ParseFromString(output.read())
        self.assertEqual(len(caffe2_net.op), 1)
        self.assertEqual(caffe2_net.op[0].type, 'Mul')

        caffe2_init_net = caffe2_pb2.NetDef()
        caffe2_init_net.ParseFromString(init_net_output.read())
        self.assertEqual(len(caffe2_init_net.op), 1)
        self.assertEqual(set(sum([list(init_op.output)
                                  for init_op in caffe2_init_net.op], [])),
                         {'W'})

    def test_onnx_to_caffe2_zipfile(self):
        buf = tempfile.NamedTemporaryFile()
        onnx_model = zipfile.ZipFile(buf, 'w')

        node_def = helper.make_node(
            "MatMul", ["X", "W"], ["Y"])
        X = np.random.rand(2, 3).astype(np.float32)
        W = np.random.rand(3, 2).flatten().astype(np.float32)
        graph_def = helper.make_graph(
            [node_def],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3)),
             helper.make_tensor_value_info("W", TensorProto.FLOAT, (3, 2))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (2, 2))],
            initializer=[helper.make_tensor("W",
                                            TensorProto.FLOAT,
                                            [3, 2],
                                            b'__EXTERNAL',
                                            raw=True)])
        model_def = helper.make_model(graph_def, producer_name='onnx-to-caffe2-test')
        onnx_model.writestr('__MODEL_PROTO', model_def.SerializeToString())
        onnx_model.writestr('W', W.tobytes())
        onnx_model.close()

        W = W.reshape((3, 2))
        Y_expect = np.matmul(X, W)

        c2_model = c2.prepare_zip_archive(buf)
        Y = c2_model.run(X).Y
        np.testing.assert_allclose(Y, Y_expect)

    def _make_fake_if_op(self, true_nodes, false_nodes, output_types):
        true = helper.make_tensor("condition", TensorProto.BOOL, (), [True])
        true_graph = helper.make_graph(true_nodes, "true_graph", [], [
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, (2, 2)),
        ])
        false_graph = helper.make_graph(false_nodes, "false_graph", [], [
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, (2, 2)),
        ])
        if_inputs = ["condition"]
        if_outputs = [name for _, _, name in output_types]
        retval_nodes = [
            helper.make_node("Constant", [], ["condition"], value=true),
            helper.make_node("If", if_inputs, if_outputs, then_branch=true_graph,
                             else_branch=false_graph)
        ]
        return retval_nodes

    def test_onnx_to_caffe2_if(self):
        true_nodes = [helper.make_node(
            "MatMul", ["X", "W"], ["Y"])]
        false_nodes = [helper.make_node("Slice", ["X"], ["Y"], axes=[0, 1],
                                        starts=[0, 0], ends=[2, 2])]
        nodes = self._make_fake_if_op(true_nodes, false_nodes, [(TensorProto.FLOAT, (2, 2), "Y")])
        X = np.random.rand(2, 3).astype(np.float32)
        W = np.random.rand(3, 2).flatten().astype(np.float32)
        graph_def = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3)),
             helper.make_tensor_value_info("W", TensorProto.FLOAT, (3, 2))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (2, 2))],
            initializer=[helper.make_tensor("W",
                                            TensorProto.FLOAT,
                                            [3, 2],
                                            W.tolist())]
        )
        onnx_id = helper.make_opsetid("", 9)
        model_def = helper.make_model(graph_def, producer_name='onnx-to-caffe2-test',
                                      opset_imports=[onnx_id])

        p = c2.prepare(model_def)
        Y = np.matmul(X, W.reshape(3, 2))
        out = p.run(X)
        np.testing.assert_allclose(out.Y, Y)

    # input_types and output_types are lists of triples of (name, type, shape)
    def _make_fake_loop_op(self, body_nodes, input_types, output_types):
        ten = helper.make_tensor("trip_count_value", TensorProto.INT64, (1,), [10])
        true = helper.make_tensor("condition", TensorProto.BOOL, (1,), [True])
        # lcd is a dummy loop-carried dependency that only exists because
        # right now the schema checker is broken and assumes a variadic
        # input needs at least one value.
        graph_inputs = [helper.make_tensor_value_info("i", TensorProto.INT32, ()),
                        helper.make_tensor_value_info("cond", TensorProto.BOOL, ())]
        for type, shape, name in input_types:
            graph_inputs.append(helper.make_tensor_value_info("_" + name, type, shape))
        graph_outputs = [helper.make_tensor_value_info("cond", TensorProto.BOOL, ())]
        for type, shape, name in output_types:
            graph_outputs.append(helper.make_tensor_value_info("_" + name, type, shape))
        body_graph = helper.make_graph(body_nodes, "body_graph", graph_inputs,
                                       graph_outputs)
        loop_inputs = ["trip_count", "condition"]
        loop_inputs.extend([name for _, _, name in input_types])
        loop_outputs = [name for _, _, name in output_types]
        retval_nodes = [
            helper.make_node("Constant", [], ["trip_count"], value=ten),
            helper.make_node("Constant", [], ["condition"], value=true),
            helper.make_node("Loop", loop_inputs, loop_outputs, body=body_graph)
        ]
        return retval_nodes

    def test_onnx_to_caffe2_loop(self):
        body_nodes = [helper.make_node(
            "MatMul", ["_X", "W"], ["_Y"])]
        nodes = self._make_fake_loop_op(body_nodes,
                                        [(TensorProto.FLOAT, (2, 2), "X")],
                                        [(TensorProto.FLOAT, (2, 2), "Y")])
        X = np.random.rand(2, 2).astype(np.float32)
        W = np.random.rand(2, 2).flatten().astype(np.float32)
        graph_def = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 2)),
             helper.make_tensor_value_info("W", TensorProto.FLOAT, (2, 2))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (2, 2))],
            initializer=[helper.make_tensor("W",
                                            TensorProto.FLOAT,
                                            [2, 2],
                                            W.tolist())]
        )
        model_def = helper.make_model(graph_def, producer_name='onnx-to-caffe2-test')
        Y = X
        for _ in range(10):
            Y = np.matmul(Y, W.reshape(2, 2))
        p = c2.prepare(model_def)
        out = p.run(X)
        np.testing.assert_allclose(out.Y, Y)

    # TODO investigate why this is failing after changing Reshape
    # operator from taking the new shape as attribute to as input
    @unittest.skip('Start failing after Reshape op change')
    def test_convert_end2end(self):
        predict_net_f = tempfile.NamedTemporaryFile()
        init_net_f = tempfile.NamedTemporaryFile()
        onnx_model_f = tempfile.NamedTemporaryFile()

        x = 'X'
        w = 'W'
        b = 'b'
        y = 'Y'

        predict_net = caffe2_pb2.NetDef()
        predict_net.name = 'test-convert-end2end'
        predict_net.external_input[:] = [x, w, b]
        predict_net.external_output[:] = [y]
        predict_net.op.extend([
            core.CreateOperator(
                'FC',
                inputs=[x, w, b],
                outputs=[y],
                axis=2,
            ),
        ])
        predict_net_f.write(predict_net.SerializeToString())
        predict_net_f.flush()

        init_net = caffe2_pb2.NetDef()
        init_net.name = 'test-convert-end2end-init'
        init_net.external_output[:] = [w, b]
        x_val = np.random.randn(1, 3, 2).astype(np.float32)
        w_val = np.random.randn(4, 2).astype(np.float32)
        b_val = np.random.randn(4).astype(np.float32)
        init_net.op.extend([
            core.CreateOperator(
                'GivenTensorFill',
                [],
                [w],
                values=w_val,
                shape=w_val.shape,
            ),
            core.CreateOperator(
                'GivenTensorFill',
                [],
                [b],
                values=b_val,
                shape=b_val.shape,
            ),
        ])
        init_net_f.write(init_net.SerializeToString())
        init_net_f.flush()

        y_val = np.matmul(x_val, w_val.transpose()) + b_val
        for _ in range(5):
            self._run_command(
                caffe2_to_onnx, [
                    predict_net_f.name,
                    '--caffe2-init-net', init_net_f.name,
                    '--output', onnx_model_f.name,
                    '--value-info',
                    json.dumps({
                        x: (TensorProto.FLOAT, (1, 3, 2)),
                    }),
                ],
                catch_exceptions=False,
            )

            onnx_model_f.seek(0)
            onnx_model = ModelProto()
            onnx_model.ParseFromString(onnx_model_f.read())
            np.testing.assert_almost_equal(
                c2.run_model(
                    onnx_model, {onnx_model.graph.input[0].name: x_val}),
                [y_val])

            self._run_command(
                onnx_to_caffe2, [
                    onnx_model_f.name,
                    '--output', predict_net_f.name,
                    '--init-net-output', init_net_f.name,
                ])
            predict_net_f.seek(0)
            predict_net = caffe2_pb2.NetDef()
            predict_net.ParseFromString(predict_net_f.read())
            init_net_f.seek(0)
            init_net = caffe2_pb2.NetDef()
            init_net.ParseFromString(init_net_f.read())
            x = predict_net.external_input[0]
            np.testing.assert_almost_equal(c2_native_run_net(init_net=init_net,
                                                             predict_net=predict_net,
                                                             inputs={x: x_val})[1],
                                           [y_val])
