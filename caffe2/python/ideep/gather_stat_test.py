from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np
import hypothesis.strategies as st
from hypothesis import given, settings
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace, utils, stat
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.ideep_test_util as mu

def add_tensor(net, name, blob):
    ''' Create an operator to store the tensor 'blob',
        run the operator to put the blob to workspace.
        uint8 is stored as an array of string with one element.
    '''
    kTypeNameMapper = {
        np.dtype('float32'): "GivenTensorFill",
        np.dtype('int32'): "GivenTensorIntFill",
        np.dtype('int64'): "GivenTensorInt64Fill",
        np.dtype('uint8'): "GivenTensorStringFill",
    }

    shape = blob.shape
    values = blob
    # pass array of uint8 as a string to save storage
    # storing uint8_t has a large overhead for now
    if blob.dtype == np.dtype('uint8'):
        shape = [1]
        values = [str(blob.data)]

    op = core.CreateOperator(
        kTypeNameMapper[blob.dtype],
        [], [name],
        arg=[
            utils.MakeArgument("shape", shape),
            utils.MakeArgument("values", values),
        ]
    )
    net.op.extend([op])


@unittest.skipIf(not workspace.C.use_mkldnn, "No MKLDNN support.")
class GatherStatTest(hu.HypothesisTestCase):
    @given(stride=st.integers(1, 3),
           pad=st.integers(0, 3),
           kernel=st.integers(3, 5),
           size=st.integers(8, 20),
           input_channels=st.integers(1, 16),
           output_channels=st.integers(1, 16),
           batch_size=st.integers(1, 3),
           use_bias=st.booleans(),
           group=st.integers(1, 1),
           **mu.gcs)
    def test_gather_stat_info(self, stride, pad, kernel, size,
                             input_channels, output_channels,
                             batch_size, use_bias, group, gc, dc):
        conv = core.CreateOperator(
            "Conv",
            ["X0", "w0", "b0"] if use_bias else ["X0", "w0"],
            ["Y0"],
            stride=stride,
            pad=pad,
            kernel=kernel,
            group=group,
            device_option=dc[0]
        )
        relu = core.CreateOperator(
            "Relu",
            ["Y0"],
            ["Y0"],
            device_option=dc[0]
        )

        X0 = np.random.rand(
            batch_size, input_channels * group, size, size).astype(np.float32) - 0.5
        X1 = np.random.rand(
            batch_size, input_channels * group, size, size).astype(np.float32) - 0.5
        X2 = np.random.rand(
            batch_size, input_channels * group, size, size).astype(np.float32) - 0.5
        X3 = np.random.rand(
            batch_size, input_channels * group, size, size).astype(np.float32) - 0.5
        X_list = [X0, X1, X2, X3]

        w = np.random.rand(
                output_channels * group, input_channels, kernel, kernel) \
            .astype(np.float32) - 0.5
        b = np.random.rand(output_channels * group).astype(np.float32) - 0.5

        abs_max = np.array([0.0, 0.0])
        old_ws_name = workspace.CurrentWorkspace()
        workspace.SwitchWorkspace("_gather_checker_", True)
        for X in X_list:
            workspace.FeedBlob('X0', X, dc[0])
            workspace.FeedBlob('w0', w, dc[0])
            workspace.FeedBlob('b0', b, dc[0])
            workspace.RunOperatorOnce(conv)
            Y0 = workspace.FetchBlob('Y0')
            abs_max[0] = max(np.absolute(Y0).max(), abs_max[0])
            workspace.RunOperatorOnce(relu)
            Y0 = workspace.FetchBlob('Y0')
            abs_max[1] = max(np.absolute(Y0).max(), abs_max[1])

        workspace.ResetWorkspace()
        predict_net = caffe2_pb2.NetDef()
        conv_old = caffe2_pb2.OperatorDef()
        conv_old.CopyFrom(conv)
        conv_old.device_option.CopyFrom(dc[1])
        relu_old = caffe2_pb2.OperatorDef()
        relu_old.CopyFrom(relu)
        relu_old.device_option.CopyFrom(dc[1])
        predict_net.op.extend([conv_old, relu_old])

        init_net = caffe2_pb2.NetDef()
        add_tensor(init_net, 'w0', w)
        add_tensor(init_net, 'b0', b)

        def data_gen():
            for X in X_list:
                yield X

        predict = stat.GatherStatInfo(predict_net, init_net, data_gen, 'X0', dc[1],
                ['SpatialBN', 'Softmax', 'FC', 'AveragePool', 'MaxPool'],
                ['Softmax', 'FC', 'AveragePool', 'MaxPool'])
        abs_max_stat = np.array([0.0, 0.0])
        for arg in predict.op[1].arg:
            if arg.name == 'absmax_input_0':
                self.assertTrue(len(arg.floats) == 1)
                abs_max_stat[0] = arg.floats[0]
            if arg.name == 'absmax_output_0':
                self.assertTrue(len(arg.floats) == 1)
                abs_max_stat[1] = arg.floats[0]

        if not np.allclose(abs_max, abs_max_stat, atol=0.01, rtol=0.01):
            print(abs_max_stat.flatten())
            print(abs_max.flatten())
            print(np.max(np.abs(abs_max_stat - abs_max)))
            self.assertTrue(False)

        workspace.SwitchWorkspace(old_ws_name)


if __name__ == "__main__":
    unittest.main()
