from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import hypothesis.strategies as st
from hypothesis import given, settings
import copy
import numpy as np
import math
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
from caffe2.python.transformations import optimizeForMKLDNN
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.ideep_test_util as mu


@unittest.skipIf(not workspace.C.use_mkldnn, "No MKLDNN support.")
class ConvFusionTest(hu.HypothesisTestCase):
    @given(size=st.integers(8, 20),
           input_channels=st.integers(1, 16),
           batch_size=st.integers(1, 3),
           use_bias=st.booleans(),
           group=st.integers(1, 1),
           **mu.gcs)
    def test_convolution_relu_fusion(self, size, input_channels,
                             batch_size, use_bias, group, gc, dc):
        ## Should reduce sparsity case ##
        '''                    
        conv1                                                  conv1 
        / \                                                     /  \ 
        |  \                                                    |   \  
        | conv2 (kernel=1,stride=1)                             |  conv2 (kernel=1,stride=1)
        |    \                                                  |     \ 
        |     \                                                 |      \ 
        |    conv3 (kernel=3,stride=1)                       max_pool  conv3 (kernel=3,stride=2)
        |       \                                               |       /
        |        \                                              |      /
        |      conv4 (kernel=1,stride=1)                        |     /
        \       /                                               |    /
         \     /                                                |   /
           sum                                               conv4_sum_fusion (kernel=1, stride=1)
          /    \                                                /    \ 
        conv5  conv6 (conv5, conv6:kernel = 1,stride = 2)    conv5  conv6 (conv5, conv6:kernel = 1,stride = 1)

         origin network                                         network after optimizeForMKLDNN     
        '''
        conv1 = core.CreateOperator(
            "Conv",
            ["X0", "w0", "b0"] if use_bias else ["X0", "w0"],
            ["Y0"],
            stride=1,
            pad=0,
            kernel=1,
            device_option=dc[0]
        )
        conv2 = core.CreateOperator(
            "Conv",
            ["Y0", "w0", "b0"] if use_bias else ["Y0", "w0"],
            ["Y1"],
            stride=1,
            pad=0,
            kernel=1,
            device_option=dc[0]
        )
        conv3 = core.CreateOperator(
            "Conv",
            ["Y1", "w1", "b1"] if use_bias else ["Y1", "w1"],
            ["Y2"],
            stride=1,
            pad=1,
            kernel=3,
            device_option=dc[0]
        )
        conv4 = core.CreateOperator(
            "Conv",
            ["Y2", "w0", "b0"] if use_bias else ["Y2", "w0"],
            ["Y3"],
            stride=1,
            pad=0,
            kernel=1,
            device_option=dc[0]
        )
        sum = core.CreateOperator(
            "Sum",
            ["Y3", "Y0"],
            ["Y_sum"],
            device_option=dc[0]
        )
        conv5 = core.CreateOperator(
            "Conv",
            ["Y_sum", "w0", "b0"] if use_bias else ["Y_sum", "w0"],
            ["Y4"],
            stride=2,
            pad=0,
            kernel=1,
            device_option=dc[0]
        )
        conv6 = core.CreateOperator(
            "Conv",
            ["Y_sum", "w0", "b0"] if use_bias else ["Y_sum", "w0"],
            ["Y5"],
            stride=2,
            pad=0,
            kernel=1,
            device_option=dc[0]
        )
        
        output_channels = input_channels 
        X = np.random.rand(
            batch_size, input_channels * group, size, size).astype(np.float32) - 0.5
        w = np.random.rand(
                output_channels * group, input_channels, 1, 1) \
            .astype(np.float32) - 0.5
        b = np.random.rand(output_channels * group).astype(np.float32) - 0.5

        w1 = np.random.rand(
                output_channels * group, input_channels, 3, 3) \
            .astype(np.float32) - 0.5
        b1 = np.random.rand(output_channels * group).astype(np.float32) - 0.5

        old_ws_name = workspace.CurrentWorkspace()
        workspace.SwitchWorkspace("_device_check_", True)
        workspace.FeedBlob('X0', X, dc[0])
        workspace.FeedBlob('w0', w, dc[0])
        workspace.FeedBlob('b0', b, dc[0])
        workspace.FeedBlob('w1', w1, dc[0])
        workspace.FeedBlob('b1', b1, dc[0])
        workspace.RunOperatorOnce(conv1)
        workspace.RunOperatorOnce(conv2)
        workspace.RunOperatorOnce(conv3)
        workspace.RunOperatorOnce(conv4)
        workspace.RunOperatorOnce(sum)
        workspace.RunOperatorOnce(conv5)
        workspace.RunOperatorOnce(conv6)
        Y4 = workspace.FetchBlob('Y4')
        Y5 = workspace.FetchBlob('Y5')
 
        workspace.ResetWorkspace()
        workspace.FeedBlob('X0', X, dc[1])
        workspace.FeedBlob('w0', w, dc[1])
        workspace.FeedBlob('b0', b, dc[1])
        workspace.FeedBlob('w1', w1, dc[1])
        workspace.FeedBlob('b1', b1, dc[1])
        old_net = caffe2_pb2.NetDef()
        conv1_old = caffe2_pb2.OperatorDef()
        conv1_old.CopyFrom(conv1)
        conv1_old.device_option.CopyFrom(dc[1])
        conv2_old = caffe2_pb2.OperatorDef()
        conv2_old.CopyFrom(conv2)
        conv2_old.device_option.CopyFrom(dc[1])
        conv3_old = caffe2_pb2.OperatorDef()
        conv3_old.CopyFrom(conv3)
        conv3_old.device_option.CopyFrom(dc[1])
        conv4_old = caffe2_pb2.OperatorDef()
        conv4_old.CopyFrom(conv4)
        conv4_old.device_option.CopyFrom(dc[1])
        sum_old = caffe2_pb2.OperatorDef()
        sum_old.CopyFrom(sum)
        sum_old.device_option.CopyFrom(dc[1])
        conv5_old = caffe2_pb2.OperatorDef()
        conv5_old.CopyFrom(conv5)
        conv5_old.device_option.CopyFrom(dc[1])
        conv6_old = caffe2_pb2.OperatorDef()
        conv6_old.CopyFrom(conv6)
        conv6_old.device_option.CopyFrom(dc[1])

        old_net.op.extend([conv1_old, conv2_old, conv3_old, conv4_old, sum_old, conv5_old, conv6_old])
        net = core.Net("net")
        net.Proto().CopyFrom(old_net)
        optimizeForMKLDNN(net)
        workspace.RunNetOnce(net.Proto())
        Y4_opt = workspace.FetchBlob(net.Proto().op[-2].output[0])
        Y5_opt = workspace.FetchBlob(net.Proto().op[-1].output[0])

        def get_arg(self, op, name):
            for i in range(len(op.arg)):
                if op.arg[i].name == name:
                    return op.arg[i]
            return None

        self.assertTrue(len(net.Proto().op) == 7)
        arg = get_arg(self, net.Proto().op[-1], "stride")
        self.assertTrue(arg.i == 1)
        arg = get_arg(self, net.Proto().op[-2], "stride")
        self.assertTrue(arg.i == 1)
        if not np.allclose(Y4, Y4_opt, atol=0.01, rtol=0.01):
            print(Y4.flatten())
            print(Y4_opt.flatten())
            print(np.max(np.abs(Y4 - Y4_opt)))
            self.assertTrue(False)
        if not np.allclose(Y5, Y5_opt, atol=0.01, rtol=0.01):
            print(Y5.flatten())
            print(Y5_opt.flatten())
            print(np.max(np.abs(Y5 - Y5_opt)))
            self.assertTrue(False)
        
        workspace.SwitchWorkspace(old_ws_name)

        ## Should not reduce sparsity case ##
        '''                    
                   conv1                                                   
                   /   \                                                   
                  /     \                                                 
                 /       \                                           
                /      conv2 
               /    (kernel=3,stride=2)                   
            conv1           \       
        (kernel=1,stride=2)  \                                      
              \              /                                  
               \            /                                  
                \          /
                 \        /
                    sum                                   
        '''
        conv1 = core.CreateOperator(
            "Conv",
            ["X0", "w0", "b0"] if use_bias else ["X0", "w0"],
            ["Y0"],
            stride=1,
            pad=0,
            kernel=1,
            device_option=dc[0]
        )
        conv2 = core.CreateOperator(
            "Conv",
            ["Y0", "w1", "b1"] if use_bias else ["Y0", "w1"],
            ["Y1"],
            stride=2,
            pad=1,
            kernel=3,
            device_option=dc[0]
        )
        conv3 = core.CreateOperator(
            "Conv",
            ["Y0", "w0", "b0"] if use_bias else ["Y0", "w0"],
            ["Y2"],
            stride=2,
            pad=0,
            kernel=1,
            device_option=dc[0]
        )
        sum = core.CreateOperator(
            "Sum",
            ["Y1", "Y2"],
            ["Y_sum"],
            device_option=dc[0]
        )
       
        output_channels = input_channels
        X = np.random.rand(
            batch_size, input_channels * group, size, size).astype(np.float32) - 0.5
        w = np.random.rand(
                output_channels * group, input_channels, 1, 1) \
            .astype(np.float32) - 0.5
        b = np.random.rand(output_channels * group).astype(np.float32) - 0.5
        w1 = np.random.rand(
                 output_channels * group, input_channels, 3, 3) \
            .astype(np.float32) - 0.5
        b1 = np.random.rand(output_channels * group).astype(np.float32) - 0.5

        old_ws_name = workspace.CurrentWorkspace()
        workspace.SwitchWorkspace("_device_check_", True)
        workspace.FeedBlob('X0', X, dc[0])
        workspace.FeedBlob('w0', w, dc[0])
        workspace.FeedBlob('b0', b, dc[0])
        workspace.FeedBlob('w1', w1, dc[0])
        workspace.FeedBlob('b1', b1, dc[0])
        workspace.RunOperatorOnce(conv1)
        workspace.RunOperatorOnce(conv2)
        workspace.RunOperatorOnce(conv3)
        workspace.RunOperatorOnce(sum)
        Y_sum = workspace.FetchBlob('Y_sum')

        workspace.ResetWorkspace()
        workspace.FeedBlob('X0', X, dc[1])
        workspace.FeedBlob('w0', w, dc[1])
        workspace.FeedBlob('b0', b, dc[1])
        workspace.FeedBlob('w1', w1, dc[1])
        workspace.FeedBlob('b1', b1, dc[1])
        old_net = caffe2_pb2.NetDef()
        conv1_old = caffe2_pb2.OperatorDef()
        conv1_old.CopyFrom(conv1)
        conv1_old.device_option.CopyFrom(dc[1])
        conv2_old = caffe2_pb2.OperatorDef()
        conv2_old.CopyFrom(conv2)
        conv2_old.device_option.CopyFrom(dc[1])
        conv3_old = caffe2_pb2.OperatorDef()
        conv3_old.CopyFrom(conv3)
        conv3_old.device_option.CopyFrom(dc[1])
        sum_old = caffe2_pb2.OperatorDef()
        sum_old.CopyFrom(sum)
        sum_old.device_option.CopyFrom(dc[1])

        old_net.op.extend([conv1_old, conv2_old, conv3_old, sum_old])
        net = core.Net("net")
        net.Proto().CopyFrom(old_net)
        optimizeForMKLDNN(net)
        workspace.RunNetOnce(net.Proto())
        Y_opt = workspace.FetchBlob(net.Proto().op[-1].output[0])

        self.assertTrue(len(net.Proto().op) == 3)
        arg = get_arg(self, net.Proto().op[-1], "stride")
        self.assertTrue(arg.i == 2)
        if not np.allclose(Y_sum, Y_opt, atol=0.01, rtol=0.01):
            print(Y_sum.flatten())
            print(Y_opt.flatten())
            print(np.max(np.abs(Y_sum - Y_opt)))
            self.assertTrue(False)

        workspace.SwitchWorkspace(old_ws_name)

if __name__ == "__main__":
    unittest.main()
