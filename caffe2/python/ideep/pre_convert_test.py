




import unittest
import hypothesis.strategies as st
from hypothesis import given
import numpy as np
from caffe2.proto import caffe2_pb2
from caffe2.python import (
    brew,
    core,
    model_helper,
    workspace,
)
from caffe2.python.transformations import optimizeForMKLDNN
import caffe2.python.hypothesis_test_util as hu


@unittest.skipIf(not workspace.C.use_mkldnn, "No MKLDNN support.")
class PreConvertTest(hu.HypothesisTestCase):
    @given(input_channels=st.integers(15, 16),
           batch_size=st.integers(1, 3))
    def test_preConvert(self, input_channels, batch_size):
        def AddModel(model, data):
            conv1 = brew.conv(model, data, 'conv1', dim_in=input_channels,
                              dim_out=10, kernel=3, stride=1, pad=1, training_mode=1)
            deconv1 = brew.conv_transpose(model, conv1, 'deconv1', dim_in=10, dim_out=10,
                                          kernel=2, stride=2, pad=0, training_mode=1)
            fc1 = brew.fc(model, deconv1, 'fc1', dim_in=10 * 56 * 56, dim_out=3)
            softmax = brew.softmax(model, fc1, 'softmax')

            return softmax

        def AddTrainingOperators(model, softmax, label):
            """Adds training operators to the model."""
            # Compute cross entropy between softmax scores and labels
            xent = model.LabelCrossEntropy([softmax, label], 'xent')
            # Compute the expected loss
            loss = model.AveragedLoss(xent, "loss")
            # Use the average loss we just computed to add gradient operators to the model
            model.AddGradientOperators([loss])

        arg_scope = {"order": "NCHW", 'no_bias': False}
        # Create the model helper for the train model
        device_opt = core.DeviceOption(caffe2_pb2.IDEEP, 0)
        with core.DeviceScope(device_opt):
            train_model = model_helper.ModelHelper(name="test_train", arg_scope=arg_scope)
            # Add the model definition (fc layers, conv layers, softmax, etc.)
            softmax = AddModel(train_model, "X")
            AddTrainingOperators(train_model, softmax, "label")

            X = np.random.rand(
                batch_size, input_channels, 28, 28).astype(np.float32) - 0.5
            label = np.random.randint(3, size=batch_size).astype(np.int32)
            blob_dict = {}
            output_dict = {}
            output_dict_cosim = {}
            old_ws_name = workspace.CurrentWorkspace()
            workspace.FeedBlob('X', X)
            workspace.FeedBlob('label', label)
            workspace.RunNetOnce(train_model.param_init_net)
            for op in train_model.net.Proto().op:
                if op.type == "Softmax":
                    break
                for j in range(1, len(op.input)):
                    blob_dict[op.input[j]] = workspace.FetchBlob(op.input[j])

            workspace.CreateNet(train_model.net, overwrite=True)
            optimizeForMKLDNN(train_model.net, training_mode=True)
            workspace.RunNet(train_model.net)
            for op in train_model.net.Proto().op:
                for blob in op.output:
                    output_dict[blob] = workspace.FetchBlob(blob)

            workspace.SwitchWorkspace("_device_check_", True)
            workspace.FeedBlob('X', X)
            workspace.FeedBlob('label', label)
            for blob in blob_dict.keys():
                workspace.FeedBlob(blob, blob_dict[blob])
            workspace.CreateNet(train_model.net, overwrite=True)
            workspace.RunNet(train_model.net)
            for blob in output_dict.keys():
                output_dict_cosim[blob] = workspace.FetchBlob(blob)

            for blob in output_dict.keys():
                if not np.allclose(output_dict[blob], output_dict_cosim[blob], atol=0.001, rtol=0.0001):
                    print("blob {} error".format(blob))
                    print(np.max(np.abs(output_dict[blob] - output_dict_cosim[blob])))
                    self.assertTrue(False)

            workspace.ResetWorkspace()
            workspace.SwitchWorkspace(old_ws_name)

if __name__ == "__main__":
    unittest.main()
