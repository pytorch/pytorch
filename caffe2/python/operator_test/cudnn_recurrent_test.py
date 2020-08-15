from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import model_helper, workspace, core, rnn_cell
from caffe2.proto import caffe2_pb2
from future.utils import viewitems
import numpy as np

import unittest


@unittest.skipIf(not workspace.has_gpu_support, "No gpu support.")
class TestLSTMs(unittest.TestCase):

    def testEqualToCudnn(self):
        with core.DeviceScope(core.DeviceOption(workspace.GpuDeviceType)):
            T = 8
            batch_size = 4
            input_dim = 8
            hidden_dim = 31

            workspace.FeedBlob(
                "seq_lengths",
                np.array([T] * batch_size, dtype=np.int32)
            )
            workspace.FeedBlob("target", np.zeros(
                [T, batch_size, hidden_dim], dtype=np.float32
            ))
            workspace.FeedBlob("hidden_init", np.zeros(
                [1, batch_size, hidden_dim], dtype=np.float32
            ))
            workspace.FeedBlob("cell_init", np.zeros(
                [1, batch_size, hidden_dim], dtype=np.float32
            ))

            own_model = model_helper.ModelHelper(name="own_lstm")

            input_shape = [T, batch_size, input_dim]
            cudnn_model = model_helper.ModelHelper(name="cudnn_lstm")
            input_blob = cudnn_model.param_init_net.UniformFill(
                [], "input", shape=input_shape)
            workspace.FeedBlob("CUDNN/hidden_init_cudnn", np.zeros(
                [1, batch_size, hidden_dim], dtype=np.float32
            ))
            workspace.FeedBlob("CUDNN/cell_init_cudnn", np.zeros(
                [1, batch_size, hidden_dim], dtype=np.float32
            ))

            cudnn_output, cudnn_last_hidden, cudnn_last_state, param_extract = rnn_cell.cudnn_LSTM(
                model=cudnn_model,
                input_blob=input_blob,
                initial_states=("hidden_init_cudnn", "cell_init_cudnn"),
                dim_in=input_dim,
                dim_out=hidden_dim,
                scope="CUDNN",
                return_params=True,
            )
            cudnn_loss = cudnn_model.AveragedLoss(
                cudnn_model.SquaredL2Distance(
                    [cudnn_output, "target"], "CUDNN/dist"
                ), "CUDNN/loss"
            )

            own_output, own_last_hidden, _, own_last_state, own_params = rnn_cell.LSTM(
                model=own_model,
                input_blob=input_blob,
                seq_lengths="seq_lengths",
                initial_states=("hidden_init", "cell_init"),
                dim_in=input_dim,
                dim_out=hidden_dim,
                scope="OWN",
                return_params=True,
            )
            own_loss = own_model.AveragedLoss(
                own_model.SquaredL2Distance([own_output, "target"], "OWN/dist"),
                "OWN/loss"
            )

            # Add gradients
            cudnn_model.AddGradientOperators([cudnn_loss])
            own_model.AddGradientOperators([own_loss])

            # Add parameter updates
            LR = cudnn_model.param_init_net.ConstantFill(
                [], shape=[1], value=0.01
            )
            ONE = cudnn_model.param_init_net.ConstantFill(
                [], shape=[1], value=1.0
            )
            for param in cudnn_model.GetParams():
                cudnn_model.WeightedSum(
                    [param, ONE, cudnn_model.param_to_grad[param], LR], param
                )
            for param in own_model.GetParams():
                own_model.WeightedSum(
                    [param, ONE, own_model.param_to_grad[param], LR], param
                )

            # Copy states over
            own_model.net.Copy(own_last_hidden, "hidden_init")
            own_model.net.Copy(own_last_state, "cell_init")
            cudnn_model.net.Copy(cudnn_last_hidden, "CUDNN/hidden_init_cudnn")
            cudnn_model.net.Copy(cudnn_last_state, "CUDNN/cell_init_cudnn")

            workspace.RunNetOnce(cudnn_model.param_init_net)
            workspace.CreateNet(cudnn_model.net)

            ##
            ##  CUDNN LSTM MODEL EXECUTION
            ##
            # Get initial values from CuDNN LSTM so we can feed them
            # to our own.
            (param_extract_net, param_extract_mapping) = param_extract
            workspace.RunNetOnce(param_extract_net)
            cudnn_lstm_params = {
                input_type: {
                    k: workspace.FetchBlob(v[0])
                    for k, v in viewitems(pars)
                }
                for input_type, pars in viewitems(param_extract_mapping)
            }

            # Run the model 3 times, so that some parameter updates are done
            workspace.RunNet(cudnn_model.net.Proto().name, 3)

            ##
            ## OWN LSTM MODEL EXECUTION
            ##
            # Map the cuDNN parameters to our own
            workspace.RunNetOnce(own_model.param_init_net)
            rnn_cell.InitFromLSTMParams(own_params, cudnn_lstm_params)

            # Run the model 3 times, so that some parameter updates are done
            workspace.CreateNet(own_model.net)
            workspace.RunNet(own_model.net.Proto().name, 3)

            ##
            ## COMPARE RESULTS
            ##
            # Then compare that final results after 3 runs are equal
            own_output_data = workspace.FetchBlob(own_output)
            own_last_hidden = workspace.FetchBlob(own_last_hidden)
            own_loss = workspace.FetchBlob(own_loss)

            cudnn_output_data = workspace.FetchBlob(cudnn_output)
            cudnn_last_hidden = workspace.FetchBlob(cudnn_last_hidden)
            cudnn_loss = workspace.FetchBlob(cudnn_loss)

            self.assertTrue(np.allclose(own_output_data, cudnn_output_data))
            self.assertTrue(np.allclose(own_last_hidden, cudnn_last_hidden))
            self.assertTrue(np.allclose(own_loss, cudnn_loss))
