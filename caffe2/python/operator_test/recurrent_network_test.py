




from caffe2.python import recurrent, workspace
from caffe2.python.model_helper import ModelHelper
from hypothesis import given, settings
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import hypothesis.strategies as st
import numpy as np

import os
import unittest

class RecurrentNetworkTest(serial.SerializedTestCase):
    @given(T=st.integers(1, 4),
           n=st.integers(1, 5),
           d=st.integers(1, 5))
    @settings(deadline=10000)
    def test_sum_mul(self, T, n, d):
        model = ModelHelper(name='external')

        input_blob, initial_input_blob = model.net.AddExternalInputs(
            'input', 'initial_input')

        step = ModelHelper(name='step', param_model=model)
        input_t, output_t_prev = step.net.AddExternalInput(
            'input_t', 'output_t_prev')
        output_t_internal = step.net.Sum([input_t, output_t_prev])
        output_t = step.net.Mul([input_t, output_t_internal])
        step.net.AddExternalOutput(output_t)

        self.simple_rnn(T, n, d, model, step, input_t, output_t, output_t_prev,
                        input_blob, initial_input_blob)

    @given(T=st.integers(1, 4),
           n=st.integers(1, 5),
           d=st.integers(1, 5))
    @settings(deadline=10000)
    def test_mul(self, T, n, d):
        model = ModelHelper(name='external')

        input_blob, initial_input_blob = model.net.AddExternalInputs(
            'input', 'initial_input')

        step = ModelHelper(name='step', param_model=model)
        input_t, output_t_prev = step.net.AddExternalInput(
            'input_t', 'output_t_prev')
        output_t = step.net.Mul([input_t, output_t_prev])
        step.net.AddExternalOutput(output_t)

        self.simple_rnn(T, n, d, model, step, input_t, output_t, output_t_prev,
                        input_blob, initial_input_blob)

    @given(T=st.integers(1, 4),
           n=st.integers(1, 5),
           d=st.integers(1, 5))
    def test_extract(self, T, n, d):
        model = ModelHelper(name='external')
        workspace.ResetWorkspace()

        input_blob, initial_input_blob = model.net.AddExternalInputs(
            'input', 'initial_input')

        step = ModelHelper(name='step', param_model=model)
        input_t, output_t_prev = step.net.AddExternalInput(
            'input_t', 'output_t_prev')
        output_t = step.net.Mul([input_t, output_t_prev])
        step.net.AddExternalOutput(output_t)

        inputs = np.random.randn(T, n, d).astype(np.float32)
        initial_input = np.random.randn(1, n, d).astype(np.float32)
        recurrent.recurrent_net(
            net=model.net,
            cell_net=step.net,
            inputs=[(input_t, input_blob)],
            initial_cell_inputs=[(output_t_prev, initial_input_blob)],
            links={output_t_prev: output_t},
            scope="test_rnn_sum_mull",
        )

        workspace.blobs[input_blob] = inputs
        workspace.blobs[initial_input_blob] = initial_input

        workspace.RunNetOnce(model.param_init_net)
        workspace.CreateNet(model.net)

        prefix = "extractTest"

        workspace.RunNet(model.net.Proto().name, T)
        retrieved_blobs = recurrent.retrieve_step_blobs(
            model.net, prefix
        )

        # needed for python3.6, which returns bytearrays instead of str
        retrieved_blobs = [x.decode() for x in retrieved_blobs]

        for i in range(T):
            blob_name = prefix + "_" + "input_t" + str(i)
            self.assertTrue(
                blob_name in retrieved_blobs,
                "blob extraction failed on timestep {}\
                    . \n\n Extracted Blobs: {} \n\n Looking for {}\
                    .".format(i, retrieved_blobs, blob_name)
            )

    def simple_rnn(self, T, n, d, model, step, input_t, output_t, output_t_prev,
                   input_blob, initial_input_blob):

        input = np.random.randn(T, n, d).astype(np.float32)
        initial_input = np.random.randn(1, n, d).astype(np.float32)
        print(locals())
        recurrent.recurrent_net(
            net=model.net,
            cell_net=step.net,
            inputs=[(input_t, input_blob)],
            initial_cell_inputs=[(output_t_prev, initial_input_blob)],
            links={output_t_prev: output_t},
            scope="test_rnn_sum_mull",
        )
        workspace.blobs[input_blob] = input
        workspace.blobs[initial_input_blob] = initial_input

        op = model.net._net.op[-1]
        # Just conviniently store all inputs in an array in the same
        # order as op.input
        inputs = [workspace.blobs[name] for name in op.input]

        def reference(input, initial_input):
            global_ws_name = workspace.CurrentWorkspace()
            input_all = workspace.blobs[input_blob]

            workspace.SwitchWorkspace("ref", create_if_missing=True)
            workspace.blobs[input_blob] = input
            workspace.blobs[output_t_prev] = initial_input.reshape(n, d)
            res_all = np.zeros(shape=input.shape, dtype=np.float32)

            for t_cur in range(T):
                workspace.blobs[input_t] = input_all[t_cur]
                workspace.RunNetOnce(step.net)
                result_t = workspace.blobs[output_t]
                workspace.blobs[output_t_prev] = result_t
                res_all[t_cur] = result_t

            workspace.SwitchWorkspace(global_ws_name)

            shape = list(input.shape)
            shape[0] = 1
            return (res_all, res_all[-1].reshape(shape))

        self.assertReferenceChecks(
            device_option=hu.cpu_do,
            op=op,
            inputs=inputs,
            reference=reference,
            output_to_grad=op.output[0],
            outputs_to_check=[0, 1],
        )

        self.assertGradientChecks(
            device_option=hu.cpu_do,
            op=op,
            inputs=inputs,
            outputs_to_check=0,
            outputs_with_grads=[0],
            threshold=0.01,
            stepsize=0.005,
        )

    # Hacky version of 1-D convolution
    def _convolution_1d(
        self,
        model,
        inputs,
        conv_window,
        conv_filter,
        conv_bias,
        output_name,
        left_pad,
    ):
        if left_pad:
            padding_width = conv_window - 1
        else:
            padding_width = 0

        # [batch_size, inputs_length, state_size]
        inputs_transposed = model.net.Transpose(
            inputs,
            'inputs_transposed',
            axes=[1, 0, 2],
        )
        # [batch_size, 1, inputs_length, state_size]
        inputs_transposed_4d = model.net.ExpandDims(
            inputs_transposed,
            'inputs_transposed_4d',
            dims=[1],
        )
        # [batch_size, 1, inputs_length - conv_window + 1, state_size]
        output_transposed_4d = model.net.Conv(
            [inputs_transposed_4d, conv_filter, conv_bias],
            output_name + '_transposed_4d',
            kernel_h=1,
            kernel_w=conv_window,
            order='NHWC',
            pad_t=0,
            pad_l=padding_width,
            pad_b=0,
            pad_r=0,
        )
        # [batch_size, inputs_length - conv_window + 1, state_size]
        output_transposed = model.net.Squeeze(
            output_transposed_4d,
            output_name + '_transposed',
            dims=[1],
        )
        # [inputs_length - conv_window + 1, batch_size, state_size]
        output = model.net.Transpose(
            output_transposed,
            output_name,
            axes=[1, 0, 2],
        )
        return output

    @given(sequence_length=st.integers(3, 7),
           conv_window=st.integers(1, 3),
           batch_size=st.integers(1, 5),
           state_size=st.integers(1, 5))
    def test_stateful_convolution_forward_only(
        self,
        sequence_length,
        conv_window,
        batch_size,
        state_size,
    ):
        '''
        This unit test demonstrates another ways of using RecurrentNetwork.

        Imagine, that you want to compute convolution over a sequence,
        but sequence elements are not given to you from the beginning,
        so you have to loop over the sequence and compute convolution
        for each element separately. This situation can occur,
        during inference/generation step of the neural networks.

        First of all, you have to provide actual input via recurrent states,
        since the input of RecurrentNetwork should be known in advance.
        Here, we use `fake_inputs` as the input,
        and it's used by the op to extract batch size and sequence length.
        The actual input sequence is stored in the recurrent state
        `input_state`. At every step we generate a new element via input_state_t
        (in this example, input_state_t is generated at random, but
        in a real situation it can be created using convolution output
        from the previous step).

        A few important differences from regular RecurrentNetwork usecase:

        1. input_state_t_prev is not only a single previous element of
        input_state sequence. It is last conv_window elements including (!)
        the current one - input_state_t. We specify that using `link_window`
        argument of RecurrentNetwork. We need that many elements to
        compute a single convolution step. Also, note that `link_window`
        specifies how many elements to link starting at
        `timestep` + `link_offset` position.

        2. First few steps might require additional zero padding from the left,
        since there is no enough element of input_state sequence are available.
        So the initial_state for input_state contains several elements
        (exactly how many pads we need for the first step). Also, because of
        that all offseting over input_state sequence is being shifted
        by length of initial_input_state: see `link_offset` and `alias_offset`
        arguments of RecurrentNetwork.

        In this test, we assert that we get the same result
        if we apply convolution over all elements simultaneously,
        since the whole input_state sequence was generated at the end.
    '''
        model = ModelHelper(name='model')
        fake_inputs = model.param_init_net.UniformFill(
            [],
            'fake_inputs',
            min=-1.0,
            max=1.0,
            shape=[sequence_length, batch_size, state_size],
        )
        initial_input_state = model.param_init_net.ConstantFill(
            [],
            'initial_input_state',
            value=0.0,
            shape=[conv_window - 1, batch_size, state_size],
        )
        initial_output_state = model.param_init_net.ConstantFill(
            [],
            'initial_output_state',
            value=0.0,
            shape=[1, batch_size, state_size],
        )
        step_model = ModelHelper(name='step_model', param_model=model)
        (
            fake_input_t,
            timestep,
            input_state_t_prev,
        ) = step_model.net.AddExternalInputs(
            'fake_input_t',
            'timestep',
            'input_state_t_prev',
        )
        conv_filter = step_model.param_init_net.XavierFill(
            [],
            'conv_filter',
            shape=[state_size, 1, conv_window, state_size],
        )
        conv_bias = step_model.param_init_net.ConstantFill(
            [],
            'conv_bias',
            shape=[state_size],
            value=0.0,
        )
        step_model.params.extend([conv_filter, conv_bias])
        input_state_t = step_model.net.UniformFill(
            [],
            'input_state_t',
            min=-1.0,
            max=1.0,
            shape=[1, batch_size, state_size],
        )
        output_state_t = self._convolution_1d(
            model=step_model,
            inputs=input_state_t_prev,
            conv_window=conv_window,
            conv_filter=conv_filter,
            conv_bias=conv_bias,
            output_name='output_state_t',
            left_pad=False,
        )
        initial_recurrent_states = [initial_input_state, initial_output_state]
        all_inputs = (
            [fake_inputs] + step_model.params + initial_recurrent_states
        )
        all_outputs = ['input_state_all', 'output_state_all']
        recurrent_states = ['input_state', 'output_state']
        input_state_all, output_state_all, _ = model.net.RecurrentNetwork(
            all_inputs,
            all_outputs + ['step_workspaces'],
            param=[all_inputs.index(p) for p in step_model.params],
            alias_src=recurrent_states,
            alias_dst=all_outputs,
            alias_offset=[conv_window - 1, 1],
            recurrent_states=recurrent_states,
            initial_recurrent_state_ids=[
                all_inputs.index(s) for s in initial_recurrent_states
            ],
            link_internal=[
                str(input_state_t_prev),
                str(input_state_t),
                str(output_state_t),
            ],
            link_external=['input_state', 'input_state', 'output_state'],
            link_offset=[0, conv_window - 1, 1],
            link_window=[conv_window, 1, 1],
            backward_link_internal=[],
            backward_link_external=[],
            backward_link_offset=[],
            step_net=step_model.net.Proto(),
            timestep='timestep' if timestep is None else str(timestep),
            outputs_with_grads=[],
        )

        output_states_2 = self._convolution_1d(
            model=model,
            inputs=input_state_all,
            conv_window=conv_window,
            conv_filter=conv_filter,
            conv_bias=conv_bias,
            output_name='output_states_2',
            left_pad=True,
        )

        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)

        np.testing.assert_almost_equal(
            workspace.FetchBlob(output_state_all),
            workspace.FetchBlob(output_states_2),
            decimal=3,
        )
