




from caffe2.python import core, workspace
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import hypothesis.strategies as st
import numpy as np


class TestPairWiseLossOps(serial.SerializedTestCase):
    @given(X=hu.arrays(dims=[2, 1],
                       elements=hu.floats(min_value=0.0, max_value=10.0)),
           label=hu.arrays(dims=[2, 1],
                           elements=st.integers(min_value=0, max_value=1),
                           dtype=np.float32),
           **hu.gcs_cpu_only)
    def test_pair_wise_loss_predictions(self, X, label, gc, dc):
        workspace.FeedBlob('X', X)
        workspace.FeedBlob('label', label)
        new_label = np.array([label[1], label[0]])
        new_x = np.array([X[1], X[0]])
        workspace.FeedBlob('new_x', new_x)
        workspace.FeedBlob('new_label', new_label)
        net = core.Net('net')
        net.PairWiseLoss(['X', 'label'], ['output'])
        net.PairWiseLoss(['new_x', 'new_label'], ['new_output'])
        plan = core.Plan('predict_data')
        plan.AddStep(core.execution_step('predict_data',
                                         [net], num_iter=1))
        workspace.RunPlan(plan)
        output = workspace.FetchBlob('output')
        new_output = workspace.FetchBlob('new_output')
        sign = 1 if label[0] > label[1] else -1
        if label[0] == label[1]:
            self.assertEqual(np.asscalar(output), 0)
            return

        self.assertAlmostEqual(
            np.asscalar(output),
            np.asscalar(np.log(1 + np.exp(sign * (X[1] - X[0])))),
            delta=1e-4
        )
        # check swapping row order doesn't alter overall loss
        self.assertAlmostEqual(output, new_output)

    @given(X=hu.arrays(dims=[2, 1],
                       elements=hu.floats(min_value=0.0, max_value=10.0)),
           label=hu.arrays(dims=[2, 1],
                           elements=st.integers(min_value=0, max_value=1),
                           dtype=np.float32),
           dY=hu.arrays(dims=[1],
                        elements=hu.floats(min_value=1, max_value=10)),
           **hu.gcs_cpu_only)
    def test_pair_wise_loss_gradient(self, X, label, dY, gc, dc):
        workspace.FeedBlob('X', X)
        workspace.FeedBlob('dY', dY)
        workspace.FeedBlob('label', label)
        net = core.Net('net')
        net.PairWiseLossGradient(
            ['X', 'label', 'dY'],
            ['dX'],
        )
        plan = core.Plan('predict_data')
        plan.AddStep(core.execution_step('predict_data',
                                         [net], num_iter=1))
        workspace.RunPlan(plan)
        dx = workspace.FetchBlob('dX')
        sign = 1 if label[0] > label[1] else -1
        if label[0] == label[1]:
            self.assertEqual(np.asscalar(dx[0]), 0)
            return
        self.assertAlmostEqual(
            np.asscalar(dx[0]),
            np.asscalar(-dY[0] * sign / (1 + np.exp(sign * (X[0] - X[1])))),
            delta=1e-2 * abs(np.asscalar(dx[0])))

        self.assertEqual(np.asscalar(dx[0]), np.asscalar(-dx[1]))
        delta = 1e-3
        up_x = np.array([[X[0] + delta], [X[1]]], dtype=np.float32)
        down_x = np.array([[X[0] - delta], [X[1]]], dtype=np.float32)
        workspace.FeedBlob('up_x', up_x)
        workspace.FeedBlob('down_x', down_x)
        new_net = core.Net('new_net')
        new_net.PairWiseLoss(['up_x', 'label'], ['up_output'])
        new_net.PairWiseLoss(['down_x', 'label'], ['down_output'])

        plan = core.Plan('predict_data')
        plan.AddStep(core.execution_step('predict_data', [new_net], num_iter=1))
        workspace.RunPlan(plan)
        down_output_pred = workspace.FetchBlob('down_output')
        up_output_pred = workspace.FetchBlob('up_output')
        np.testing.assert_allclose(
            np.asscalar(dx[0]),
            np.asscalar(
                0.5 * dY[0] *
                (up_output_pred[0] - down_output_pred[0]) / delta),
            rtol=1e-2, atol=1e-2)

    @serial.given(n=st.integers(0, 10), k=st.integers(1, 5), **hu.gcs_cpu_only)
    def test_pair_wise_loss_batch(self, n, k, gc, dc):
        lengths = np.random.randint(k, size=n).astype(np.int32) + 1
        X = np.random.rand(sum(lengths)).astype(np.float32)
        label = np.random.randint(k, size=sum(lengths)).astype(np.float32)

        def pair_wise_op(X, label, lengths):
            N = lengths.size
            output = np.zeros(N).astype(np.float32)

            def f(x):
                return np.log(1 + np.exp(x))

            offset = 0
            for idx in range(N):
                offset += lengths[idx - 1] if idx > 0 else 0
                count = 0
                for i in range(offset, offset + lengths[idx]):
                    for j in range(offset, i):
                        if label[i] == label[j]:
                            continue
                        sign = 1 if label[i] > label[j] else -1
                        output[idx] += f(sign * (X[j] - X[i]))
                        count += 1
                if count > 0:
                    output[idx] /= count
            return [output]

        op = core.CreateOperator(
            'PairWiseLoss',
            ['X', 'label', 'lengths'],
            'out'
        )

        # Check against numpy reference
        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X, label, lengths],
            reference=pair_wise_op,
        )
        # Check over multiple devices
        self.assertDeviceChecks(dc, op, [X, label, lengths], [0])
        # Gradient check
        self.assertGradientChecks(gc, op, [X, label, lengths], 0, [0])
