



from caffe2.python import workspace, crf, brew
from caffe2.python.model_helper import ModelHelper
import numpy as np
from scipy.special import logsumexp
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
from hypothesis import given, settings


class TestCRFOp(hu.HypothesisTestCase):

    @given(num_tags=st.integers(2, 4),
           num_words=st.integers(2, 15))
    @settings(deadline=1000)
    def test_crf_with_loss_op(self, num_tags, num_words):
        model = ModelHelper(name='external')
        embeddings_dim = 200
        embeddings = np.random.randn(num_words, embeddings_dim).astype(np.float32)
        transitions = np.random.uniform(
            low=-1, high=1, size=(num_tags + 2, num_tags + 2)
        ).astype(np.float32)
        labels = np.random.randint(num_tags, size=(num_words)).astype(np.int64)
        embeddings_blob, labels_blob, transitions_blob = (
            model.net.AddExternalInputs(
                'embeddings_blob',
                'labels_blob',
                'crf_transitions')
        )
        workspace.FeedBlob(str(embeddings_blob), embeddings)
        workspace.FeedBlob(str(labels_blob), labels)
        workspace.FeedBlob(str(transitions_blob), transitions)
        predictions_blob = brew.fc(
            model,
            embeddings_blob, "fc_0",
            embeddings_dim, num_tags,
            ('UniformFill', {'min': -1.0}, {'max': 1.0}),
            ('UniformFill', {'min': -1.0}, {'max': 1.0})
        )
        crf_layer = crf.CRFWithLoss(model, num_tags, transitions_blob)
        crf_loss = crf_layer.crf_loss(predictions_blob, labels_blob)
        model.net.AddGradientOperators([crf_loss])
        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)
        loss = workspace.FetchBlob(str(crf_loss))
        predictions = workspace.FetchBlob(str(predictions_blob))
        np.testing.assert_allclose(
            loss,
            self._compute_loss_manual(
                predictions, num_tags, labels, transitions
            ),
            atol=0.001,
            rtol=0.001,
            err_msg='CRF LOSS is not matching the reference'
        )

    @given(num_tags=st.integers(1, 4),
           num_words=st.integers(2, 4))
    @settings(deadline=10000)
    def test_crf_gradient(self, num_tags, num_words):
        base_model = ModelHelper(name='base_model')
        transitions = np.random.randn(
            num_tags + 2, num_tags + 2
        ).astype(np.float32)
        predictions = np.random.randn(num_words, 1, num_tags + 2).astype(np.float32)
        initial = np.random.randn(1, num_tags + 2).astype(np.float32)
        predictions_blob, transitions_blob, initial_blob = (
            base_model.net.AddExternalInputs(
                'predictions_blob', 'crf_transitions', 'inital_blob'
            )
        )

        workspace.FeedBlob(str(predictions_blob), predictions)
        workspace.FeedBlob(str(transitions_blob), transitions)
        workspace.FeedBlob(str(initial_blob), initial)

        crf_layer = crf.CRFWithLoss(base_model, num_tags, transitions_blob)
        crf_layer.build_crf_net(
            predictions_blob, initial_blob, transitions_blob
        )
        op = base_model.net._net.op[-1]
        workspace.RunNetOnce(base_model.param_init_net)
        gradients_to_check = (
            index for (index, input_name) in enumerate(op.input)
            if input_name != "crf_net/zero_segment_id"
        )

        inputs = [workspace.FetchBlob(name) for name in op.input]
        for param in gradients_to_check:
            self.assertGradientChecks(
                device_option=hu.cpu_do,
                op=op,
                inputs=inputs,
                outputs_to_check=param,
                outputs_with_grads=[1],
                threshold=0.05,
                stepsize=0.001,
            )

    def _compute_loss_manual(self, predictions, num_tags, labels, transitions):
        low_score = -1000
        b_s = np.array(
            [[low_score] * num_tags + [0, low_score]]
        ).astype(np.float32)
        e_s = np.array(
            [[low_score] * num_tags + [low_score, 0]]
        ).astype(np.float32)
        predictions = np.concatenate(
            [predictions, low_score * np.ones((predictions.shape[0], 2))],
            axis=1
        )
        predictions = np.concatenate(
            [b_s, predictions, e_s],
            axis=0
        )
        b_id = np.array([num_tags], dtype=np.int32)
        e_id = np.array([num_tags + 1], dtype=np.int32)
        labels = np.concatenate(
            [b_id, labels, e_id],
            axis=0
        )
        curr_state = predictions[0]
        input_states = predictions[1:]

        for input_state in input_states:
            prev = np.expand_dims(curr_state, axis=1)
            curr_input = np.expand_dims(input_state, axis=0)
            curr_state = logsumexp(prev + curr_input + transitions, axis=0)

        total_score = logsumexp(curr_state, axis=0)
        # Compute best path score
        unary_scores = sum(w[labels[i]] for i, w in enumerate(predictions))
        binary_scores = sum(
            transitions[a][b] for a, b in zip(labels[:-1], labels[1:])
        )
        loss = total_score - (binary_scores + unary_scores)
        return loss
