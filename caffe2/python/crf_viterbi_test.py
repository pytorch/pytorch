



from caffe2.python import workspace, crf

from caffe2.python.cnn import CNNModelHelper
from caffe2.python.crf_predict import crf_update_predictions
from caffe2.python.test_util import TestCase
import hypothesis.strategies as st
from hypothesis import given, settings
import numpy as np


class TestCrfDecode(TestCase):

    @given(num_tags=st.integers(2, 4), num_words=st.integers(2, 15))
    @settings(deadline=2000)
    def test_crf_viterbi(self, num_tags, num_words):
        model = CNNModelHelper(name='external')
        predictions = np.random.randn(num_words, num_tags).astype(np.float32)
        transitions = np.random.uniform(
            low=-1, high=1, size=(num_tags + 2, num_tags + 2)
        ).astype(np.float32)
        predictions_blob, transitions_blob = (
            model.net.AddExternalInputs('predictions', 'crf_transitions')
        )
        workspace.FeedBlob(str(transitions_blob), transitions)
        workspace.FeedBlob(str(predictions_blob), predictions)
        crf_layer = crf.CRFWithLoss(model, num_tags, transitions_blob)

        updated_predictions = crf_update_predictions(
            model, crf_layer, predictions_blob
        )
        ref_predictions = crf_layer.update_predictions(predictions_blob)

        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)

        updated_predictions = workspace.FetchBlob(str(updated_predictions))
        ref_predictions = workspace.FetchBlob(str(ref_predictions))
        np.testing.assert_allclose(
            updated_predictions,
            ref_predictions,
            atol=1e-4, rtol=1e-4, err_msg='Mismatch in CRF predictions'
        )
