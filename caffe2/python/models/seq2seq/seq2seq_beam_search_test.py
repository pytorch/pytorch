




import numpy as np
import os
import tempfile

from caffe2.python import test_util, workspace
import caffe2.python.models.seq2seq.seq2seq_util as seq2seq_util
from caffe2.python.models.seq2seq.train import Seq2SeqModelCaffe2
from caffe2.python.models.seq2seq.translate import (
    Seq2SeqModelCaffe2EnsembleDecoder,
)


class Seq2SeqBeamSearchTest(test_util.TestCase):

    def _build_seq2seq_model(
        self,
        model_params,
        tmp_dir,
        source_vocab_size=20,
        target_vocab_size=20,
        num_gpus=0,
        batch_size=2,
    ):
        training_params = dict(
            model_params,
            batch_size=batch_size,
            optimizer_params=dict(
                learning_rate=0.1,
            ),
            max_gradient_norm=1.0,
        )

        model_obj = Seq2SeqModelCaffe2(
            training_params,
            source_vocab_size,
            target_vocab_size,
            num_gpus,
        )
        model_obj.initialize_from_scratch()

        checkpoint_path_prefix = os.path.join(tmp_dir, 'checkpoint')
        checkpoint_path = model_obj.save(
            checkpoint_path_prefix=checkpoint_path_prefix,
            current_step=0,
        )

        return model_obj, checkpoint_path

    def _run_compare_train_inference(self, model_params):
        tmp_dir = tempfile.mkdtemp()

        model_obj, checkpoint_path = self._build_seq2seq_model(
            model_params,
            tmp_dir=tmp_dir,
            source_vocab_size=20,
            target_vocab_size=20,
            num_gpus=0,
            batch_size=2,
        )
        assert model_obj is not None

        translate_params = dict(
            ensemble_models=[dict(
                source_vocab={i: str(i) for i in range(20)},
                target_vocab={i: str(i) for i in range(20)},
                model_params=model_params,
                model_file=checkpoint_path,
            )],
            decoding_params=dict(
                beam_size=3,
                word_reward=0,
                unk_reward=0,
            ),
        )

        beam_decoder_model = Seq2SeqModelCaffe2EnsembleDecoder(translate_params)
        beam_decoder_model.load_models()

        encoder_lengths = 5
        decoder_lengths = 7

        for _ in range(3):
            encoder_inputs = np.random.random_integers(
                low=3,  # after GO_ID (1) and EOS_ID (2)
                high=19,
                size=encoder_lengths,
            )
            targets, _, beam_model_score = beam_decoder_model.decode(
                encoder_inputs,
                decoder_lengths,
            )
            targets_2, _, beam_model_score = beam_decoder_model.decode(
                encoder_inputs,
                decoder_lengths,
            )
            self.assertEqual(targets, targets_2)

            workspace.FeedBlob(
                'encoder_inputs',
                np.array(
                    [list(reversed(encoder_inputs))]
                ).transpose().astype(dtype=np.int32))
            workspace.FeedBlob(
                'encoder_lengths',
                np.array([len(encoder_inputs)]).astype(dtype=np.int32),
            )
            decoder_inputs = [seq2seq_util.GO_ID] + targets[:-1]
            workspace.FeedBlob(
                'decoder_inputs',
                np.array([decoder_inputs]).transpose().astype(dtype=np.int32),
            )
            workspace.FeedBlob(
                'decoder_lengths',
                np.array([len(decoder_inputs)]).astype(dtype=np.int32),
            )
            workspace.FeedBlob(
                'targets',
                np.array([targets]).transpose().astype(dtype=np.int32),
            )
            workspace.FeedBlob(
                'target_weights',
                np.array([[1.0] * len(targets)]).astype(dtype=np.float32),
            )

            workspace.RunNet(model_obj.forward_net)
            train_model_score = workspace.FetchBlob('total_loss_scalar')

            np.testing.assert_almost_equal(
                beam_model_score,
                train_model_score,
                decimal=4,
            )

    def test_attention(self):
        model_params = dict(
            attention='regular',
            decoder_layer_configs=[
                dict(
                    num_units=32,
                ),
            ],
            encoder_type=dict(
                encoder_layer_configs=[
                    dict(
                        num_units=16,
                    ),
                ],
                use_bidirectional_encoder=True,
            ),
            encoder_embedding_size=8,
            decoder_embedding_size=8,
            decoder_softmax_size=None,
        )
        self._run_compare_train_inference(model_params)

    def test_2layer_attention(self):
        model_params = dict(
            attention='regular',
            decoder_layer_configs=[
                dict(
                    num_units=32,
                ),
                dict(
                    num_units=32,
                ),
            ],
            encoder_type=dict(
                encoder_layer_configs=[
                    dict(
                        num_units=16,
                    ),
                    dict(
                        num_units=32,
                    ),
                ],
                use_bidirectional_encoder=True,
            ),
            encoder_embedding_size=8,
            decoder_embedding_size=8,
            decoder_softmax_size=None,
        )
        self._run_compare_train_inference(model_params)

    def test_multi_decoder(self):
        model_params = dict(
            attention='regular',
            decoder_layer_configs=[
                dict(
                    num_units=32,
                ),
                dict(
                    num_units=32,
                ),
                dict(
                    num_units=32,
                ),
            ],
            encoder_type=dict(
                encoder_layer_configs=[
                    dict(
                        num_units=32,
                    ),
                ],
                use_bidirectional_encoder=False,
            ),
            encoder_embedding_size=8,
            decoder_embedding_size=8,
            decoder_softmax_size=None,
        )
        self._run_compare_train_inference(model_params)
