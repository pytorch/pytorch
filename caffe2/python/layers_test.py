from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from caffe2.python import (
    layer_model_instantiator,
    schema,
    workspace,
)
from caffe2.python.layer_test_util import (
    LayersTestCase,
    OpSpec,
)


class TestLayers(LayersTestCase):

    def testFCWithoutBias(self):
        output_dims = 2
        fc_without_bias = self.model.FCWithoutBias(
            self.model.input_feature_schema.float_features, output_dims)

        self.assertEqual(
            schema.Scalar((np.float32, (output_dims, ))),
            fc_without_bias
        )

        train_init_net, train_net = self.get_training_nets()

        init_ops = self.assertNetContainOps(
            train_init_net,
            [
                OpSpec("UniformFill", None, None),
            ]
        )

        mat_mul_spec = OpSpec(
            "MatMul",
            [
                self.model.input_feature_schema.float_features(),
                init_ops[0].output[0],
            ],
            fc_without_bias.field_blobs()
        )

        self.assertNetContainOps(train_net, [mat_mul_spec])

        predict_net = self.get_predict_net()
        self.assertNetContainOps(predict_net, [mat_mul_spec])

    def testSamplingTrain(self):
        output_dims = 1000

        indices = self.new_record(schema.Scalar((np.int32, (10,))))
        sampling_prob = self.new_record(schema.Scalar((np.float32, (10, ))))

        sampled_fc = self.model.SamplingTrain(
            schema.Struct(
                ('input', self.model.input_feature_schema.float_features),
                ('indices', indices),
                ('sampling_prob', sampling_prob),
            ),
            "FC",
            output_dims,
        )

        # Check that we don't add prediction layer into the model
        self.assertEqual(1, len(self.model.layers))

        self.assertEqual(
            schema.Scalar((np.float32, (output_dims, ))),
            sampled_fc
        )

        train_init_net, train_net = self.get_training_nets()

        init_ops = self.assertNetContainOps(
            train_init_net,
            [
                OpSpec("UniformFill", None, None),
                OpSpec("UniformFill", None, None),
            ]
        )

        sampled_fc_layer = self.model.layers[0]

        gather_w_spec = OpSpec(
            "Gather",
            [
                init_ops[0].output[0],
                indices(),
            ],
            [
                sampled_fc_layer._prediction_layer.train_param_blobs[0]
            ]
        )
        gather_b_spec = OpSpec(
            "Gather",
            [
                init_ops[1].output[0],
                indices(),
            ],
            [
                sampled_fc_layer._prediction_layer.train_param_blobs[1]
            ]
        )
        train_fc_spec = OpSpec(
            "FC",
            [
                self.model.input_feature_schema.float_features(),
            ] + sampled_fc_layer._prediction_layer.train_param_blobs,
            sampled_fc.field_blobs()
        )
        log_spec = OpSpec("Log", [sampling_prob()], [None])
        sub_spec = OpSpec(
            "Sub",
            [sampled_fc.field_blobs()[0], None],
            sampled_fc.field_blobs()
        )

        train_ops = self.assertNetContainOps(
            train_net,
            [gather_w_spec, gather_b_spec, train_fc_spec, log_spec, sub_spec])

        self.assertEqual(train_ops[3].output[0], train_ops[4].input[1])

        predict_net = self.get_predict_net()
        self.assertNetContainOps(
            predict_net,
            [
                OpSpec(
                    "FC",
                    [
                        self.model.input_feature_schema.float_features(),
                        init_ops[0].output[0],
                        init_ops[1].output[0],
                    ],
                    sampled_fc.field_blobs()
                )
            ]
        )

    def testBatchLRLoss(self):
        input_record = self.new_record(schema.Struct(
            ('label', schema.Scalar((np.float64, (1,)))),
            ('prediction', schema.Scalar((np.float32, (2,)))),
            ('weight', schema.Scalar((np.float64, (1,))))
        ))
        loss = self.model.BatchLRLoss(input_record)
        self.assertEqual(schema.Scalar((np.float32, tuple())), loss)

    def testBatchSigmoidCrossEntropyLoss(self):
        input_record = self.new_record(schema.Struct(
            ('label', schema.Scalar((np.float32, (32,)))),
            ('prediction', schema.Scalar((np.float32, (32,))))
        ))
        loss = self.model.BatchSigmoidCrossEntropyLoss(input_record)
        self.assertEqual(schema.Scalar((np.float32, tuple())), loss)

    def testBatchSoftmaxLoss(self):
        input_record = self.new_record(schema.Struct(
            ('label', schema.Scalar((np.float32, tuple()))),
            ('prediction', schema.Scalar((np.float32, (32,))))
        ))
        loss = self.model.BatchSoftmaxLoss(input_record)
        self.assertEqual(schema.Struct(
            ('softmax', schema.Scalar((np.float32, (32,)))),
            ('loss', schema.Scalar(np.float32)),
        ), loss)

    def testUniformSampling(self):
        input_record = self.new_record(schema.Scalar(np.int32))
        input_array = np.array([3, 10, 11, 15, 20, 99], dtype=np.int32)
        schema.FeedRecord(input_record, [input_array])
        num_samples = 20
        num_elements = 100
        uniform_sampling_output = self.model.UniformSampling(
            input_record, num_samples, num_elements)
        self.model.loss = uniform_sampling_output
        self.run_train_net()
        samples = workspace.FetchBlob(uniform_sampling_output.samples())
        sampling_prob = workspace.FetchBlob(
            uniform_sampling_output.sampling_prob())
        self.assertEqual(num_samples, len(samples))
        np.testing.assert_array_equal(input_array, samples[:len(input_array)])
        np.testing.assert_almost_equal(
            np.array([float(num_samples) / num_elements] * num_samples,
                     dtype=np.float32),
            sampling_prob
        )

    def testFunctionalLayer(self):
        def normalize(net, in_record, out_record):
            mean = net.ReduceFrontMean(in_record(), 1)
            net.Sub(
                [in_record(), mean],
                out_record[0](),
                broadcast=1)
        normalized = self.model.Functional(
            self.model.input_feature_schema.float_features, 1,
            normalize, name="normalizer")

        # Attach metadata to one of the outputs and use it in FC
        normalized[0].set_type((np.float32, 32))
        self.model.FC(normalized[0], 2)

        predict_net = layer_model_instantiator.generate_predict_net(
            self.model)
        ops = predict_net.Proto().op
        assert len(ops) == 3
        assert ops[0].type == "ReduceFrontMean"
        assert ops[1].type == "Sub"
        assert ops[2].type == "FC"
        assert len(ops[0].input) == 1
        assert ops[0].input[0] ==\
            self.model.input_feature_schema.float_features()
        assert len(ops[1].output) == 1
        assert ops[1].output[0] in ops[2].input

    def testFunctionalLayerHelper(self):
        mean = self.model.ReduceFrontMean(
            self.model.input_feature_schema.float_features, 1)
        normalized = self.model.Sub(
            schema.Tuple(
                self.model.input_feature_schema.float_features, mean[0]),
            1, broadcast=1)
        # Attach metadata to one of the outputs and use it in FC
        normalized[0].set_type((np.float32, (32,)))
        self.model.FC(normalized[0], 2)

        predict_net = layer_model_instantiator.generate_predict_net(
            self.model)
        ops = predict_net.Proto().op
        assert len(ops) == 3
        assert ops[0].type == "ReduceFrontMean"
        assert ops[1].type == "Sub"
        assert ops[2].type == "FC"
        assert len(ops[0].input) == 1
        assert ops[0].input[0] ==\
            self.model.input_feature_schema.float_features()
        assert len(ops[1].output) == 1
        assert ops[1].output[0] in ops[2].input

    def testFunctionalLayerHelperAutoInference(self):
        softsign = self.model.Softsign(
            schema.Tuple(self.model.input_feature_schema.float_features),
            1)
        assert len(softsign.field_types()) == 1
        assert softsign.field_types()[0].base == np.float32
        assert softsign.field_types()[0].shape == (32,)
        self.model.FC(softsign[0], 2)

        predict_net = layer_model_instantiator.generate_predict_net(
            self.model)
        ops = predict_net.Proto().op
        assert len(ops) == 2
        assert ops[0].type == "Softsign"
        assert ops[1].type == "FC"
        assert len(ops[0].input) == 1
        assert ops[0].input[0] ==\
            self.model.input_feature_schema.float_features()
        assert len(ops[0].output) == 1
        assert ops[0].output[0] in ops[1].input

    def testFunctionalLayerHelperAutoInferenceScalar(self):
        loss = self.model.AveragedLoss(self.model.input_feature_schema, 1)
        self.assertEqual(1, len(loss.field_types()))
        self.assertEqual(np.float32, loss.field_types()[0].base)
        self.assertEqual(tuple(), loss.field_types()[0].shape)

    def testFunctionalLayerWithOutputNames(self):
        k = 3
        topk = self.model.TopK(
            self.model.input_feature_schema,
            output_names_or_num=['values', 'indices'],
            k=k,
        )
        self.assertEqual(2, len(topk.field_types()))
        self.assertEqual(np.float32, topk.field_types()[0].base)
        self.assertEqual((k,), topk.field_types()[0].shape)
        self.assertEqual(np.int32, topk.field_types()[1].base)
        self.assertEqual((k,), topk.field_types()[1].shape)
        self.assertEqual(['TopK/values', 'TopK/indices'], topk.field_blobs())
