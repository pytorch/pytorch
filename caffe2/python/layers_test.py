from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
from hypothesis import given

import caffe2.python.hypothesis_test_util as hu

from caffe2.python import (
    layer_model_instantiator,
    schema,
    workspace,
)
from caffe2.python.layers.layers import (
    InstantiationContext,
)
from caffe2.python.layers.tags import Tags
from caffe2.python.layer_test_util import (
    LayersTestCase,
    OpSpec,
)
from caffe2.python.layers.layers import (
    set_request_only,
    is_request_only_scalar,
)


class TestLayers(LayersTestCase):

    def testFCWithoutBias(self):
        output_dims = 2
        fc_without_bias = self.model.FCWithoutBias(
            self.model.input_feature_schema.float_features, output_dims)
        self.model.output_schema = fc_without_bias

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
        self.model.output_schema = sampled_fc

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

    def testBatchMSELoss(self):
        input_record = self.new_record(schema.Struct(
            ('label', schema.Scalar((np.float64, (1,)))),
            ('prediction', schema.Scalar((np.float32, (2,)))),
        ))
        loss = self.model.BatchMSELoss(input_record)
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

    def testBatchSoftmaxLossWeight(self):
        input_record = self.new_record(schema.Struct(
            ('label', schema.Scalar((np.float32, tuple()))),
            ('prediction', schema.Scalar((np.float32, (32,)))),
            ('weight', schema.Scalar((np.float64, (1,))))
        ))
        loss = self.model.BatchSoftmaxLoss(input_record)
        self.assertEqual(schema.Struct(
            ('softmax', schema.Scalar((np.float32, (32,)))),
            ('loss', schema.Scalar(np.float32)),
        ), loss)

    @given(
        X=hu.arrays(dims=[2, 5]),
    )
    def testBatchNormalization(self, X):
        input_record = self.new_record(schema.Scalar((np.float32, (5,))))
        schema.FeedRecord(input_record, [X])
        bn_output = self.model.BatchNormalization(input_record)
        self.assertEqual(schema.Scalar((np.float32, (5,))), bn_output)
        self.model.output_schema = schema.Struct()

        train_init_net, train_net = self.get_training_nets()

        init_ops = self.assertNetContainOps(
            train_init_net,
            [
                OpSpec("ConstantFill", None, None),
                OpSpec("ConstantFill", None, None),
                OpSpec("ConstantFill", None, None),
                OpSpec("ConstantFill", None, None),
            ]
        )

        input_blob = input_record.field_blobs()[0]
        output_blob = bn_output.field_blobs()[0]

        expand_dims_spec = OpSpec(
            "ExpandDims",
            [input_blob],
            [input_blob],
        )

        train_bn_spec = OpSpec(
            "SpatialBN",
            [input_blob, init_ops[0].output[0], init_ops[1].output[0],
                init_ops[2].output[0], init_ops[3].output[0]],
            [output_blob, init_ops[2].output[0], init_ops[3].output[0], None, None],
            {'is_test': 0, 'order': 'NCHW', 'momentum': 0.9},
        )

        test_bn_spec = OpSpec(
            "SpatialBN",
            [input_blob, init_ops[0].output[0], init_ops[1].output[0],
                init_ops[2].output[0], init_ops[3].output[0]],
            [output_blob],
            {'is_test': 1, 'order': 'NCHW', 'momentum': 0.9},
        )

        squeeze_spec = OpSpec(
            "Squeeze",
            [output_blob],
            [output_blob],
        )

        self.assertNetContainOps(
            train_net,
            [expand_dims_spec, train_bn_spec, squeeze_spec]
        )

        eval_net = self.get_eval_net()

        self.assertNetContainOps(
            eval_net,
            [expand_dims_spec, test_bn_spec, squeeze_spec]
        )

        predict_net = self.get_predict_net()

        self.assertNetContainOps(
            predict_net,
            [expand_dims_spec, test_bn_spec, squeeze_spec]
        )

        workspace.RunNetOnce(train_init_net)
        workspace.RunNetOnce(train_net)

        schema.FeedRecord(input_record, [X])
        workspace.RunNetOnce(eval_net)

        schema.FeedRecord(input_record, [X])
        workspace.RunNetOnce(predict_net)

    @given(
        X=hu.arrays(dims=[5, 2]),
        num_to_collect=st.integers(min_value=1, max_value=10),
    )
    def testLastNWindowCollector(self, X, num_to_collect):
        input_record = self.new_record(schema.Scalar(np.float32))
        schema.FeedRecord(input_record, [X])
        last_n = self.model.LastNWindowCollector(input_record, num_to_collect)
        self.run_train_net_forward_only()
        output_record = schema.FetchRecord(last_n)
        start = max(0, 5 - num_to_collect)
        npt.assert_array_equal(X[start:], output_record())

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

    def testGatherRecord(self):
        indices = np.array([1, 3, 4], dtype=np.int32)
        dense = np.array(list(range(20)), dtype=np.float32).reshape(10, 2)
        lengths = np.array(list(range(10)), dtype=np.int32)
        items = np.array(list(range(lengths.sum())), dtype=np.int64)
        items_lengths = np.array(list(range(lengths.sum())), dtype=np.int32)
        items_items = np.array(list(range(items_lengths.sum())), dtype=np.int64)
        record = self.new_record(schema.Struct(
            ('dense', schema.Scalar(np.float32)),
            ('sparse', schema.Struct(
                ('list', schema.List(np.int64)),
                ('list_of_list', schema.List(schema.List(np.int64))),
            )),
            ('empty_struct', schema.Struct())
        ))
        indices_record = self.new_record(schema.Scalar(np.int32))
        input_record = schema.Struct(
            ('indices', indices_record),
            ('record', record),
        )
        schema.FeedRecord(
            input_record,
            [indices, dense, lengths, items, lengths, items_lengths,
             items_items])
        gathered_record = self.model.GatherRecord(input_record)
        self.assertTrue(schema.equal_schemas(gathered_record, record))

        self.run_train_net_forward_only()
        gathered_dense = workspace.FetchBlob(gathered_record.dense())
        np.testing.assert_array_equal(
            np.concatenate([dense[i:i + 1] for i in indices]), gathered_dense)
        gathered_lengths = workspace.FetchBlob(
            gathered_record.sparse.list.lengths())
        np.testing.assert_array_equal(
            np.concatenate([lengths[i:i + 1] for i in indices]),
            gathered_lengths)
        gathered_items = workspace.FetchBlob(
            gathered_record.sparse.list.items())
        offsets = lengths.cumsum() - lengths
        np.testing.assert_array_equal(
            np.concatenate([
                items[offsets[i]: offsets[i] + lengths[i]]
                for i in indices
            ]), gathered_items)

        gathered_items_lengths = workspace.FetchBlob(
            gathered_record.sparse.list_of_list.items.lengths())
        np.testing.assert_array_equal(
            np.concatenate([
                items_lengths[offsets[i]: offsets[i] + lengths[i]]
                for i in indices
            ]),
            gathered_items_lengths
        )

        nested_offsets = []
        nested_lengths = []
        nested_offset = 0
        j = 0
        for l in lengths:
            nested_offsets.append(nested_offset)
            nested_length = 0
            for _i in range(l):
                nested_offset += items_lengths[j]
                nested_length += items_lengths[j]
                j += 1
            nested_lengths.append(nested_length)

        gathered_items_items = workspace.FetchBlob(
            gathered_record.sparse.list_of_list.items.items())
        np.testing.assert_array_equal(
            np.concatenate([
                items_items[nested_offsets[i]:
                            nested_offsets[i] + nested_lengths[i]]
                for i in indices
            ]),
            gathered_items_items
        )

    def testMapToRange(self):
        input_record = self.new_record(schema.Scalar(np.int32))
        map_to_range_output = self.model.MapToRange(input_record,
                                                    max_index=100)
        self.model.output_schema = schema.Struct()

        train_init_net, train_net = self.get_training_nets()

        schema.FeedRecord(
            input_record,
            [np.array([10, 3, 20, 99, 15, 11, 3, 11], dtype=np.int32)]
        )
        workspace.RunNetOnce(train_init_net)
        workspace.RunNetOnce(train_net)
        indices = workspace.FetchBlob(map_to_range_output())
        np.testing.assert_array_equal(
            np.array([1, 2, 3, 4, 5, 6, 2, 6], dtype=np.int32),
            indices
        )

        schema.FeedRecord(
            input_record,
            [np.array([10, 3, 23, 35, 60, 15, 10, 15], dtype=np.int32)]
        )
        workspace.RunNetOnce(train_net)
        indices = workspace.FetchBlob(map_to_range_output())
        np.testing.assert_array_equal(
            np.array([1, 2, 7, 8, 9, 5, 1, 5], dtype=np.int32),
            indices
        )

        eval_net = self.get_eval_net()

        schema.FeedRecord(
            input_record,
            [np.array([10, 3, 23, 35, 60, 15, 200], dtype=np.int32)]
        )
        workspace.RunNetOnce(eval_net)
        indices = workspace.FetchBlob(map_to_range_output())
        np.testing.assert_array_equal(
            np.array([1, 2, 7, 8, 9, 5, 0], dtype=np.int32),
            indices
        )

        schema.FeedRecord(
            input_record,
            [np.array([10, 3, 23, 15, 101, 115], dtype=np.int32)]
        )
        workspace.RunNetOnce(eval_net)
        indices = workspace.FetchBlob(map_to_range_output())
        np.testing.assert_array_equal(
            np.array([1, 2, 7, 5, 0, 0], dtype=np.int32),
            indices
        )

        predict_net = self.get_predict_net()

        schema.FeedRecord(
            input_record,
            [np.array([3, 3, 20, 23, 151, 35, 60, 15, 200], dtype=np.int32)]
        )
        workspace.RunNetOnce(predict_net)
        indices = workspace.FetchBlob(map_to_range_output())
        np.testing.assert_array_equal(
            np.array([2, 2, 3, 7, 0, 8, 9, 5, 0], dtype=np.int32),
            indices
        )

    def testSelectRecordByContext(self):
        float_features = self.model.input_feature_schema.float_features

        float_array = np.array([1.0, 2.0], dtype=np.float32)

        schema.FeedRecord(float_features, [float_array])

        with Tags(Tags.EXCLUDE_FROM_PREDICTION):
            log_float_features, = self.model.Log(float_features, 1)
        joined = self.model.SelectRecordByContext(
            schema.Struct(
                (InstantiationContext.PREDICTION, float_features),
                (InstantiationContext.TRAINING, log_float_features),
                # TODO: TRAIN_ONLY layers are also generated in eval
                (InstantiationContext.EVAL, log_float_features),
            )
        )

        # model.output_schema has to a struct
        self.model.output_schema = schema.Struct((
            'joined', joined
        ))
        predict_net = layer_model_instantiator.generate_predict_net(self.model)
        workspace.RunNetOnce(predict_net)
        predict_output = schema.FetchRecord(predict_net.output_record())
        npt.assert_array_equal(float_array,
                               predict_output['joined']())
        eval_net = layer_model_instantiator.generate_eval_net(self.model)
        workspace.RunNetOnce(eval_net)
        eval_output = schema.FetchRecord(eval_net.output_record())
        npt.assert_array_equal(np.log(float_array),
                               eval_output['joined']())
        _, train_net = (
            layer_model_instantiator.generate_training_nets_forward_only(
                self.model
            )
        )
        workspace.RunNetOnce(train_net)
        train_output = schema.FetchRecord(train_net.output_record())
        npt.assert_array_equal(np.log(float_array),
                               train_output['joined']())

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
        self.model.output_schema = self.model.FC(normalized[0], 2)

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
        self.model.output_schema = self.model.FC(normalized[0], 2)

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
        self.model.output_schema = self.model.FC(softsign[0], 2)

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

    def testFunctionalLayerInputCoercion(self):
        one = self.model.global_constants['ONE']
        two = self.model.Add([one, one], 1)
        self.model.loss = two
        self.run_train_net()
        data = workspace.FetchBlob(two.field_blobs()[0])
        np.testing.assert_array_equal([2.0], data)

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

    def testFunctionalLayerWithOutputDtypes(self):
        loss = self.model.AveragedLoss(
            self.model.input_feature_schema,
            1,
            output_dtypes=(np.float32, (1,)),
        )
        self.assertEqual(1, len(loss.field_types()))
        self.assertEqual(np.float32, loss.field_types()[0].base)
        self.assertEqual((1,), loss.field_types()[0].shape)

    def testPropagateRequestOnly(self):
        # test case when output is request only
        input_record = self.new_record(schema.Struct(
            ('input1', schema.Scalar((np.float32, (32, )))),
            ('input2', schema.Scalar((np.float32, (64, )))),
            ('input3', schema.Scalar((np.float32, (16, )))),
        ))

        set_request_only(input_record)
        concat_output = self.model.Concat(input_record)
        self.assertEqual(is_request_only_scalar(concat_output), True)

        # test case when output is not request only
        input_record2 = self.new_record(schema.Struct(
            ('input4', schema.Scalar((np.float32, (100, ))))
        )) + input_record

        concat_output2 = self.model.Concat(input_record2)
        self.assertEqual(is_request_only_scalar(concat_output2), False)

    def testSetRequestOnly(self):
        input_record = schema.Scalar(np.int64)
        schema.attach_metadata_to_scalars(
            input_record,
            schema.Metadata(
                categorical_limit=100000000,
                expected_value=99,
                feature_specs=schema.FeatureSpec(
                    feature_ids=[1, 100, 1001]
                )
            )
        )

        set_request_only(input_record)
        self.assertEqual(input_record.metadata.categorical_limit, 100000000)
        self.assertEqual(input_record.metadata.expected_value, 99)
        self.assertEqual(
            input_record.metadata.feature_specs.feature_ids,
            [1, 100, 1001]
        )

    @given(
        X=hu.arrays(dims=[5, 5]),  # Shape of X is irrelevant
    )
    def testDropout(self, X):
        input_record = self.new_record(schema.Scalar((np.float32, (1,))))
        schema.FeedRecord(input_record, [X])
        d_output = self.model.Dropout(input_record)
        self.assertEqual(schema.Scalar((np.float32, (1,))), d_output)
        self.model.output_schema = schema.Struct()

        train_init_net, train_net = self.get_training_nets()

        input_blob = input_record.field_blobs()[0]
        output_blob = d_output.field_blobs()[0]

        train_d_spec = OpSpec(
            "Dropout",
            [input_blob],
            [output_blob, None],
            {'is_test': 0, 'ratio': 0.5}
        )

        test_d_spec = OpSpec(
            "Dropout",
            [input_blob],
            [output_blob, None],
            {'is_test': 1, 'ratio': 0.5}
        )

        self.assertNetContainOps(
            train_net,
            [train_d_spec]
        )

        eval_net = self.get_eval_net()

        self.assertNetContainOps(
            eval_net,
            [test_d_spec]
        )

        predict_net = self.get_predict_net()

        self.assertNetContainOps(
            predict_net,
            [test_d_spec]
        )

        workspace.RunNetOnce(train_init_net)
        workspace.RunNetOnce(train_net)

        schema.FeedRecord(input_record, [X])
        workspace.RunNetOnce(eval_net)

        schema.FeedRecord(input_record, [X])
        workspace.RunNetOnce(predict_net)
