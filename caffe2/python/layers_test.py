




import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
from hypothesis import given, settings

import caffe2.python.hypothesis_test_util as hu

from caffe2.python import (
    layer_model_instantiator,
    core,
    schema,
    workspace,
)
from caffe2.python.layers.layers import (
    AccessedFeatures,
    almost_equal_schemas,
    get_key,
    IdList,
    IdScoreList,
    InstantiationContext,
    is_request_only_scalar,
    set_request_only,
)
from caffe2.python.layers.tags import Tags
from caffe2.python.layer_test_util import (
    LayersTestCase,
    OpSpec,
)
import logging
logger = logging.getLogger(__name__)


class TestLayers(LayersTestCase):
    def testSparseDropoutWithReplacement(self):
        input_record = schema.NewRecord(self.model.net, IdList)
        self.model.output_schema = schema.Struct()

        lengths_blob = input_record.field_blobs()[0]
        values_blob = input_record.field_blobs()[1]
        lengths = np.array([1] * 10).astype(np.int32)
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.int64)
        workspace.FeedBlob(lengths_blob, lengths)
        workspace.FeedBlob(values_blob, values)

        out = self.model.SparseDropoutWithReplacement(
            input_record, 0.0, 0.5, 1.0, -1, output_names_or_num=1)
        self.assertEqual(schema.List(schema.Scalar(np.int64,)), out)

        train_init_net, train_net = self.get_training_nets()
        eval_net = self.get_eval_net()
        predict_net = self.get_predict_net()

        workspace.RunNetOnce(train_init_net)
        workspace.RunNetOnce(train_net)
        out_values = workspace.FetchBlob(out.items())
        out_lengths = workspace.FetchBlob(out.lengths())
        self.assertBlobsEqual(out_values, values)
        self.assertBlobsEqual(out_lengths, lengths)

        workspace.RunNetOnce(eval_net)

        workspace.RunNetOnce(predict_net)
        predict_values = workspace.FetchBlob("values_auto_0")
        predict_lengths = workspace.FetchBlob("lengths_auto_0")
        self.assertBlobsEqual(predict_values, np.array([-1] * 10).astype(np.int64))
        self.assertBlobsEqual(predict_lengths, lengths)

    def testAddLoss(self):
        input_record_LR = self.new_record(
            schema.Struct(
                ('label', schema.Scalar((np.float64, (1, )))),
                ('logit', schema.Scalar((np.float32, (2, )))),
                ('weight', schema.Scalar((np.float64, (1, ))))
            )
        )
        loss_LR = self.model.BatchLRLoss(input_record_LR)

        self.model.add_loss(loss_LR)
        assert 'unnamed' in self.model.loss
        self.assertEqual(
            schema.Scalar((np.float32, tuple())), self.model.loss.unnamed
        )
        self.assertEqual(loss_LR, self.model.loss.unnamed)

        self.model.add_loss(loss_LR, 'addLoss')
        assert 'addLoss' in self.model.loss
        self.assertEqual(
            schema.Scalar((np.float32, tuple())), self.model.loss.addLoss
        )
        self.assertEqual(loss_LR, self.model.loss.addLoss)

        self.model.add_loss(
            schema.Scalar(
                dtype=np.float32, blob=core.BlobReference('loss_blob_1')
            ), 'addLoss'
        )
        assert 'addLoss_auto_0' in self.model.loss
        self.assertEqual(
            schema.Scalar((np.float32, tuple())), self.model.loss.addLoss_auto_0
        )
        assert core.BlobReference('loss_blob_1') in self.model.loss.field_blobs()

        self.model.add_loss(
            schema.Struct(
                (
                    'structName', schema.Scalar(
                        dtype=np.float32,
                        blob=core.BlobReference('loss_blob_2')
                    )
                )
            ), 'addLoss'
        )
        assert 'addLoss_auto_1' in self.model.loss
        self.assertEqual(
            schema.Struct(('structName', schema.Scalar((np.float32, tuple())))),
            self.model.loss.addLoss_auto_1
        )
        assert core.BlobReference('loss_blob_2') in self.model.loss.field_blobs()

        loss_in_tuple_0 = schema.Scalar(
            dtype=np.float32, blob=core.BlobReference('loss_blob_in_tuple_0')
        )

        loss_in_tuple_1 = schema.Scalar(
            dtype=np.float32, blob=core.BlobReference('loss_blob_in_tuple_1')
        )

        loss_tuple = schema.NamedTuple(
            'loss_in_tuple', * [loss_in_tuple_0, loss_in_tuple_1]
        )
        self.model.add_loss(loss_tuple, 'addLoss')
        assert 'addLoss_auto_2' in self.model.loss
        self.assertEqual(
            schema.Struct(
                ('loss_in_tuple_0', schema.Scalar((np.float32, tuple()))),
                ('loss_in_tuple_1', schema.Scalar((np.float32, tuple())))
            ), self.model.loss.addLoss_auto_2
        )
        assert core.BlobReference('loss_blob_in_tuple_0')\
         in self.model.loss.field_blobs()
        assert core.BlobReference('loss_blob_in_tuple_1')\
         in self.model.loss.field_blobs()

    def testFilterMetricSchema(self):
        self.model.add_metric_field("a:b", schema.Scalar())
        self.model.add_metric_field("a:c", schema.Scalar())
        self.model.add_metric_field("d", schema.Scalar())

        self.assertEqual(
            self.model.metrics_schema,
            schema.Struct(
                ("a", schema.Struct(
                    ("b", schema.Scalar()),
                    ("c", schema.Scalar()),
                )),
                ("d", schema.Scalar()),
            ))

        self.model.filter_metrics_schema({"a:b", "d"})
        self.assertEqual(
            self.model.metrics_schema,
            schema.Struct(
                ("a", schema.Struct(
                    ("b", schema.Scalar()),
                )),
                ("d", schema.Scalar()),
            ))

    def testAddOutputSchema(self):
        # add the first field
        self.model.add_output_schema('struct', schema.Struct())
        expected_output_schema = schema.Struct(('struct', schema.Struct()))
        self.assertEqual(
            self.model.output_schema,
            expected_output_schema,
        )

        # add the second field
        self.model.add_output_schema('scalar', schema.Scalar(np.float64))
        expected_output_schema = schema.Struct(
            ('struct', schema.Struct()),
            ('scalar', schema.Scalar(np.float64)),
        )
        self.assertEqual(
            self.model.output_schema,
            expected_output_schema,
        )

        # overwrite a field should raise
        with self.assertRaises(AssertionError):
            self.model.add_output_schema('scalar', schema.Struct())

    def _test_net(self, net, ops_list):
        '''
        Helper function to assert the net contains some set of operations and
        then to run the net.

        Inputs:
            net -- the network to test and run
            ops_list -- the list of operation specifications to check for
                        in the net
        '''
        ops_output = self.assertNetContainOps(net, ops_list)
        workspace.RunNetOnce(net)
        return ops_output

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

    def testFCWithBootstrap(self):
        output_dims = 1
        fc_with_bootstrap = self.model.FCWithBootstrap(
            self.model.input_feature_schema.float_features,
            output_dims=output_dims,
            num_bootstrap=2,
            max_fc_size=-1
        )
        self.model.output_schema = fc_with_bootstrap


        self.assertEqual(len(fc_with_bootstrap), 4)

        # must be in this order
        assert (
            core.BlobReference("fc_with_bootstrap/bootstrap_iteration_0/indices") == fc_with_bootstrap[0].field_blobs()[0]
        )
        assert (
            core.BlobReference("fc_with_bootstrap/bootstrap_iteration_0/preds") == fc_with_bootstrap[1].field_blobs()[0]
        )
        assert (
            core.BlobReference("fc_with_bootstrap/bootstrap_iteration_1/indices") == fc_with_bootstrap[2].field_blobs()[0]
        )
        assert (
            core.BlobReference("fc_with_bootstrap/bootstrap_iteration_1/preds") == fc_with_bootstrap[3].field_blobs()[0]
        )

        train_init_net, train_net = self.get_training_nets()
        predict_net = layer_model_instantiator.generate_predict_net(self.model)

        train_proto = train_net.Proto()
        eval_proto = predict_net.Proto()

        train_ops = train_proto.op
        eval_ops = eval_proto.op

        master_train_ops = [
            "Shape",
            "GivenTensorInt64Fill",
            "Gather",
            "GivenTensorIntFill",
            "GivenTensorIntFill",
            "Cast",
            "Sub",
            "UniformIntFill",
            "Gather",
            "FC",
            "UniformIntFill",
            "Gather",
            "FC",
        ]

        master_eval_ops = [
            "Shape",
            "GivenTensorInt64Fill",
            "Gather",
            "GivenTensorIntFill",
            "GivenTensorIntFill",
            "Cast",
            "Sub",
            "UniformIntFill",
            "FC",
            "UniformIntFill",
            "FC",
        ]

        assert len(train_ops) == len(master_train_ops)
        assert len(eval_ops) == len(master_eval_ops)

        assert train_proto.external_input == eval_proto.external_input
        assert train_proto.external_output == list()

        # make sure all the ops are present and unchanged for train_net and eval_net
        for idx, op in enumerate(master_train_ops):
            assert train_ops[idx].type == op

        for idx, op in enumerate(master_eval_ops):
            assert eval_ops[idx].type == op


    def testFCwithAxis2(self):
        input_dim = 10
        output_dim = 30
        max_length = 20
        input_record = self.new_record(
            schema.Struct(
                ('history_sequence', schema.Scalar((np.float32, (max_length,
                    input_dim)))),
            )
        )
        fc_out = self.model.FC(
            input_record.history_sequence, output_dim,
            axis=2)
        self.model.output_schema = fc_out
        self.assertEqual(
            schema.Scalar((np.float32, (max_length, output_dim))),
            fc_out
        )

        train_init_net, train_net = self.get_training_nets()

    def testFCTransposed(self):
        input_dim = 10
        output_dim = 30
        max_length = 20
        input_record = self.new_record(
            schema.Struct(
                ('history_sequence', schema.Scalar((np.float32, (max_length,
                    input_dim)))),
            )
        )
        fc_transposed_out = self.model.FC(
            input_record.history_sequence, output_dim,
            axis=2, transposed=True)
        self.model.output_schema = fc_transposed_out
        self.assertEqual(
            schema.Scalar((np.float32, (max_length, output_dim))),
            fc_transposed_out
        )

        train_init_net, train_net = self.get_training_nets()

    def testFCTransposedWithMaxFCSize(self):
        input_dim = 10
        output_dim = 30
        max_length = 20
        input_record = self.new_record(
            schema.Struct(
                ('history_sequence', schema.Scalar((np.float32, (max_length,
                    input_dim)))),
            )
        )
        fc_transposed_out = self.model.FC(
            input_record.history_sequence, output_dim,
            max_fc_size=input_dim * output_dim // 2,
            axis=2, transposed=True)
        self.model.output_schema = fc_transposed_out
        self.assertEqual(
            schema.Scalar((np.float32, (max_length, output_dim))),
            fc_transposed_out
        )

        train_init_net, train_net = self.get_training_nets()

    def testSparseLookupSumPoolingWithEviction(self):
        # Create test embedding table of 1 row
        record = schema.NewRecord(self.model.net, schema.Struct(
            ('sparse', schema.Struct(
                ('sparse_feature_0', schema.ListWithEvicted(
                    schema.Scalar(np.int64,
                                  metadata=schema.Metadata(categorical_limit=1)),)),)),
        ))
        embedding_dim = 8
        lengths_blob = record.sparse.sparse_feature_0.lengths.get()
        values_blob = record.sparse.sparse_feature_0.items.get()
        evicted_values_blob = record.sparse.sparse_feature_0._evicted_values.get()
        lengths = np.array([1]).astype(np.int32)
        values = np.array([0]).astype(np.int64)
        # Need to reset row 0
        evicted_values = np.array([0]).astype(np.int64)
        workspace.FeedBlob(lengths_blob, lengths)
        workspace.FeedBlob(values_blob, values)
        workspace.FeedBlob(evicted_values_blob, evicted_values)

        embedding_after_pooling = self.model.SparseLookup(
            record.sparse.sparse_feature_0, [embedding_dim], 'Sum', weight_init=("ConstantFill", {"value": 1.0}))

        self.model.output_schema = schema.Struct()
        self.assertEqual(
            schema.Scalar((np.float32, (embedding_dim, ))),
            embedding_after_pooling
        )
        train_init_net, train_net = self.get_training_nets()
        workspace.RunNetOnce(train_init_net)
        embedding_after_init = workspace.FetchBlob("sparse_lookup/w")
        # Change row 0's value before reset
        new_values = np.array([[2, 2, 2, 2, 2, 2, 2, 2]]).astype(np.float32)
        workspace.FeedBlob("sparse_lookup/w", new_values)
        workspace.RunNetOnce(train_net.Proto())
        embedding_after_training = workspace.FetchBlob("sparse_lookup/w")
        # Verify row 0's value does not change after reset
        self.assertEqual(embedding_after_training.all(), embedding_after_init.all())


    def testSparseLookupSumPooling(self):
        record = schema.NewRecord(self.model.net, schema.Struct(
            ('sparse', schema.Struct(
                ('sparse_feature_0', schema.List(
                    schema.Scalar(np.int64,
                                  metadata=schema.Metadata(categorical_limit=1000)))),
            )),
        ))
        embedding_dim = 64
        embedding_after_pooling = self.model.SparseLookup(
            record.sparse.sparse_feature_0, [embedding_dim], 'Sum')
        self.model.output_schema = schema.Struct()
        self.assertEqual(
            schema.Scalar((np.float32, (embedding_dim, ))),
            embedding_after_pooling
        )

        train_init_net, train_net = self.get_training_nets()

        init_ops = self.assertNetContainOps(
            train_init_net,
            [
                OpSpec("UniformFill", None, None),
                OpSpec("ConstantFill", None, None),
            ]
        )
        sparse_lookup_op_spec = OpSpec(
            'SparseLengthsSum',
            [
                init_ops[0].output[0],
                record.sparse.sparse_feature_0.items(),
                record.sparse.sparse_feature_0.lengths(),
            ],
            [embedding_after_pooling()]
        )
        self.assertNetContainOps(train_net, [sparse_lookup_op_spec])

        predict_net = self.get_predict_net()
        self.assertNetContainOps(predict_net, [sparse_lookup_op_spec])

    @given(
        use_hashing=st.booleans(),
        modulo=st.integers(min_value=100, max_value=200),
        use_divide_mod=st.booleans(),
        divisor=st.integers(min_value=10, max_value=20),
    )
    def testSparseFeatureHashIdList(self, use_hashing, modulo, use_divide_mod, divisor):
        record = schema.NewRecord(
            self.model.net,
            schema.List(schema.Scalar(
                np.int64,
                metadata=schema.Metadata(categorical_limit=60000)
            ))
        )
        use_divide_mod = use_divide_mod if use_hashing is False else False
        output_schema = self.model.SparseFeatureHash(
            record,
            modulo=modulo,
            use_hashing=use_hashing,
            use_divide_mod=use_divide_mod,
            divisor=divisor,
        )

        self.model.output_schema = output_schema

        self.assertEqual(len(self.model.layers), 1)
        self.assertEqual(output_schema._items.metadata.categorical_limit,
                modulo)
        train_init_net, train_net = self.get_training_nets()
        if use_divide_mod:
            self.assertEqual(len(train_net.Proto().op), 3)
        else:
            self.assertEqual(len(train_net.Proto().op), 2)

    @given(
        use_hashing=st.booleans(),
        modulo=st.integers(min_value=100, max_value=200),
    )
    def testSparseFeatureHashIdScoreList(self, use_hashing, modulo):
        record = schema.NewRecord(self.model.net,
                schema.Map(schema.Scalar(np.int64,
                    metadata=schema.Metadata(
                        categorical_limit=60000)),
                    np.float32))

        output_schema = self.model.SparseFeatureHash(
            record,
            modulo=modulo,
            use_hashing=use_hashing)

        self.model.output_schema = output_schema

        self.assertEqual(len(self.model.layers), 1)
        self.assertEqual(output_schema._items.keys.metadata.categorical_limit,
                modulo)
        train_init_net, train_net = self.get_training_nets()

    def testSparseLookupIncorrectPositionWeightedOnIdList(self):
        '''
        Currently the implementation of SparseLookup assumed input is id_score_list
        when use PositionWeighted.
        '''
        record = schema.NewRecord(self.model.net, schema.Struct(
            ('sparse', schema.Struct(
                ('sparse_feature_0', schema.List(
                    schema.Scalar(np.int64,
                                  metadata=schema.Metadata(categorical_limit=1000)))),
            )),
        ))

        embedding_dim = 64
        with self.assertRaises(AssertionError):
            self.model.SparseLookup(
                record.sparse.sparse_feature_0, [embedding_dim], 'PositionWeighted')

    def testSparseLookupPositionWeightedOnIdList(self):
        record = schema.NewRecord(self.model.net, schema.Struct(
            ('sparse', schema.Struct(
                ('sparse_feature_0', schema.List(
                    schema.Scalar(np.int64,
                                  metadata=schema.Metadata(categorical_limit=1000)))),
            )),
        ))

        # convert id_list to id_score_list with PositionWeighted layer
        sparse_segment = record.sparse.sparse_feature_0
        pos_w_layer = self.model.PositionWeighted(sparse_segment)

        sparse_segment = schema.Map(
            keys=get_key(sparse_segment),
            values=pos_w_layer.position_weights,
            lengths_blob=sparse_segment.lengths
        )

        embedding_dim = 64
        embedding_after_pooling = self.model.SparseLookup(
            sparse_segment, [embedding_dim], 'PositionWeighted')
        self.model.output_schema = schema.Struct()
        self.assertEqual(
            schema.Scalar((np.float32, (embedding_dim, ))),
            embedding_after_pooling
        )

        train_init_net, train_net = self.get_training_nets()

        self.assertNetContainOps(
            train_init_net,
            [
                OpSpec("ConstantFill", None, None),  # position_weights/pos_w
                OpSpec("UniformFill", None, None),
                OpSpec("ConstantFill", None, None),
            ]
        )
        self.assertNetContainOps(train_net, [
            OpSpec("LengthsRangeFill", None, None),
            OpSpec("Gather", None, None),
            OpSpec("SparseLengthsWeightedSum", None, None),
        ])

        predict_net = self.get_predict_net()
        self.assertNetContainOps(predict_net, [
            OpSpec("LengthsRangeFill", None, None),
            OpSpec("Gather", None, None),
            OpSpec("SparseLengthsWeightedSum", None, None),
        ])

    def testSparseLookupPositionWeightedOnIdScoreList(self):
        record = schema.NewRecord(self.model.net, schema.Struct(
            ('sparse', schema.Struct(
                ('id_score_list_0', schema.Map(
                    schema.Scalar(
                        np.int64,
                        metadata=schema.Metadata(
                            categorical_limit=1000
                        ),
                    ),
                    np.float32
                )),
            )),
        ))

        embedding_dim = 64
        embedding_after_pooling = self.model.SparseLookup(
            record.sparse.id_score_list_0, [embedding_dim], 'PositionWeighted')
        self.model.output_schema = schema.Struct()
        self.assertEqual(
            schema.Scalar((np.float32, (embedding_dim, ))),
            embedding_after_pooling
        )

        train_init_net, train_net = self.get_training_nets()

        init_ops = self.assertNetContainOps(
            train_init_net,
            [
                OpSpec("UniformFill", None, None),
                OpSpec("ConstantFill", None, None),
            ]
        )
        sparse_lookup_op_spec = OpSpec(
            'SparseLengthsWeightedSum',
            [
                init_ops[0].output[0],
                record.sparse.id_score_list_0.values(),
                record.sparse.id_score_list_0.keys(),
                record.sparse.id_score_list_0.lengths(),
            ],
            [embedding_after_pooling()]
        )
        self.assertNetContainOps(train_net, [sparse_lookup_op_spec])

        predict_net = self.get_predict_net()
        self.assertNetContainOps(predict_net, [sparse_lookup_op_spec])

    def testSparseLookupIncorrectRecencyWeightedOnIdList(self):
        '''
        Currently the implementation of SparseLookup assumed input is id_score_list
        when use RecencyWeighted.
        '''
        record = schema.NewRecord(self.model.net, schema.Struct(
            ('sparse', schema.Struct(
                ('sparse_feature_0', schema.List(
                    schema.Scalar(np.int64,
                                  metadata=schema.Metadata(categorical_limit=1000)))),
            )),
        ))

        embedding_dim = 64
        with self.assertRaises(AssertionError):
            self.model.SparseLookup(
                record.sparse.sparse_feature_0, [embedding_dim], 'RecencyWeighted')

    def testSparseLookupRecencyWeightedOnIdScoreList(self):
        record = schema.NewRecord(self.model.net, schema.Struct(
            ('sparse', schema.Struct(
                ('id_score_list_0', schema.Map(
                    schema.Scalar(
                        np.int64,
                        metadata=schema.Metadata(
                            categorical_limit=1000
                        ),
                    ),
                    np.float32
                )),
            )),
        ))

        embedding_dim = 64
        embedding_after_pooling = self.model.SparseLookup(
            record.sparse.id_score_list_0, [embedding_dim], 'RecencyWeighted')
        self.model.output_schema = schema.Struct()
        self.assertEqual(
            schema.Scalar((np.float32, (embedding_dim, ))),
            embedding_after_pooling
        )

        train_init_net, train_net = self.get_training_nets()

        init_ops = self.assertNetContainOps(
            train_init_net,
            [
                OpSpec("UniformFill", None, None),
                OpSpec("ConstantFill", None, None),
            ]
        )
        sparse_lookup_op_spec = OpSpec(
            'SparseLengthsWeightedSum',
            [
                init_ops[0].output[0],
                record.sparse.id_score_list_0.values(),
                record.sparse.id_score_list_0.keys(),
                record.sparse.id_score_list_0.lengths(),
            ],
            [embedding_after_pooling()]
        )
        self.assertNetContainOps(train_net, [sparse_lookup_op_spec])

        predict_net = self.get_predict_net()
        self.assertNetContainOps(predict_net, [sparse_lookup_op_spec])

    def testPairwiseSimilarityWithAllEmbeddings(self):
        embedding_dim = 64
        N = 5
        record = schema.NewRecord(self.model.net, schema.Struct(
            ('all_embeddings', schema.Scalar(
                ((np.float32, (N, embedding_dim)))
            )),
        ))
        current = self.model.PairwiseSimilarity(
            record, N * N)

        self.assertEqual(
            schema.Scalar((np.float32, (N * N, ))),
            current
        )

        train_init_net, train_net = self.get_training_nets()
        self.assertNetContainOps(train_init_net, [])
        self.assertNetContainOps(train_net, [
            OpSpec("BatchMatMul", None, None),
            OpSpec("Flatten", None, None),
        ])

    def testPairwiseSimilarityWithXandYEmbeddings(self):
        embedding_dim = 64
        record = schema.NewRecord(self.model.net, schema.Struct(
            ('x_embeddings', schema.Scalar(
                ((np.float32, (5, embedding_dim)))
            )),
            ('y_embeddings', schema.Scalar(
                ((np.float32, (6, embedding_dim)))
            )),
        ))
        current = self.model.PairwiseSimilarity(
            record, 5 * 6)

        self.assertEqual(
            schema.Scalar((np.float32, (5 * 6, ))),
            current
        )

        train_init_net, train_net = self.get_training_nets()
        self.assertNetContainOps(train_init_net, [])
        self.assertNetContainOps(train_net, [
            OpSpec("BatchMatMul", None, None),
            OpSpec("Flatten", None, None),
        ])

    def testPairwiseSimilarityWithXandYEmbeddingsAndGather(self):
        embedding_dim = 64

        output_idx = [1, 3, 5]
        output_idx_blob = self.model.add_global_constant(
            str(self.model.net.NextScopedBlob('pairwise_dot_product_gather')),
            output_idx,
            dtype=np.int32,
        )
        indices_to_gather = schema.Scalar(
            (np.int32, len(output_idx)),
            output_idx_blob,
        )

        record = schema.NewRecord(self.model.net, schema.Struct(
            ('x_embeddings', schema.Scalar(
                ((np.float32, (5, embedding_dim)))
            )),
            ('y_embeddings', schema.Scalar(
                ((np.float32, (6, embedding_dim)))
            )),
            ('indices_to_gather', indices_to_gather),
        ))
        current = self.model.PairwiseSimilarity(
            record, len(output_idx))

        # This assert is not necessary,
        # output size is passed into PairwiseSimilarity
        self.assertEqual(
            schema.Scalar((np.float32, (len(output_idx), ))),
            current
        )

        train_init_net, train_net = self.get_training_nets()
        self.assertNetContainOps(train_init_net, [])
        self.assertNetContainOps(train_net, [
            OpSpec("BatchMatMul", None, None),
            OpSpec("Flatten", None, None),
            OpSpec("BatchGather", None, None),
        ])

    def testPairwiseSimilarityIncorrectInput(self):
        embedding_dim = 64
        record = schema.NewRecord(self.model.net, schema.Struct(
            ('x_embeddings', schema.Scalar(
                ((np.float32, (5, embedding_dim)))
            )),
        ))
        with self.assertRaises(AssertionError):
            self.model.PairwiseSimilarity(
                record, 25)

        record = schema.NewRecord(self.model.net, schema.Struct(
            ('all_embeddings', schema.List(np.float32))
        ))
        with self.assertRaises(AssertionError):
            self.model.PairwiseSimilarity(
                record, 25)

    def testConcat(self):
        embedding_dim = 64
        input_record = self.new_record(schema.Struct(
            ('input1', schema.Scalar((np.float32, (embedding_dim, )))),
            ('input2', schema.Scalar((np.float32, (embedding_dim, )))),
            ('input3', schema.Scalar((np.float32, (embedding_dim, )))),
        ))

        output = self.model.Concat(input_record)
        self.assertEqual(
            schema.Scalar((np.float32, ((len(input_record.fields) * embedding_dim, )))),
            output
        )

        # Note that in Concat layer we assume first dimension is batch.
        # so input is B * embedding_dim
        # add_axis=1 make it B * 1 * embedding_dim
        # concat on axis=1 make it B * N * embedding_dim
        output = self.model.Concat(input_record, axis=1, add_axis=1)
        self.assertEqual(
            schema.Scalar((np.float32, ((len(input_record.fields), embedding_dim)))),
            output
        )

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
            ('logit', schema.Scalar((np.float32, (2,)))),
            ('weight', schema.Scalar((np.float64, (1,))))
        ))
        loss = self.model.BatchLRLoss(input_record)
        self.assertEqual(schema.Scalar((np.float32, tuple())), loss)

    def testBatchLRLossWithUncertainty(self):
        input_record = self.new_record(schema.Struct(
            ('label', schema.Scalar((np.float64, (1,)))),
            ('logit', schema.Scalar((np.float32, (2,)))),
            ('weight', schema.Scalar((np.float64, (1,)))),
            ('log_variance', schema.Scalar((np.float64, (1,)))),
        ))
        loss = self.model.BatchLRLoss(input_record)
        self.assertEqual(schema.Scalar((np.float32, tuple())), loss)

    def testMarginRankLoss(self):
        input_record = self.new_record(schema.Struct(
            ('pos_prediction', schema.Scalar((np.float32, (1,)))),
            ('neg_prediction', schema.List(np.float32)),
        ))
        pos_items = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        neg_lengths = np.array([1, 2, 3], dtype=np.int32)
        neg_items = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
        schema.FeedRecord(
            input_record,
            [pos_items, neg_lengths, neg_items]
        )
        loss = self.model.MarginRankLoss(input_record)
        self.run_train_net_forward_only()
        self.assertEqual(schema.Scalar((np.float32, tuple())), loss)

    def testBPRLoss(self):
        input_record = self.new_record(schema.Struct(
            ('pos_prediction', schema.Scalar((np.float32, (1,)))),
            ('neg_prediction', schema.List(np.float32)),
        ))
        pos_items = np.array([0.8, 0.9], dtype=np.float32)
        neg_lengths = np.array([1, 2], dtype=np.int32)
        neg_items = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        schema.FeedRecord(
            input_record,
            [pos_items, neg_lengths, neg_items]
        )
        loss = self.model.BPRLoss(input_record)
        self.run_train_net_forward_only()
        self.assertEqual(schema.Scalar((np.float32, tuple())), loss)
        result = workspace.FetchBlob('bpr_loss/output')
        np.testing.assert_array_almost_equal(np.array(1.24386, dtype=np.float32), result)

    def testBatchMSELoss(self):
        input_record = self.new_record(schema.Struct(
            ('label', schema.Scalar((np.float64, (1,)))),
            ('prediction', schema.Scalar((np.float32, (2,)))),
        ))
        loss = self.model.BatchMSELoss(input_record)
        self.assertEqual(schema.Scalar((np.float32, tuple())), loss)

    def testBatchHuberLoss(self):
        input_record = self.new_record(schema.Struct(
            ('label', schema.Scalar((np.float32, (1,)))),
            ('prediction', schema.Scalar((np.float32, (2,)))),
        ))
        loss = self.model.BatchHuberLoss(input_record)
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
            None,
        )

        train_bn_spec = OpSpec(
            "SpatialBN",
            [None, init_ops[0].output[0], init_ops[1].output[0],
                init_ops[2].output[0], init_ops[3].output[0]],
            [output_blob, init_ops[2].output[0], init_ops[3].output[0], None, None],
            {'is_test': 0, 'order': 'NCHW', 'momentum': 0.9},
        )

        test_bn_spec = OpSpec(
            "SpatialBN",
            [None, init_ops[0].output[0], init_ops[1].output[0],
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
        X=hu.arrays(dims=[2, 5, 6]),
        use_layer_norm_op=st.booleans(),
    )
    def testLayerNormalization(self, X, use_layer_norm_op):
        expect = (5, 6,)
        if not use_layer_norm_op:
            X = X.reshape(10, 6)
            expect = (6,)
        input_record = self.new_record(schema.Scalar((np.float32, expect)))
        schema.FeedRecord(input_record, [X])
        ln_output = self.model.LayerNormalization(
            input_record, use_layer_norm_op=use_layer_norm_op
        )
        self.assertEqual(schema.Scalar((np.float32, expect)), ln_output)
        self.model.output_schema = schema.Struct()

        train_init_net, train_net = self.get_training_nets(add_constants=True)
        workspace.RunNetOnce(train_init_net)
        workspace.RunNetOnce(train_net)

    @given(
        X=hu.arrays(dims=[5, 2]),
        num_to_collect=st.integers(min_value=1, max_value=10),
    )
    def testLastNWindowCollector(self, X, num_to_collect):
        input_record = self.new_record(schema.Scalar(np.float32))
        schema.FeedRecord(input_record, [X])
        last_n = self.model.LastNWindowCollector(input_record, num_to_collect)
        self.run_train_net_forward_only()
        output_record = schema.FetchRecord(last_n.last_n)
        start = max(0, 5 - num_to_collect)
        npt.assert_array_equal(X[start:], output_record())
        num_visited = schema.FetchRecord(last_n.num_visited)
        npt.assert_array_equal([5], num_visited())

    @given(
        X=hu.arrays(dims=[5, 2]),
        num_to_collect=st.integers(min_value=3, max_value=3),
    )
    @settings(deadline=1000)
    def testReservoirSamplingWithID(self, X, num_to_collect):
        ID = np.array([1, 2, 3, 1, 2], dtype=np.int64)
        input_record = self.new_record(
            schema.Struct(
                ('record', schema.Struct(
                    ('dense', schema.Scalar()),
                )),
                ('object_id', schema.Scalar(np.int64)),
            )
        )
        schema.FeedRecord(input_record, [X, ID])
        packed_record = self.model.PackRecords(
            input_record.record, 1, fields=input_record.record.field_names())
        reservoir_input = schema.Struct(
            ('data', packed_record),
            ('object_id', input_record.object_id),
        )
        reservoir = self.model.ReservoirSampling(
            reservoir_input, num_to_collect)
        self.model.output_schema = schema.Struct()
        train_init_net, train_net = \
            layer_model_instantiator.generate_training_nets_forward_only(
                self.model)
        workspace.RunNetOnce(train_init_net)
        workspace.CreateNet(train_net)
        workspace.RunNet(train_net.Proto().name, num_iter=2)
        num_visited = schema.FetchRecord(reservoir.num_visited)
        npt.assert_array_equal([3], num_visited())
        for param in self.model.params:
            serialized = workspace.SerializeBlob(str(param))
            workspace.DeserializeBlob(str(param), serialized)
        ID = np.array([3, 5, 3, 3, 5], dtype=np.int64)
        schema.FeedRecord(input_record.object_id, [ID])
        workspace.RunNet(train_net.Proto().name, num_iter=2)
        num_visited = schema.FetchRecord(reservoir.num_visited)
        npt.assert_array_equal([2], num_visited())

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

    def testUniformSamplingWithIncorrectSampleSize(self):
        input_record = self.new_record(schema.Scalar(np.int32))
        num_samples = 200
        num_elements = 100
        with self.assertRaises(AssertionError):
            self.model.UniformSampling(input_record, num_samples, num_elements)

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
        indices_blob = self.model.MapToRange(input_record,
                                             max_index=100).indices
        self.model.output_schema = schema.Struct()

        train_init_net, train_net = self.get_training_nets()

        schema.FeedRecord(
            input_record,
            [np.array([10, 3, 20, 99, 15, 11, 3, 11], dtype=np.int32)]
        )
        workspace.RunNetOnce(train_init_net)
        workspace.RunNetOnce(train_net)
        indices = workspace.FetchBlob(indices_blob())
        np.testing.assert_array_equal(
            np.array([1, 2, 3, 4, 5, 6, 2, 6], dtype=np.int32),
            indices
        )

        schema.FeedRecord(
            input_record,
            [np.array([10, 3, 23, 35, 60, 15, 10, 15], dtype=np.int32)]
        )
        workspace.RunNetOnce(train_net)
        indices = workspace.FetchBlob(indices_blob())
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
        indices = workspace.FetchBlob(indices_blob())
        np.testing.assert_array_equal(
            np.array([1, 2, 7, 8, 9, 5, 0], dtype=np.int32),
            indices
        )

        schema.FeedRecord(
            input_record,
            [np.array([10, 3, 23, 15, 101, 115], dtype=np.int32)]
        )
        workspace.RunNetOnce(eval_net)
        indices = workspace.FetchBlob(indices_blob())
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
        indices = workspace.FetchBlob(indices_blob())
        np.testing.assert_array_equal(
            np.array([2, 2, 3, 7, 0, 8, 9, 5, 0], dtype=np.int32),
            indices
        )

    def testSelectRecordByContext(self):
        float_features = self.model.input_feature_schema.float_features

        float_array = np.array([1.0, 2.0], dtype=np.float32)

        schema.FeedRecord(float_features, [float_array])

        with Tags(Tags.EXCLUDE_FROM_PREDICTION):
            log_float_features = self.model.Log(float_features, 1)
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
                out_record(),
                broadcast=1)
        normalized = self.model.Functional(
            self.model.input_feature_schema.float_features, 1,
            normalize, name="normalizer")

        # Attach metadata to one of the outputs and use it in FC
        normalized.set_type((np.float32, 32))
        self.model.output_schema = self.model.FC(normalized, 2)

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
                self.model.input_feature_schema.float_features, mean),
            1, broadcast=1)
        # Attach metadata to one of the outputs and use it in FC
        normalized.set_type((np.float32, (32,)))
        self.model.output_schema = self.model.FC(normalized, 2)

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
        assert softsign.field_type().base == np.float32
        assert softsign.field_type().shape == (32,)
        self.model.output_schema = self.model.FC(softsign, 2)

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

    def testHalfToFloatTypeInference(self):
        input = self.new_record(schema.Scalar((np.float32, (32,))))

        output = self.model.FloatToHalf(input, 1)
        assert output.field_type().base == np.float16
        assert output.field_type().shape == (32, )

        output = self.model.HalfToFloat(output, 1)
        assert output.field_type().base == np.float32
        assert output.field_type().shape == (32, )

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

    def testFunctionalLayerSameOperatorOutputNames(self):
        Con1 = self.model.ConstantFill([], 1, value=1)
        Con2 = self.model.ConstantFill([], 1, value=2)
        self.assertNotEqual(str(Con1), str(Con2))

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
        dropout_for_eval=st.booleans(),
    )
    def testDropout(self, X, dropout_for_eval):
        input_record = self.new_record(schema.Scalar((np.float32, (1,))))
        schema.FeedRecord(input_record, [X])
        d_output = self.model.Dropout(
            input_record,
            dropout_for_eval=dropout_for_eval
        )
        self.assertEqual(schema.Scalar((np.float32, (1,))), d_output)
        self.model.output_schema = schema.Struct()

        train_init_net, train_net = self.get_training_nets()

        input_blob = input_record.field_blobs()[0]
        output_blob = d_output.field_blobs()[0]

        with_d_spec = OpSpec(
            "Dropout",
            [input_blob],
            [output_blob, None],
            {'is_test': 0, 'ratio': 0.5}
        )

        without_d_spec = OpSpec(
            "Dropout",
            [input_blob],
            [output_blob, None],
            {'is_test': 1, 'ratio': 0.5}
        )

        self.assertNetContainOps(
            train_net,
            [with_d_spec]
        )

        eval_net = self.get_eval_net()
        predict_net = self.get_predict_net()

        if dropout_for_eval:
            self.assertNetContainOps(
                eval_net,
                [with_d_spec]
            )
            self.assertNetContainOps(
                predict_net,
                [with_d_spec]
            )
        else:
            self.assertNetContainOps(
                eval_net,
                [without_d_spec]
            )
            self.assertNetContainOps(
                predict_net,
                [without_d_spec]
            )

        workspace.RunNetOnce(train_init_net)
        workspace.RunNetOnce(train_net)

        schema.FeedRecord(input_record, [X])
        workspace.RunNetOnce(eval_net)

        schema.FeedRecord(input_record, [X])
        workspace.RunNetOnce(predict_net)

    @given(
        num_inputs=st.integers(1, 3),
        batch_size=st.integers(5, 10)
    )
    def testMergeIdListsLayer(self, num_inputs, batch_size):
        inputs = []
        for _ in range(num_inputs):
            lengths = np.random.randint(5, size=batch_size).astype(np.int32)
            size = lengths.sum()
            values = np.random.randint(1, 10, size=size).astype(np.int64)
            inputs.append(lengths)
            inputs.append(values)
        input_schema = schema.Tuple(
            *[schema.List(
                schema.Scalar(dtype=np.int64, metadata=schema.Metadata(
                    categorical_limit=20
                ))) for _ in range(num_inputs)]
        )

        input_record = schema.NewRecord(self.model.net, input_schema)
        schema.FeedRecord(input_record, inputs)
        output_schema = self.model.MergeIdLists(input_record)
        assert schema.equal_schemas(
            output_schema, IdList,
            check_field_names=False)

    @given(
        batch_size=st.integers(min_value=2, max_value=10),
        input_dims=st.integers(min_value=5, max_value=10),
        output_dims=st.integers(min_value=5, max_value=10),
        bandwidth=st.floats(min_value=0.1, max_value=5),
    )
    def testRandomFourierFeatures(self, batch_size, input_dims, output_dims, bandwidth):

        def _rff_hypothesis_test(rff_output, X, W, b, scale):
            '''
            Runs hypothesis test for Semi Random Features layer.

            Inputs:
                rff_output -- output of net after running random fourier features layer
                X -- input data
                W -- weight parameter from train_init_net
                b -- bias parameter from train_init_net
                scale -- value by which to scale the output vector
            '''
            output = workspace.FetchBlob(rff_output)
            output_ref = scale * np.cos(np.dot(X, np.transpose(W)) + b)
            npt.assert_allclose(output, output_ref, rtol=1e-3, atol=1e-3)

        X = np.random.random((batch_size, input_dims)).astype(np.float32)
        scale = np.sqrt(2.0 / output_dims)
        input_record = self.new_record(schema.Scalar((np.float32, (input_dims,))))
        schema.FeedRecord(input_record, [X])
        input_blob = input_record.field_blobs()[0]
        rff_output = self.model.RandomFourierFeatures(input_record,
                                                      output_dims,
                                                      bandwidth)
        self.model.output_schema = schema.Struct()

        self.assertEqual(
            schema.Scalar((np.float32, (output_dims, ))),
            rff_output
        )

        train_init_net, train_net = self.get_training_nets()

        # Init net assertions
        init_ops_list = [
            OpSpec("GaussianFill", None, None),
            OpSpec("UniformFill", None, None),
        ]
        init_ops = self._test_net(train_init_net, init_ops_list)
        W = workspace.FetchBlob(self.model.layers[0].w)
        b = workspace.FetchBlob(self.model.layers[0].b)

        # Operation specifications
        fc_spec = OpSpec("FC", [input_blob, init_ops[0].output[0],
                         init_ops[1].output[0]], None)
        cosine_spec = OpSpec("Cos", None, None)
        scale_spec = OpSpec("Scale", None, rff_output.field_blobs(),
                            {'scale': scale})
        ops_list = [
            fc_spec,
            cosine_spec,
            scale_spec
        ]

        # Train net assertions
        self._test_net(train_net, ops_list)
        _rff_hypothesis_test(rff_output(), X, W, b, scale)

        # Eval net assertions
        eval_net = self.get_eval_net()
        self._test_net(eval_net, ops_list)
        _rff_hypothesis_test(rff_output(), X, W, b, scale)

        # Predict net assertions
        predict_net = self.get_predict_net()
        self._test_net(predict_net, ops_list)
        _rff_hypothesis_test(rff_output(), X, W, b, scale)

    @given(
        batch_size=st.integers(min_value=2, max_value=10),
        input_dims=st.integers(min_value=5, max_value=10),
        output_dims=st.integers(min_value=5, max_value=10),
        s=st.integers(min_value=0, max_value=3),
        scale=st.floats(min_value=0.1, max_value=5),
        set_weight_as_global_constant=st.booleans()
    )
    def testArcCosineFeatureMap(self, batch_size, input_dims, output_dims, s, scale,
                                set_weight_as_global_constant):

        def _arc_cosine_hypothesis_test(ac_output, X, W, b, s):
            '''
            Runs hypothesis test for Arc Cosine layer.

            Inputs:
                ac_output -- output of net after running arc cosine layer
                X -- input data
                W -- weight parameter from train_init_net
                b -- bias parameter from train_init_net
                s -- degree parameter
            '''
            # Get output from net
            net_output = workspace.FetchBlob(ac_output)

            # Computing output directly
            x_rand = np.matmul(X, np.transpose(W)) + b
            x_pow = np.power(x_rand, s)
            if s > 0:
                h_rand_features = np.piecewise(x_rand,
                                               [x_rand <= 0, x_rand > 0],
                                               [0, 1])
            else:
                h_rand_features = np.piecewise(x_rand,
                                               [x_rand <= 0, x_rand > 0],
                                               [0, lambda x: x / (1 + x)])
            output_ref = np.multiply(x_pow, h_rand_features)

            # Comparing net output and computed output
            npt.assert_allclose(net_output, output_ref, rtol=1e-3, atol=1e-3)

        X = np.random.normal(size=(batch_size, input_dims)).astype(np.float32)
        input_record = self.new_record(schema.Scalar((np.float32, (input_dims,))))
        schema.FeedRecord(input_record, [X])
        input_blob = input_record.field_blobs()[0]

        ac_output = self.model.ArcCosineFeatureMap(
            input_record,
            output_dims,
            s=s,
            scale=scale,
            set_weight_as_global_constant=set_weight_as_global_constant
        )
        self.model.output_schema = schema.Struct()
        self.assertEqual(
            schema.Scalar((np.float32, (output_dims, ))),
            ac_output
        )

        train_init_net, train_net = self.get_training_nets()

        # Run create_init_net to initialize the global constants, and W and b
        workspace.RunNetOnce(train_init_net)
        workspace.RunNetOnce(self.model.create_init_net(name='init_net'))

        if set_weight_as_global_constant:
            W = workspace.FetchBlob(
                self.model.global_constants['arc_cosine_feature_map_fixed_rand_W']
            )
            b = workspace.FetchBlob(
                self.model.global_constants['arc_cosine_feature_map_fixed_rand_b']
            )
        else:
            W = workspace.FetchBlob(self.model.layers[0].random_w)
            b = workspace.FetchBlob(self.model.layers[0].random_b)

        # Operation specifications
        fc_spec = OpSpec("FC", [input_blob, None, None], None)
        softsign_spec = OpSpec("Softsign", None, None)
        relu_spec = OpSpec("Relu", None, None)
        relu_spec_output = OpSpec("Relu", None, ac_output.field_blobs())
        pow_spec = OpSpec("Pow", None, None, {'exponent': float(s - 1)})
        mul_spec = OpSpec("Mul", None, ac_output.field_blobs())

        if s == 0:
            ops_list = [
                fc_spec,
                softsign_spec,
                relu_spec_output,
            ]
        elif s == 1:
            ops_list = [
                fc_spec,
                relu_spec_output,
            ]
        else:
            ops_list = [
                fc_spec,
                relu_spec,
                pow_spec,
                mul_spec,
            ]

        # Train net assertions
        self._test_net(train_net, ops_list)
        _arc_cosine_hypothesis_test(ac_output(), X, W, b, s)

        # Eval net assertions
        eval_net = self.get_eval_net()
        self._test_net(eval_net, ops_list)
        _arc_cosine_hypothesis_test(ac_output(), X, W, b, s)

        # Predict net assertions
        predict_net = self.get_predict_net()
        self._test_net(predict_net, ops_list)
        _arc_cosine_hypothesis_test(ac_output(), X, W, b, s)

    @given(
        batch_size=st.integers(min_value=2, max_value=10),
        input_dims=st.integers(min_value=5, max_value=10),
        output_dims=st.integers(min_value=5, max_value=10),
        s=st.integers(min_value=0, max_value=3),
        scale=st.floats(min_value=0.1, max_value=5),
        set_weight_as_global_constant=st.booleans(),
        use_struct_input=st.booleans(),
    )
    def testSemiRandomFeatures(self, batch_size, input_dims, output_dims, s, scale,
                               set_weight_as_global_constant, use_struct_input):

        def _semi_random_hypothesis_test(srf_output, X_full, X_random, rand_w,
                                         rand_b, s):
            '''
            Runs hypothesis test for Semi Random Features layer.

            Inputs:
                srf_output -- output of net after running semi random features layer
                X_full -- full input data
                X_random -- random-output input data
                rand_w -- random-initialized weight parameter from train_init_net
                rand_b -- random-initialized bias parameter from train_init_net
                s -- degree parameter

            '''
            # Get output from net
            net_output = workspace.FetchBlob(srf_output)

            # Fetch learned parameter blobs
            learned_w = workspace.FetchBlob(self.model.layers[0].learned_w)
            learned_b = workspace.FetchBlob(self.model.layers[0].learned_b)

            # Computing output directly
            x_rand = np.matmul(X_random, np.transpose(rand_w)) + rand_b
            x_learn = np.matmul(X_full, np.transpose(learned_w)) + learned_b
            x_pow = np.power(x_rand, s)
            if s > 0:
                h_rand_features = np.piecewise(x_rand,
                                               [x_rand <= 0, x_rand > 0],
                                               [0, 1])
            else:
                h_rand_features = np.piecewise(x_rand,
                                               [x_rand <= 0, x_rand > 0],
                                               [0, lambda x: x / (1 + x)])
            output_ref = np.multiply(np.multiply(x_pow, h_rand_features), x_learn)

            # Comparing net output and computed output
            npt.assert_allclose(net_output, output_ref, rtol=1e-3, atol=1e-3)

        X_full = np.random.normal(size=(batch_size, input_dims)).astype(np.float32)
        if use_struct_input:
            X_random = np.random.normal(size=(batch_size, input_dims)).\
                astype(np.float32)
            input_data = [X_full, X_random]
            input_record = self.new_record(schema.Struct(
                ('full', schema.Scalar(
                    (np.float32, (input_dims,))
                )),
                ('random', schema.Scalar(
                    (np.float32, (input_dims,))
                ))
            ))
        else:
            X_random = X_full
            input_data = [X_full]
            input_record = self.new_record(schema.Scalar(
                (np.float32, (input_dims,))
            ))

        schema.FeedRecord(input_record, input_data)
        srf_output = self.model.SemiRandomFeatures(
            input_record,
            output_dims,
            s=s,
            scale_random=scale,
            scale_learned=scale,
            set_weight_as_global_constant=set_weight_as_global_constant
        )

        self.model.output_schema = schema.Struct()

        self.assertEqual(
            schema.Struct(
                ('full', schema.Scalar(
                    (np.float32, (output_dims,))
                )),
                ('random', schema.Scalar(
                    (np.float32, (output_dims,))
                ))
            ),
            srf_output
        )

        init_ops_list = [
            OpSpec("GaussianFill", None, None),
            OpSpec("UniformFill", None, None),
            OpSpec("GaussianFill", None, None),
            OpSpec("UniformFill", None, None),
        ]
        train_init_net, train_net = self.get_training_nets()

        # Need to run to initialize the global constants for layer
        workspace.RunNetOnce(self.model.create_init_net(name='init_net'))

        if set_weight_as_global_constant:
            # If weight params are global constants, they won't be in train_init_net
            init_ops = self._test_net(train_init_net, init_ops_list[:2])
            rand_w = workspace.FetchBlob(
                self.model.global_constants['semi_random_features_fixed_rand_W']
            )
            rand_b = workspace.FetchBlob(
                self.model.global_constants['semi_random_features_fixed_rand_b']
            )

            # Operation specifications
            fc_random_spec = OpSpec("FC", [None, None, None], None)
            fc_learned_spec = OpSpec("FC", [None, init_ops[0].output[0],
                                     init_ops[1].output[0]], None)
        else:
            init_ops = self._test_net(train_init_net, init_ops_list)
            rand_w = workspace.FetchBlob(self.model.layers[0].random_w)
            rand_b = workspace.FetchBlob(self.model.layers[0].random_b)

            # Operation specifications
            fc_random_spec = OpSpec("FC", [None, init_ops[0].output[0],
                                    init_ops[1].output[0]], None)
            fc_learned_spec = OpSpec("FC", [None, init_ops[2].output[0],
                                     init_ops[3].output[0]], None)

        softsign_spec = OpSpec("Softsign", None, None)
        relu_spec = OpSpec("Relu", None, None)
        relu_output_spec = OpSpec("Relu", None, srf_output.random.field_blobs())
        pow_spec = OpSpec("Pow", None, None, {'exponent': float(s - 1)})
        mul_interim_spec = OpSpec("Mul", None, srf_output.random.field_blobs())
        mul_spec = OpSpec("Mul", None, srf_output.full.field_blobs())

        if s == 0:
            ops_list = [
                fc_learned_spec,
                fc_random_spec,
                softsign_spec,
                relu_output_spec,
                mul_spec,
            ]
        elif s == 1:
            ops_list = [
                fc_learned_spec,
                fc_random_spec,
                relu_output_spec,
                mul_spec,
            ]
        else:
            ops_list = [
                fc_learned_spec,
                fc_random_spec,
                relu_spec,
                pow_spec,
                mul_interim_spec,
                mul_spec,
            ]

        # Train net assertions
        self._test_net(train_net, ops_list)
        _semi_random_hypothesis_test(srf_output.full(), X_full, X_random,
                                     rand_w, rand_b, s)

        # Eval net assertions
        eval_net = self.get_eval_net()
        self._test_net(eval_net, ops_list)
        _semi_random_hypothesis_test(srf_output.full(), X_full, X_random,
                                     rand_w, rand_b, s)

        # Predict net assertions
        predict_net = self.get_predict_net()
        self._test_net(predict_net, ops_list)
        _semi_random_hypothesis_test(srf_output.full(), X_full, X_random,
                                     rand_w, rand_b, s)

    def testConv(self):
        batch_size = 50
        H = 1
        W = 10
        C = 50
        output_dims = 32
        kernel_h = 1
        kernel_w = 3
        stride_h = 1
        stride_w = 1
        pad_t = 0
        pad_b = 0
        pad_r = None
        pad_l = None

        input_record = self.new_record(schema.Scalar((np.float32, (H, W, C))))
        X = np.random.random((batch_size, H, W, C)).astype(np.float32)
        schema.FeedRecord(input_record, [X])
        conv = self.model.Conv(
            input_record,
            output_dims,
            kernel_h=kernel_h,
            kernel_w=kernel_w,
            stride_h=stride_h,
            stride_w=stride_w,
            pad_t=pad_t,
            pad_b=pad_b,
            pad_r=pad_r,
            pad_l=pad_l,
            order='NHWC'
        )

        self.assertEqual(
            schema.Scalar((np.float32, (output_dims,))),
            conv
        )

        self.run_train_net_forward_only()
        output_record = schema.FetchRecord(conv)
        # check the number of output channels is the same as input in this example
        assert output_record.field_types()[0].shape == (H, W, output_dims)
        assert output_record().shape == (batch_size, H, W, output_dims)

        train_init_net, train_net = self.get_training_nets()
        # Init net assertions
        init_ops = self.assertNetContainOps(
            train_init_net,
            [
                OpSpec("XavierFill", None, None),
                OpSpec("ConstantFill", None, None),
            ]
        )
        conv_spec = OpSpec(
            "Conv",
            [
                input_record.field_blobs()[0],
                init_ops[0].output[0],
                init_ops[1].output[0],
            ],
            conv.field_blobs()
        )

        # Train net assertions
        self.assertNetContainOps(train_net, [conv_spec])

        # Predict net assertions
        predict_net = self.get_predict_net()
        self.assertNetContainOps(predict_net, [conv_spec])

        # Eval net assertions
        eval_net = self.get_eval_net()
        self.assertNetContainOps(eval_net, [conv_spec])

    @given(
        num=st.integers(min_value=10, max_value=100),
        feed_weight=st.booleans(),
        use_inv_var_parameterization=st.booleans(),
        use_log_barrier=st.booleans(),
        enable_diagnose=st.booleans(),
        **hu.gcs
    )
    @settings(deadline=1000)
    def testAdaptiveWeight(
        self, num, feed_weight, use_inv_var_parameterization, use_log_barrier,
        enable_diagnose, gc, dc
    ):
        input_record = self.new_record(schema.RawTuple(num))
        data = np.random.random(num)
        schema.FeedRecord(
            input_record, [np.array(x).astype(np.float32) for x in data]
        )
        weights = np.random.random(num) if feed_weight else None
        result = self.model.AdaptiveWeight(
            input_record,
            weights=weights,
            estimation_method=(
                'inv_var' if use_inv_var_parameterization else 'log_std'
            ),
            pos_optim_method=(
                'log_barrier' if use_log_barrier else 'pos_grad_proj'
            ),
            enable_diagnose=enable_diagnose
        )
        train_init_net, train_net = self.get_training_nets(True)
        workspace.RunNetOnce(train_init_net)
        workspace.RunNetOnce(train_net)
        result = workspace.FetchBlob(result())
        if not feed_weight:
            weights = np.array([1. / num for _ in range(num)])
        expected = np.sum(weights * data + 0.5 * np.log(1. / 2. / weights))
        npt.assert_allclose(expected, result, atol=1e-4, rtol=1e-4)
        if enable_diagnose:
            assert len(self.model.ad_hoc_plot_blobs) == num
            reconst_weights_from_ad_hoc = np.array(
                [workspace.FetchBlob(b) for b in self.model.ad_hoc_plot_blobs]
            ).flatten()
            npt.assert_allclose(
                reconst_weights_from_ad_hoc, weights, atol=1e-4, rtol=1e-4
            )
        else:
            assert len(self.model.ad_hoc_plot_blobs) == 0

    @given(num=st.integers(min_value=10, max_value=100), **hu.gcs)
    def testConstantWeight(self, num, gc, dc):
        input_record = self.new_record(schema.RawTuple(num))
        data = np.random.random(num)
        schema.FeedRecord(
            input_record, [np.array(x).astype(np.float32) for x in data]
        )
        weights = np.random.random(num)
        result = self.model.ConstantWeight(input_record, weights=weights)
        train_init_net, train_net = self.get_training_nets(True)
        workspace.RunNetOnce(train_init_net)
        workspace.RunNetOnce(train_net)
        result = workspace.FetchBlob(result())
        expected = np.sum(weights * data)
        npt.assert_allclose(expected, result, atol=1e-4, rtol=1e-4)

    @given(**hu.gcs)
    @settings(deadline=10000)
    def testHomotopyWeight(self, gc, dc):
        input_record = self.new_record(schema.RawTuple(2))
        data = np.random.random(2)
        schema.FeedRecord(
            input_record, [np.array(x).astype(np.float32) for x in data]
        )
        # ensure: quad_life > 2 * half_life
        half_life = int(np.random.random() * 1e2 + 1)
        quad_life = int(np.random.random() * 1e3 + 2 * half_life + 1)
        min_weight = np.random.random()
        max_weight = np.random.random() + min_weight + 1e-5
        result = self.model.HomotopyWeight(
            input_record,
            min_weight=min_weight,
            max_weight=max_weight,
            half_life=half_life,
            quad_life=quad_life,
        )
        train_init_net, train_net = self.get_training_nets(True)
        workspace.RunNetOnce(train_init_net)
        workspace.CreateNet(train_net)
        workspace.RunNet(train_net.Name(), num_iter=half_life)
        half_life_result = workspace.FetchBlob(result())
        workspace.RunNet(train_net.Name(), num_iter=quad_life - half_life)
        quad_life_result = workspace.FetchBlob(result())

        alpha = (min_weight + max_weight) / 2.
        beta = (min_weight + max_weight) / 2.
        expected_half_life_result = alpha * data[0] + beta * data[1]
        alpha = (3 * min_weight + max_weight) / 4.
        beta = (min_weight + 3 * max_weight) / 4.
        expected_quad_life_result = alpha * data[0] + beta * data[1]
        npt.assert_allclose(
            expected_half_life_result, half_life_result, atol=1e-2, rtol=1e-2
        )
        npt.assert_allclose(
            expected_quad_life_result, quad_life_result, atol=1e-2, rtol=1e-2
        )

    def _testLabelSmooth(self, categories, binary_prob_label, bsz):
        label = self.new_record(schema.Scalar((np.float32, (1, ))))
        label_np = np.random.randint(categories, size=bsz).astype(np.float32)
        schema.FeedRecord(label, [label_np])
        smooth_matrix_shape = (
            2 if binary_prob_label else (categories, categories)
        )
        smooth_matrix = np.random.random(smooth_matrix_shape)
        smoothed_label = self.model.LabelSmooth(label, smooth_matrix)
        train_init_net, train_net = self.get_training_nets(True)
        workspace.RunNetOnce(train_init_net)
        workspace.RunNetOnce(train_net)
        smoothed_label_np = workspace.FetchBlob(smoothed_label())
        if binary_prob_label:
            expected = np.array(
                [
                    smooth_matrix[0] if x == 0.0 else smooth_matrix[1]
                    for x in label_np
                ]
            )
        else:
            expected = np.array([smooth_matrix[int(x)] for x in label_np])
        npt.assert_allclose(expected, smoothed_label_np, atol=1e-4, rtol=1e-4)

    @given(
        categories=st.integers(min_value=2, max_value=10),
        bsz=st.integers(min_value=10, max_value=100),
        **hu.gcs
    )
    def testLabelSmoothForCategoricalLabel(self, categories, bsz, gc, dc):
        self._testLabelSmooth(categories, False, bsz)

    @given(
        bsz=st.integers(min_value=10, max_value=100),
        **hu.gcs
    )
    def testLabelSmoothForBinaryProbLabel(self, bsz, gc, dc):
        self._testLabelSmooth(2, True, bsz)

    @given(
        num_inputs=st.integers(min_value=2, max_value=10),
        batch_size=st.integers(min_value=2, max_value=10),
        input_dim=st.integers(min_value=5, max_value=10),
        seed=st.integers(1, 10),
    )
    def testBlobWeightedSum(self, num_inputs, batch_size, input_dim, seed):

        def get_blob_weighted_sum():
            weights = []
            for i in range(num_inputs):
                w_blob_name = 'blob_weighted_sum/w_{0}'.format(i)
                assert workspace.HasBlob(w_blob_name), (
                    "cannot fine blob {}".format(w_blob_name)
                )
                w = workspace.FetchBlob(w_blob_name)
                weights.append(w)

            result = np.sum([
                input_data[idx] * weights[idx] for idx in range(num_inputs)
            ], axis=0)
            return result

        np.random.seed(seed)
        expected_output_schema = schema.Scalar((np.float32, (input_dim,)))
        input_schema = schema.Tuple(
            *[expected_output_schema for _ in range(num_inputs)]
        )
        input_data = [
            np.random.random((batch_size, input_dim)).astype(np.float32)
            for _ in range(num_inputs)
        ]
        input_record = self.new_record(input_schema)
        schema.FeedRecord(input_record, input_data)

        # test output schema
        ws_output = self.model.BlobWeightedSum(input_record)
        self.assertEqual(len(self.model.layers), 1)
        assert schema.equal_schemas(ws_output, expected_output_schema)

        # test train net
        train_init_net, train_net = self.get_training_nets()
        workspace.RunNetOnce(train_init_net)
        workspace.RunNetOnce(train_net)
        output = workspace.FetchBlob(ws_output())
        npt.assert_almost_equal(get_blob_weighted_sum(), output, decimal=5)

        self.run_train_net_forward_only()
        output = workspace.FetchBlob(ws_output())
        npt.assert_almost_equal(get_blob_weighted_sum(), output, decimal=5)

        # test eval net
        eval_net = self.get_eval_net()
        workspace.RunNetOnce(eval_net)
        output = workspace.FetchBlob(ws_output())
        npt.assert_almost_equal(get_blob_weighted_sum(), output, decimal=5)

        # test pred net
        pred_net = self.get_predict_net()
        workspace.RunNetOnce(pred_net)
        output = workspace.FetchBlob(ws_output())
        npt.assert_almost_equal(get_blob_weighted_sum(), output, decimal=5)

    def testFeatureSparseToDenseGetAccessedFeatures(self):
        float_features_column = "float_features"
        float_features_type = "FLOAT"
        float_features_ids = [1, 2, 3]

        id_list_features_column = "id_list_features"
        id_list_features_type = "ID_LIST"
        id_list_features_ids = [4, 5, 6]

        id_score_list_features_column = "id_score_list_features"
        id_score_list_features_type = "ID_SCORE_LIST"
        id_score_list_features_ids = [7, 8 , 9]

        feature_names = ["a", "b", "c"]

        input_record = self.new_record(schema.Struct(
            (float_features_column, schema.Map(np.int32, np.float32)),
            (id_list_features_column,
                schema.Map(np.int32, schema.List(np.int64))),
            (id_score_list_features_column,
                schema.Map(np.int32, schema.Map(np.int64, np.float32))),
        ))

        input_specs = [
            (
                float_features_column,
                schema.FeatureSpec(
                    feature_type=float_features_type,
                    feature_ids=float_features_ids,
                    feature_names=feature_names,
                ),
            ),
            (
                id_list_features_column,
                schema.FeatureSpec(
                    feature_type=id_list_features_type,
                    feature_ids=id_list_features_ids,
                    feature_names=feature_names,
                ),
            ),
            (
                id_score_list_features_column,
                schema.FeatureSpec(
                    feature_type=id_score_list_features_type,
                    feature_ids=id_score_list_features_ids,
                    feature_names=feature_names,
                ),
            ),
        ]

        self.model.FeatureSparseToDense(input_record, input_specs)

        expected_accessed_features = {
            float_features_column: [
                AccessedFeatures(float_features_type, set(float_features_ids))],
            id_list_features_column: [
                AccessedFeatures(id_list_features_type, set(id_list_features_ids))],
            id_score_list_features_column: [
                AccessedFeatures(id_score_list_features_type, set(id_score_list_features_ids))],
        }

        self.assertEqual(len(self.model.layers), 1)
        self.assertEqual(
            self.model.layers[0].get_accessed_features(),
            expected_accessed_features
        )

    def test_get_key(self):
        def _is_id_list(input_record):
            return almost_equal_schemas(input_record, IdList)


        def _is_id_score_list(input_record):
            return almost_equal_schemas(input_record,
                                        IdScoreList,
                                        check_field_types=False)

        def old_get_sparse_key_logic(input_record):
            if _is_id_list(input_record):
                sparse_key = input_record.items()
            elif _is_id_score_list(input_record):
                sparse_key = input_record.keys()
            else:
                raise NotImplementedError()
            return sparse_key

        id_score_list_record = schema.NewRecord(
            self.model.net,
            schema.Map(
                schema.Scalar(
                    np.int64,
                    metadata=schema.Metadata(
                        categorical_limit=1000
                    ),
                ),
                np.float32
            )
        )

        self.assertEqual(
            get_key(id_score_list_record)(),
            old_get_sparse_key_logic(id_score_list_record)
        )

        id_list_record = schema.NewRecord(
            self.model.net,
            schema.List(
                schema.Scalar(
                    np.int64,
                    metadata=schema.Metadata(categorical_limit=1000)
                )
            )
        )

        self.assertEqual(
            get_key(id_list_record)(),
            old_get_sparse_key_logic(id_list_record)
        )

    def testSparseLookupWithAttentionWeightOnIdScoreList(self):
        record = schema.NewRecord(
            self.model.net,
            schema.Map(
                schema.Scalar(
                    np.int64,
                    metadata=schema.Metadata(categorical_limit=1000),
                ),
                np.float32,
            ),
        )
        embedding_dim = 64
        embedding_after_pooling = self.model.SparseLookup(
            record, [embedding_dim], "Sum", use_external_weights=True
        )
        self.model.output_schema = schema.Struct()
        self.assertEqual(
            schema.Scalar((np.float32, (embedding_dim,))), embedding_after_pooling
        )

        train_init_net, train_net = self.get_training_nets()

        init_ops = self.assertNetContainOps(
            train_init_net,
            [OpSpec("UniformFill", None, None), OpSpec("ConstantFill", None, None)],
        )
        sparse_lookup_op_spec = OpSpec(
            "SparseLengthsWeightedSum",
            [
                init_ops[0].output[0],
                record.values(),
                record.keys(),
                record.lengths(),
            ],
            [embedding_after_pooling()],
        )
        self.assertNetContainOps(train_net, [sparse_lookup_op_spec])

        predict_net = self.get_predict_net()
        self.assertNetContainOps(predict_net, [sparse_lookup_op_spec])

    def testSparseItemwiseDropoutWithReplacement(self):
        input_record = schema.NewRecord(self.model.net, IdList)
        self.model.output_schema = schema.Struct()

        lengths_blob = input_record.field_blobs()[0]
        values_blob = input_record.field_blobs()[1]
        lengths = np.array([1] * 10).astype(np.int32)
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.int64)
        workspace.FeedBlob(lengths_blob, lengths)
        workspace.FeedBlob(values_blob, values)

        out = self.model.SparseItemwiseDropoutWithReplacement(
            input_record, 0.0, 0.5, 1.0, -1, output_names_or_num=1)
        self.assertEqual(schema.List(schema.Scalar(np.int64,)), out)

        train_init_net, train_net = self.get_training_nets()
        eval_net = self.get_eval_net()
        predict_net = self.get_predict_net()

        workspace.RunNetOnce(train_init_net)
        workspace.RunNetOnce(train_net)
        out_values = workspace.FetchBlob(out.items())
        out_lengths = workspace.FetchBlob(out.lengths())
        self.assertBlobsEqual(out_values, values)
        self.assertBlobsEqual(out_lengths, lengths)

        workspace.RunNetOnce(eval_net)

        workspace.RunNetOnce(predict_net)
        predict_values = workspace.FetchBlob("values_auto_0")
        predict_lengths = workspace.FetchBlob("lengths_auto_0")
        self.assertBlobsEqual(predict_values, np.array([-1] * 10).astype(np.int64))
        self.assertBlobsEqual(predict_lengths, lengths)
