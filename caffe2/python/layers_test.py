from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import namedtuple

from caffe2.python import (
    core,
    layer_model_instantiator,
    layer_model_helper,
    schema,
    test_util,
)
import numpy as np


OpSpec = namedtuple("OpSpec", "type input output")


class TestLayers(test_util.TestCase):

    def setUp(self):
        super(TestLayers, self).setUp()
        input_feature_schema = schema.Struct(
            ('float_features', schema.Scalar((np.float32, (32,)))),
        )
        trainer_extra_schema = schema.Struct()

        self.model = layer_model_helper.LayerModelHelper(
            'test_model',
            input_feature_schema=input_feature_schema,
            trainer_extra_schema=trainer_extra_schema)

    def new_record(self, schema_obj):
        return schema.NewRecord(self.model.net, schema_obj)

    def get_training_nets(self):
        """
        We don't use
        layer_model_instantiator.generate_training_nets_forward_only()
        here because it includes initialization of global constants, which make
        testing tricky
        """
        train_net = core.Net('train_net')
        train_init_net = core.Net('trai_init_net')
        for layer in self.model.layers:
            layer.add_operators(train_net, train_init_net)
        return train_init_net, train_net

    def get_predict_net(self):
        return layer_model_instantiator.generate_predict_net(self.model)

    def assertBlobsEqual(self, spec_blobs, op_blobs):
        """
        spec_blobs can either be None or a list of blob names. If it's None,
        then no assertion is performed. The elements of the list can be None,
        in that case, it means that position will not be checked.
        """
        if spec_blobs is None:
            return
        self.assertEqual(len(spec_blobs), len(op_blobs))
        for spec_blob, op_blob in zip(spec_blobs, op_blobs):
            if spec_blob is None:
                continue
            self.assertEqual(spec_blob, op_blob)

    def assertNetContainOps(self, net, op_specs):
        """
        Given a net and a list of OpSpec's, check that the net match the spec
        """
        ops = net.Proto().op
        self.assertEqual(len(op_specs), len(ops))
        for op, op_spec in zip(ops, op_specs):
            self.assertEqual(op_spec.type, op.type)
            self.assertBlobsEqual(op_spec.input, op.input)
            self.assertBlobsEqual(op_spec.output, op.output)
        return ops

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

    def testBatchSigmoidCrossEntropyLoss(self):
        input_record = self.new_record(schema.Struct(
            ('label', schema.Scalar((np.float32, (32,)))),
            ('prediction', schema.Scalar((np.float32, (32,))))
        ))
        loss = self.model.BatchSigmoidCrossEntropyLoss(input_record)
        self.assertEqual(schema.Scalar((np.float32, tuple())), loss)

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
