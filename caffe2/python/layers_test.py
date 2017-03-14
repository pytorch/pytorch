from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import (
    layer_model_instantiator,
    layer_model_helper,
    schema,
    test_util,
)
import numpy as np


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
        self.assertEquals(1, len(loss.field_types()))
        self.assertEquals(np.float32, loss.field_types()[0].base)
        self.assertEquals(tuple(), loss.field_types()[0].shape)
