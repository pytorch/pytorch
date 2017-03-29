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


class LayersTestCase(test_util.TestCase):

    def setUp(self):
        super(LayersTestCase, self).setUp()
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
        train_init_net = core.Net('train_init_net')
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
