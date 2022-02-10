## @package layer_test_util
# Module caffe2.python.layer_test_util





from collections import namedtuple

from caffe2.python import (
    core,
    layer_model_instantiator,
    layer_model_helper,
    schema,
    test_util,
    workspace,
    utils,
)
from caffe2.proto import caffe2_pb2
import numpy as np


# pyre-fixme[13]: Pyre can't detect attribute initialization through the
#    super().__new__ call
class OpSpec(namedtuple("OpSpec", "type input output arg")):

    def __new__(cls, op_type, op_input, op_output, op_arg=None):
        return super(OpSpec, cls).__new__(cls, op_type, op_input,
                                          op_output, op_arg)


class LayersTestCase(test_util.TestCase):

    def setUp(self):
        super(LayersTestCase, self).setUp()
        self.setup_example()

    def setup_example(self):
        """
        This is undocumented feature in hypothesis,
        https://github.com/HypothesisWorks/hypothesis-python/issues/59
        """
        workspace.ResetWorkspace()
        self.reset_model()

    def reset_model(self, input_feature_schema=None, trainer_extra_schema=None):
        input_feature_schema = input_feature_schema or schema.Struct(
            ('float_features', schema.Scalar((np.float32, (32,)))),
        )
        trainer_extra_schema = trainer_extra_schema or schema.Struct()
        self.model = layer_model_helper.LayerModelHelper(
            'test_model',
            input_feature_schema=input_feature_schema,
            trainer_extra_schema=trainer_extra_schema)

    def new_record(self, schema_obj):
        return schema.NewRecord(self.model.net, schema_obj)

    def get_training_nets(self, add_constants=False):
        """
        We don't use
        layer_model_instantiator.generate_training_nets_forward_only()
        here because it includes initialization of global constants, which make
        testing tricky
        """
        train_net = core.Net('train_net')
        if add_constants:
            train_init_net = self.model.create_init_net('train_init_net')
        else:
            train_init_net = core.Net('train_init_net')
        for layer in self.model.layers:
            layer.add_operators(train_net, train_init_net)
        return train_init_net, train_net

    def get_eval_net(self):
        return layer_model_instantiator.generate_eval_net(self.model)

    def get_predict_net(self):
        return layer_model_instantiator.generate_predict_net(self.model)

    def run_train_net(self):
        self.model.output_schema = schema.Struct()
        train_init_net, train_net = \
            layer_model_instantiator.generate_training_nets(self.model)
        workspace.RunNetOnce(train_init_net)
        workspace.RunNetOnce(train_net)

    def run_train_net_forward_only(self, num_iter=1):
        self.model.output_schema = schema.Struct()
        train_init_net, train_net = \
            layer_model_instantiator.generate_training_nets_forward_only(
                self.model)
        workspace.RunNetOnce(train_init_net)
        assert num_iter > 0, 'num_iter must be larger than 0'
        workspace.CreateNet(train_net)
        workspace.RunNet(train_net.Proto().name, num_iter=num_iter)

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

    def assertArgsEqual(self, spec_args, op_args):
        self.assertEqual(len(spec_args), len(op_args))
        keys = [a.name for a in op_args]

        def parse_args(args):
            operator = caffe2_pb2.OperatorDef()
            # Generate the expected value in the same order
            for k in keys:
                v = args[k]
                arg = utils.MakeArgument(k, v)
                operator.arg.add().CopyFrom(arg)
            return operator.arg

        self.assertEqual(parse_args(spec_args), op_args)

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
            if op_spec.arg is not None:
                self.assertArgsEqual(op_spec.arg, op.arg)
        return ops
