




from caffe2.python import core, scope
from caffe2.python.modeling.parameter_sharing import (
    ParameterSharing,
)
from caffe2.python.optimizer import AdagradOptimizer, AdamOptimizer
from caffe2.python.layer_test_util import LayersTestCase


class ParameterSharingTest(LayersTestCase):

    def test_layer_parameter_name(self):
        output_dims = 2
        with scope.NameScope('global_scope'):
            fc1_output = self.model.FC(
                self.model.input_feature_schema.float_features,
                output_dims
            )
            self.assertEqual(self.model.layers[-1].w, 'global_scope/fc/w')
            self.assertEqual(fc1_output(), 'global_scope/fc/output')

            with scope.NameScope('nested_scope'):
                fc2_output = self.model.FC(
                    fc1_output,
                    output_dims
                )
                self.assertEqual(self.model.layers[-1].w,
                                  'global_scope/nested_scope/fc/w')
                self.assertEqual(fc2_output(),
                                  'global_scope/nested_scope/fc/output')

                fc3_output = self.model.FC(
                    fc1_output,
                    output_dims
                )
                self.assertEqual(self.model.layers[-1].w,
                                  'global_scope/nested_scope/fc_auto_0/w')
                self.assertEqual(fc3_output(),
                                  'global_scope/nested_scope/fc_auto_0/output')

    def test_layer_shared_parameter_name_different_namescopes(self):
        output_dims = 2
        with scope.NameScope('global_scope'):
            with ParameterSharing({'scope_1': 'scope_0'}):
                with scope.NameScope('scope_0'):
                    fc1_output = self.model.FC(
                        self.model.input_feature_schema.float_features,
                        output_dims
                    )
                    self.assertEqual(self.model.layers[-1].w,
                                      'global_scope/scope_0/fc/w')
                    self.assertEqual(fc1_output(),
                                      'global_scope/scope_0/fc/output')

                with scope.NameScope('scope_1'):
                    fc2_output = self.model.FC(
                        self.model.input_feature_schema.float_features,
                        output_dims
                    )
                    self.assertEqual(self.model.layers[-1].w,
                                      'global_scope/scope_0/fc/w')
                    self.assertEqual(fc2_output(),
                                      'global_scope/scope_1/fc/output')

    def test_layer_shared_parameter_name_within_same_namescope(self):
        output_dims = 2
        with scope.NameScope('global_scope'):
            with ParameterSharing({'fc_auto_0': 'fc'}):
                self.model.FC(
                    self.model.input_feature_schema.float_features,
                    output_dims
                )
                self.assertEqual(self.model.layers[-1].w,
                                  'global_scope/fc/w')

                self.model.FC(
                    self.model.input_feature_schema.float_features,
                    output_dims
                )
                self.assertEqual(self.model.layers[-1].w,
                                  'global_scope/fc/w')

    def test_layer_shared_parameter_name_within_same_namescope_customized_name(self):
        output_dims = 2
        with scope.NameScope('global_scope'):
            with ParameterSharing({'new_fc': 'shared_fc'}):
                self.model.FC(
                    self.model.input_feature_schema.float_features,
                    output_dims,
                    name='shared_fc'
                )
                self.assertEqual(self.model.layers[-1].w,
                                  'global_scope/shared_fc/w')

                self.model.FC(
                    self.model.input_feature_schema.float_features,
                    output_dims,
                    name='new_fc'
                )
                self.assertEqual(self.model.layers[-1].w,
                                  'global_scope/shared_fc/w')

    def test_layer_shared_parameter_name_different_shapes(self):
        output_dims = 2
        with scope.NameScope('global_scope'):
            with ParameterSharing({'fc_auto_0': 'fc'}):
                self.model.FC(
                    self.model.input_feature_schema.float_features,
                    output_dims
                )
                self.assertEqual(self.model.layers[-1].w,
                                  'global_scope/fc/w')

                with self.assertRaisesRegex(ValueError, 'Got inconsistent shapes .*'):
                    self.model.FC(
                        self.model.input_feature_schema.float_features,
                        output_dims + 1
                    )

    def test_layer_duplicated_parameter_init(self):
        output_dims = 2
        with scope.NameScope('global_scope'):
            with ParameterSharing({'new_fc': 'shared_fc'}):
                self.model.FC(
                    self.model.input_feature_schema.float_features,
                    output_dims,
                    name='shared_fc'
                )
                self.model.FC(
                    self.model.input_feature_schema.float_features,
                    output_dims,
                    name='new_fc'
                )

        train_init_net = core.Net('train_init_net')
        train_net = core.Net('train_net')
        for layer in self.model.layers:
            layer.add_operators(train_net, train_init_net)
        op_outputs = []
        for op in train_init_net._net.op:
            op_outputs.extend(op.output)

        # only fill these parameter blobs once
        self.assertEqual(
            sorted(op_outputs),
            ['global_scope/shared_fc/b', 'global_scope/shared_fc/w']
        )

    def test_layer_shared_parameter_optim_validator(self):
        """
        This test is to cover the _validate_param_optim function in
        layer_model_helper class.
        """

        output_dims = 2

        adagrad_optim = AdagradOptimizer(
            alpha=0.004,
            epsilon=0.02,
        )

        self.model.default_optimizer = adagrad_optim

        # the following covers the branch -- optim is None
        with scope.NameScope('global_scope_0'):
            with ParameterSharing({'scope_1': 'scope_0'}):
                with scope.NameScope('scope_0'):
                    fc1_output = self.model.FC(
                        self.model.input_feature_schema.float_features,
                        output_dims,
                        weight_optim=self.model.NoOptim,
                    )

                with scope.NameScope('scope_1'), self.assertRaises(Exception):
                    fc2_output = self.model.FC(
                        self.model.input_feature_schema.float_features,
                        output_dims
                    )

        # the following covers the branch -- optim is NoOptim
        with scope.NameScope('global_scope_1'):
            with ParameterSharing({'scope_1': 'scope_0'}):
                with scope.NameScope('scope_0'):
                    fc1_output = self.model.FC(
                        self.model.input_feature_schema.float_features,
                        output_dims,
                        weight_optim=None,
                    )

                with scope.NameScope('scope_1'), self.assertRaises(Exception):
                    fc2_output = self.model.FC(
                        self.model.input_feature_schema.float_features,
                        output_dims,
                        weight_optim=self.model.NoOptim,
                    )

        # the following covers the branch -- optim is an instance of Optimizer
        adagrad_optim_2 = AdagradOptimizer(
            alpha=0.005,
            epsilon=0.02,
        )

        adam_optim = AdamOptimizer()

        self.model.default_optimizer = adagrad_optim_2

        with scope.NameScope('global_scope_2'):
            with ParameterSharing({'scope_1': 'scope_0', 'scope_2': 'scope_0'}):
                with scope.NameScope('scope_0'):
                    fc1_output = self.model.FC(
                        self.model.input_feature_schema.float_features,
                        output_dims,
                        weight_optim=None,   # it will use adagrad_optim_2
                    )

                with scope.NameScope('scope_1'), self.assertRaises(Exception):
                    fc2_output = self.model.FC(
                        self.model.input_feature_schema.float_features,
                        output_dims,
                        weight_optim=adagrad_optim,
                    )

                with scope.NameScope('scope_2'), self.assertRaises(Exception):
                    fc2_output = self.model.FC(
                        self.model.input_feature_schema.float_features,
                        output_dims,
                        weight_optim=adam_optim,
                    )
