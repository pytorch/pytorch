




from caffe2.python import brew, model_helper, scope
from caffe2.python.modeling.parameter_sharing import (
    ParameterSharing,
    parameter_sharing_context,
)
from caffe2.python.modeling.initializers import (
    Initializer
)
import unittest


class ParameterSharingTest(unittest.TestCase):

    def test_parameter_sharing_default_scopes(self):
        # Test no sharing default scopes
        param_1 = parameter_sharing_context.get_parameter_name('w')
        self.assertEquals(param_1, 'w')
        with scope.NameScope('scope'):
            param_2 = parameter_sharing_context.get_parameter_name('w')
            self.assertEquals(param_2, 'scope/w')
            with scope.NameScope('scope_2'):
                param_3 = parameter_sharing_context.get_parameter_name('w')
                self.assertEquals(param_3, 'scope/scope_2/w')

    def test_parameter_sharing_nested_scopes(self):
        # Test parameter sharing
        with scope.NameScope('global_scope'):
            with ParameterSharing({'model_b': 'model_a'}):
                param_global = parameter_sharing_context.get_parameter_name('w')
                self.assertEquals(param_global, 'global_scope/w')
                # This scope is overridden to match 'model_a'
                with scope.NameScope('model_b'):
                    with ParameterSharing({'shared_scope': ''}):
                        param_4 = parameter_sharing_context.get_parameter_name(
                            'w')
                        self.assertEquals(param_4, 'global_scope/model_a/w')
                        with scope.NameScope('shared_scope'):
                            param_5 = parameter_sharing_context.\
                                get_parameter_name('w')
                            self.assertEquals(param_5, 'global_scope/model_a/w')
                # This scope is supposed to have not sharing
                with scope.NameScope('model_c'):
                    with ParameterSharing({'shared_scope': ''}):
                        param_4 = parameter_sharing_context.get_parameter_name(
                            'w')
                        self.assertEquals(param_4, 'global_scope/model_c/w')
                        with scope.NameScope('shared_scope'):
                            param_5 = parameter_sharing_context.\
                                get_parameter_name('w')
                            self.assertEquals(param_5, 'global_scope/model_c/w')

    def test_parameter_sharing_subscopes(self):
        # Sharing only one of the subscopes
        with ParameterSharing({'global_scope/b': 'global_scope/a'}):
            with scope.NameScope('global_scope'):
                param_6 = parameter_sharing_context.get_parameter_name('w')
                self.assertEquals(param_6, 'global_scope/w')
                with scope.NameScope('a'):
                    param_7 = parameter_sharing_context.get_parameter_name('w')
                    self.assertEquals(param_7, 'global_scope/a/w')
                with scope.NameScope('b'):
                    param_8 = parameter_sharing_context.get_parameter_name('w')
                    self.assertEquals(param_8, 'global_scope/a/w')
                with scope.NameScope('c'):
                    param_9 = parameter_sharing_context.get_parameter_name('w')
                    self.assertEquals(param_9, 'global_scope/c/w')

    def test_create_param(self):
        model = model_helper.ModelHelper(name="test")
        # Test no sharing default scopes
        p1 = model.create_param(
            'w',
            shape=[2],
            initializer=Initializer("ConstantFill")
        )
        with scope.NameScope('some_global_scope'):
            p2 = model.create_param(
                'w',
                shape=[2],
                initializer=Initializer("ConstantFill")
            )
        self.assertNotEqual(model.get_param_info(p1), None)
        self.assertNotEqual(model.get_param_info(p2), None)
        self.assertNotEqual(model.get_param_info(p1), model.get_param_info(p2))
        model.Validate()

    def test_deep_hierarchy(self):
        model = model_helper.ModelHelper(name="test")
        with ParameterSharing({'a': 'b'}):
            with scope.NameScope('a'):
                with ParameterSharing({'c': 'd'}):
                    with scope.NameScope('c'):
                        with ParameterSharing({'e': 'f'}):
                            with scope.NameScope('e'):
                                p = model.create_param(
                                    'w',
                                    shape=[2],
                                    initializer=Initializer("ConstantFill")
                                )
        self.assertNotEqual(model.get_param_info(p), None)


    def test_parameter_sharing_brew(self):
        # Test no sharing default scopes
        model = model_helper.ModelHelper(name="test")
        data = model.net.AddExternalInput("data")
        fc1 = brew.fc(model, data, "fc1", dim_in=16, dim_out=16)
        # Shared params are expected to share the same shape and fail if it's
        # not true
        with self.assertRaises(AssertionError):
            _ = brew.fc(model, data, "fc1", dim_in=2, dim_out=2)  # noqa

        output_blobs = set()
        with scope.NameScope('some_global_scope'):
            with scope.NameScope('model_a'):
                output_blobs.add(str(brew.fc(model, fc1, 'output', 16, 16)))
            with ParameterSharing({'model_b': 'model_a'}),\
                    scope.NameScope('model_b'):
                with ParameterSharing({'shared_1': '', 'shared_2': ''}):
                    # All params in DenseLayers from shared_1, shared_2 and
                    # model_a are shared and will be pointing to:
                    # [some_global_scope/model_a/output_W,
                    #  some_global_scope/model_a/output_b]
                    with scope.NameScope('shared_1'):
                        output_blobs.add(
                            str(brew.fc(model, fc1, 'output', 16, 16)))
                    with scope.NameScope('shared_2'):
                        output_blobs.add(
                            str(brew.fc(model, fc1, 'output', 16, 16)))
                    # Params of this layer are not shared with anyone unless
                    # there is some explicit sharing with model_a/unshared (not
                    # in this example).
                    # Names of the blobs are
                    # [some_global_scope/model_a/unshared/output_W,
                    #  some_global_scope/model_a/unshared/output_b]
                    with scope.NameScope('unshared'):
                        output_blobs.add(
                            str(brew.fc(model, fc1, 'output', 16, 16)))

        self.assertEqual(len(model._parameters_info), 6)
        self.assertEqual(len(output_blobs), 4)
        self.assertEqual(sorted(model._parameters_info.keys()), [
            'fc1_b',
            'fc1_w',
            'some_global_scope/model_a/output_b',
            'some_global_scope/model_a/output_w',
            'some_global_scope/model_a/unshared/output_b',
            'some_global_scope/model_a/unshared/output_w',
        ])
        model.Validate()
