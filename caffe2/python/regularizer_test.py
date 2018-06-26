from __future__ import absolute_import, division, print_function

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
from caffe2.python import core, layer_model_instantiator, regularizer, schema, workspace
from caffe2.python.layer_test_util import LayersTestCase
from caffe2.python.optimizer import SgdOptimizer
from caffe2.python.regularizer import L1Norm, RegularizationBy
from caffe2.python.regularizer_context import RegularizerContext, UseRegularizer
from hypothesis import given


class TestRegularizerContext(LayersTestCase):
    @given(X=hu.arrays(dims=[2, 5]))
    def test_regularizer_context(self, X):
        weight_reg_out = L1Norm(0.2)
        bias_reg_out = L1Norm(0)
        regularizers = {"WEIGHT": weight_reg_out, "BIAS": bias_reg_out}

        output_dims = 2
        input_record = self.new_record(schema.Scalar((np.float32, (5,))))
        schema.FeedRecord(input_record, [X])

        with UseRegularizer(regularizers):
            weight_reg = RegularizerContext.current().get_regularizer("WEIGHT")
            bias_reg = RegularizerContext.current().get_regularizer("BIAS")
            optim = SgdOptimizer(0.15)

            assert (
                weight_reg == weight_reg_out
            ), "fail to get correct weight reg from context"
            assert bias_reg == bias_reg_out, "fail to get correct bias reg from context"
            fc_output = self.model.FC(
                input_record,
                output_dims,
                weight_optim=optim,
                bias_optim=optim,
                weight_reg=weight_reg,
                bias_reg=bias_reg,
            )
            # model.output_schema has to a struct
            self.model.output_schema = schema.Struct(("fc_output", fc_output))

            self.assertEqual(schema.Scalar((np.float32, (output_dims,))), fc_output)

            _, train_net = layer_model_instantiator.generate_training_nets(self.model)
            ops = train_net.Proto().op
            ops_type_list = [ops[i].type for i in range(len(ops))]
            assert ops_type_list.count("LpNorm") == 2
            assert ops_type_list.count("Scale") == 4
            assert ops_type_list.count("LpNormGradient") == 2


class TestRegularizer(LayersTestCase):
    @given(X=hu.arrays(dims=[2, 5], elements=st.floats(min_value=-1.0, max_value=1.0)))
    def test_log_barrier(self, X):
        param = core.BlobReference("X")
        workspace.FeedBlob(param, X)
        train_init_net, train_net = self.get_training_nets()
        reg = regularizer.LogBarrier(1.0)
        output = reg(train_net, train_init_net, param, by=RegularizationBy.ON_LOSS)
        reg(
            train_net,
            train_init_net,
            param,
            grad=None,
            by=RegularizationBy.AFTER_OPTIMIZER,
        )
        workspace.RunNetOnce(train_init_net)
        workspace.RunNetOnce(train_net)

        def ref(X):
            return (
                np.array(np.sum(-np.log(np.clip(X, 1e-9, None))) * 0.5).astype(
                    np.float32
                ),
                np.clip(X, 1e-9, None),
            )

        for x, y in zip(workspace.FetchBlobs([output, param]), ref(X)):
            npt.assert_allclose(x, y, rtol=1e-3)
