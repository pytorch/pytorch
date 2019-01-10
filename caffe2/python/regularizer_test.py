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

    @given(
        X=hu.arrays(dims=[2, 5], elements=st.floats(min_value=-1.0, max_value=1.0)),
        left_open=st.booleans(),
        right_open=st.booleans(),
        eps=st.floats(min_value=1e-6, max_value=1e-4),
        ub=st.floats(min_value=-1.0, max_value=1.0),
        lb=st.floats(min_value=-1.0, max_value=1.0),
        **hu.gcs_cpu_only
    )
    def test_bounded_grad_proj(self, X, left_open, right_open, eps, ub, lb, gc, dc):
        if ub - (eps if right_open else 0.) < lb + (eps if left_open else 0.):
            return
        param = core.BlobReference("X")
        workspace.FeedBlob(param, X)
        train_init_net, train_net = self.get_training_nets()
        reg = regularizer.BoundedGradientProjection(
            lb=lb, ub=ub, left_open=left_open, right_open=right_open, epsilon=eps
        )
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
            return np.clip(
                X, lb + (eps if left_open else 0.), ub - (eps if right_open else 0.)
            )

        assert output is None
        npt.assert_allclose(workspace.blobs[param], ref(X), atol=1e-7)

    @given(
        output_dim=st.integers(1, 10),
        input_num=st.integers(3, 30),
        reg_weight=st.integers(0, 10)
    )
    def test_group_l1_norm(self, output_dim, input_num, reg_weight):
        """
        1. create a weight blob
        2. create random group splits
        3. run group_l1_nrom with the weight blob
        4. run equivalent np operations to calculate group l1 norm
        5. compare if the results from 3 and 4 are equal
        """
        def compare_reference(weight, group_boundaries, reg_lambda, output):
            group_splits = np.hsplit(weight, group_boundaries[1:-1])
            l2_reg = np.sqrt([np.sum(np.square(g)) for g in group_splits])
            l2_normalized = np.multiply(l2_reg,
                np.array([np.sqrt(g.shape[1]) for g in group_splits]))
            result = np.multiply(np.sum(l2_normalized), reg_lambda)
            npt.assert_almost_equal(result, workspace.blobs[output], decimal=2)

        weight = np.random.rand(output_dim, input_num).astype(np.float32)

        feature_num = np.random.randint(low=1, high=input_num - 1)
        group_boundaries = [0]
        group_boundaries = np.append(
            group_boundaries,
            np.sort(
                np.random.choice(range(1, input_num - 1), feature_num, replace=False)
            ),
        )
        group_boundaries = np.append(group_boundaries, [input_num])
        split_info = np.diff(group_boundaries)

        weight_blob = core.BlobReference("weight_blob")
        workspace.FeedBlob(weight_blob, weight)

        train_init_net, train_net = self.get_training_nets()
        reg = regularizer.GroupL1Norm(reg_weight * 0.1, split_info.tolist())
        output = reg(
            train_net, train_init_net, weight_blob, by=RegularizationBy.ON_LOSS
        )

        workspace.RunNetOnce(train_init_net)
        workspace.RunNetOnce(train_net)

        compare_reference(weight, group_boundaries, reg_weight * 0.1, output)
