# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_device_type import ops
from torch._decomp import core_aten_decompositions
from test_proxy_tensor import xfail, skipOps, _get_safe_inplace, symbolic_tensor_failures

import itertools

export_failures = {
    xfail("H"),
    xfail("__getitem__"),
    xfail("__rdiv__"),
    xfail("__rmatmul__"),
    xfail("__rmod__"),
    xfail("__rpow__"),
    xfail("__rsub__"),
    xfail("argwhere"),
    xfail("bernoulli"),
    xfail("cdouble"),
    xfail("cfloat"),
    xfail("cholesky_inverse"),
    xfail("corrcoef"),
    xfail("cov"),
    xfail("equal"),
    xfail("mH"),
    xfail("masked_select"),
    xfail("multinomial"),
    xfail("nanquantile"),
    xfail("narrow"),
    xfail("nn.functional.alpha_dropout"),
    xfail("nn.functional.dropout2d"),
    xfail("nn.functional.dropout3d"),
    xfail("nn.functional.dropout"),
    xfail("nn.functional.feature_alpha_dropout", "with_train"),
    xfail("nn.functional.feature_alpha_dropout", "without_train"),
    xfail("normal"),
    xfail("normal", "number_mean"),
    xfail("quantile"),
    xfail("rand_like"),
    xfail("randint"),
    xfail("randint_like"),
    xfail("randn_like"),
    xfail("tensor_split"),
    xfail("uniform"),
}

export_failures_with_kwargs = {
    xfail("allclose"),
    xfail("bucketize"),
    xfail("cholesky"),
    xfail("combinations"),
    xfail("gradient"),
    xfail("histogram"),
    xfail("histogramdd"),
    xfail("linalg.eigh"),
    xfail("linalg.eigvalsh"),
    xfail("linalg.lstsq"),
    xfail("linalg.lstsq", "grad_oriented"),
    xfail("linalg.matrix_rank"),
    xfail("linalg.pinv"),
    xfail("masked.amax"),
    xfail("masked.amin"),
    xfail("masked.argmax"),
    xfail("masked.argmin"),
    xfail("masked.cumprod"),
    xfail("masked.cumsum"),
    xfail("masked.log_softmax"),
    xfail("masked.logaddexp"),
    xfail("masked.logsumexp"),
    xfail("masked.mean"),
    xfail("masked.median"),
    xfail("masked.norm"),
    xfail("masked.prod"),
    xfail("masked.softmax"),
    xfail("masked.softmin"),
    xfail("masked.std"),
    xfail("masked.sum"),
    xfail("masked.var"),
    xfail("nn.functional.binary_cross_entropy"),
    xfail("nn.functional.binary_cross_entropy_with_logits"),
    xfail("nn.functional.cross_entropy"),
    xfail("nn.functional.ctc_loss"),
    xfail("nn.functional.embedding_bag"),
    xfail("nn.functional.fractional_max_pool2d"),
    xfail("nn.functional.fractional_max_pool3d"),
    xfail("nn.functional.gaussian_nll_loss"),
    xfail("nn.functional.group_norm"),
    xfail("nn.functional.instance_norm"),
    xfail("nn.functional.multilabel_soft_margin_loss"),
    xfail("nn.functional.nll_loss"),
    xfail("nn.functional.prelu"),
    xfail("nn.functional.rrelu"),
    xfail("nn.functional.scaled_dot_product_attention"),
    xfail("nn.functional.triplet_margin_with_distance_loss"),
    xfail("nonzero"),
    xfail("pca_lowrank"),
    xfail("randn"),
    xfail("repeat_interleave"),
    xfail("searchsorted"),
    xfail("segment_reduce", "lengths"),
    xfail("segment_reduce", "offsets"),
    xfail("stft"),
    xfail("svd_lowrank"),
    xfail("unique_consecutive"),
    xfail("unique"),
}

export_failures_dynamic = {
    xfail("block_diag"),
    xfail("complex"),
    xfail("i0"),
    xfail("ldexp"),
    xfail("masked_scatter"),
    xfail("nn.functional.max_unpool1d"),
    xfail("nn.functional.max_unpool2d"),
    xfail("nn.functional.max_unpool3d"),
    xfail("to_sparse"),
}


def _test_export_helper(self, device, dtype, op,
                        aten_graph=False,
                        decomposition_table=None,
                        tracing_mode="real",
                        inplace=False):
    fn = _get_safe_inplace(op.get_inplace()) if inplace else op.op

    def f(*args, **kwargs):
        return fn(*args, **kwargs)

    sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=False)

    # Limit ourselves to first 100 inputs so symbolic tracing tests don't take too long
    for sample_input in itertools.islice(sample_inputs_itr, 100):
        args = [sample_input.input] + list(sample_input.args)
        kwargs = sample_input.kwargs

        gm, guard = torch._dynamo.export(
            f, *args, aten_graph=aten_graph, decomposition_table=decomposition_table, tracing_mode=tracing_mode, **kwargs)

        for node in gm.graph.nodes:
            if node.op == 'call_function' and isinstance(node.target, torch._ops.OpOverload):
                if aten_graph:
                    self.assertTrue(
                        node.target.namespace == "aten",
                        f"dynamo.export(aten_graph=True) should only results aten ops, seeign {node.target.name()}"
                    )

            # TODO: enable this when core aten ops are finalized
            # if decomposition_table is core_aten_decompositions:
            #     self.assertTrue(
            #         torch.Tag.core in node.target.tags,
            #         f"Decomposed {op.name} should only contain core aten ops, but found {node.target.name()}"
            #     )

class TestExportOpInfo(TestCase):
    @ops(op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestExportOpInfo', 'test_export_with_aten', export_failures | export_failures_with_kwargs)
    def test_export_with_aten(self, device, dtype, op):
        _test_export_helper(self, device, dtype, op, aten_graph=True)

    @ops(op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestExportOpInfo', 'test_export_with_aten_dynamic',
             export_failures | export_failures_with_kwargs | symbolic_tensor_failures | export_failures_dynamic)
    def test_export_with_aten_dynamic(self, device, dtype, op):
        _test_export_helper(self, device, dtype, op, aten_graph=True, tracing_mode="symbolic")

    @ops(op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestExportOpInfo', 'test_export_with_aten_decomp', export_failures | export_failures_with_kwargs)
    def test_export_with_aten_decomp(self, device, dtype, op):
        _test_export_helper(self, device, dtype, op, aten_graph=True, decomposition_table=core_aten_decompositions)


only_for = ("cpu")
instantiate_device_type_tests(TestExportOpInfo, globals(), only_for=only_for)


if __name__ == '__main__':
    run_tests()
