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
    xfail("__getitem__"),
    xfail("__rdiv__"),
    xfail("__rmatmul__"),
    xfail("__rmod__"),
    xfail("__rpow__"),
    xfail("__rsub__"),
    xfail("allclose"),
    xfail("argwhere"),
    xfail("bernoulli"),
    xfail("bucketize"),
    xfail("cdouble"),
    xfail("cfloat"),
    xfail("cholesky_inverse"),
    xfail("cholesky"),
    xfail("combinations"),
    xfail("corrcoef"),
    xfail("cov"),
    xfail("equal"),
    xfail("H"),
    xfail("linalg.eigh"),
    xfail("linalg.eigvalsh"),
    xfail("linalg.lstsq", "grad_oriented"),
    xfail("linalg.lstsq"),
    xfail("masked_select"),
    xfail("mH"),
    xfail("multinomial"),
    xfail("nanquantile"),
    xfail("narrow"),
    xfail("nn.functional.alpha_dropout"),
    xfail("nn.functional.ctc_loss"),
    xfail("nn.functional.dropout"),
    xfail("nn.functional.dropout2d"),
    xfail("nn.functional.dropout3d"),
    xfail("nn.functional.feature_alpha_dropout", "with_train"),
    xfail("nn.functional.feature_alpha_dropout", "without_train"),
    xfail("nn.functional.fractional_max_pool2d"),
    xfail("nn.functional.fractional_max_pool3d"),
    xfail("nn.functional.gaussian_nll_loss"),
    xfail("nn.functional.rrelu"),
    xfail("nn.functional.scaled_dot_product_attention"),
    xfail("nn.functional.triplet_margin_with_distance_loss"),
    xfail("nonzero"),
    xfail("normal", "number_mean"),
    xfail("normal"),
    xfail("pca_lowrank"),
    xfail("quantile"),
    xfail("rand_like"),
    xfail("randint_like"),
    xfail("randint"),
    xfail("randn_like"),
    xfail("randn"),
    xfail("repeat_interleave"),
    xfail("segment_reduce", "lengths"),
    xfail("segment_reduce", "offsets"),
    xfail("svd_lowrank"),
    xfail("tensor_split"),
    xfail("uniform"),
    xfail("unique_consecutive"),
    xfail("unique"),
}

export_failures_dynamic = {
    xfail("block_diag"),
    xfail("complex"),
    xfail("i0"),
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


class TestExportOpInfo(TestCase):
    @ops(op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestExportOpInfo', 'test_export_with_aten', export_failures)
    def test_export_with_aten(self, device, dtype, op):
        _test_export_helper(self, device, dtype, op, aten_graph=True)

    @ops(op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestExportOpInfo', 'test_export_with_aten_decomp', export_failures)
    def test_export_with_aten_decomp(self, device, dtype, op):
        _test_export_helper(self, device, dtype, op, aten_graph=True, decomposition_table=core_aten_decompositions)


only_for = ("cpu")
instantiate_device_type_tests(TestExportOpInfo, globals(), only_for=only_for)


if __name__ == '__main__':
    run_tests()
