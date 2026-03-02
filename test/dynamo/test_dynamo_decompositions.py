# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo.config
import torch._dynamo.test_case
from torch._dynamo.testing import EagerAndRecordGraphs, normalize_gm
from torch.testing._internal.common_utils import run_tests, skipIfCrossRef


class TestDynamoDecompositions(torch._dynamo.test_case.TestCase):
    """Tests for enable_dynamo_decompositions config flag.

    When enable_dynamo_decompositions=True, certain optimizer ops are decomposed
    into their constituent ops to avoid item() graph breaks.
    When False, the original ops are preserved.
    """

    @skipIfCrossRef
    def test_addcmul_inplace_decomposition_enabled(self):
        """With decompositions enabled, addcmul_ should decompose into mul and add_."""

        def fn(x, tensor1, tensor2, value):
            return x.addcmul_(tensor1, tensor2, value=value)

        eager = EagerAndRecordGraphs()
        with torch._dynamo.config.patch(enable_dynamo_decompositions=True):
            x = torch.randn(4)
            tensor1 = torch.randn(4)
            tensor2 = torch.randn(4)
            value = torch.tensor(0.5)
            torch.compile(fn, backend=eager, fullgraph=True)(x, tensor1, tensor2, value)

        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))

        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4]", L_tensor1_: "f32[4]", L_tensor2_: "f32[4]", L_value_: "f32[]"):
        l_x_ = L_x_
        l_tensor1_ = L_tensor1_
        l_tensor2_ = L_tensor2_
        l_value_ = L_value_

        mul: "f32[4]" = l_tensor1_ * l_tensor2_;  l_tensor1_ = l_tensor2_ = None
        mul_1: "f32[4]" = mul * l_value_;  mul = l_value_ = None
        add_: "f32[4]" = l_x_.add_(mul_1);  l_x_ = mul_1 = None
        return (add_,)
""",
        )

    @skipIfCrossRef
    def test_addcmul_inplace_decomposition_disabled(self):
        """With decompositions disabled, addcmul_ should remain as the original op.

        Note: When using a tensor value and decompositions are disabled, there can be
        graph breaks due to item() calls. This test uses a scalar value to avoid that.
        """

        def fn(x, tensor1, tensor2):
            return x.addcmul_(tensor1, tensor2, value=0.5)

        eager = EagerAndRecordGraphs()
        with torch._dynamo.config.patch(enable_dynamo_decompositions=False):
            x = torch.randn(4)
            tensor1 = torch.randn(4)
            tensor2 = torch.randn(4)
            torch.compile(fn, backend=eager, fullgraph=True)(x, tensor1, tensor2)

        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))

        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4]", L_tensor1_: "f32[4]", L_tensor2_: "f32[4]"):
        l_x_ = L_x_
        l_tensor1_ = L_tensor1_
        l_tensor2_ = L_tensor2_

        addcmul_: "f32[4]" = l_x_.addcmul_(l_tensor1_, l_tensor2_, value = 0.5);  l_x_ = l_tensor1_ = l_tensor2_ = None
        return (addcmul_,)
""",
        )

    @skipIfCrossRef
    def test_addcmul_inplace_decomposition_disabled_capture_scalar(self):
        """With decompositions disabled and capture_scalar_outputs=True, addcmul_ with
        scalar value should work without graph breaks.
        """

        def fn(x, tensor1, tensor2):
            return x.addcmul_(tensor1, tensor2, value=0.5)

        eager = EagerAndRecordGraphs()
        with torch._dynamo.config.patch(
            enable_dynamo_decompositions=False, capture_scalar_outputs=True
        ):
            x = torch.randn(4)
            tensor1 = torch.randn(4)
            tensor2 = torch.randn(4)
            torch.compile(fn, backend=eager, fullgraph=True)(x, tensor1, tensor2)

        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))

        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4]", L_tensor1_: "f32[4]", L_tensor2_: "f32[4]"):
        l_x_ = L_x_
        l_tensor1_ = L_tensor1_
        l_tensor2_ = L_tensor2_

        addcmul_: "f32[4]" = l_x_.addcmul_(l_tensor1_, l_tensor2_, value = 0.5);  l_x_ = l_tensor1_ = l_tensor2_ = None
        return (addcmul_,)
""",
        )

    @skipIfCrossRef
    def test_add_inplace_with_alpha_decomposition_enabled(self):
        """With decompositions enabled, add_ with alpha should decompose into mul and add_."""

        def fn(x, other, alpha):
            return x.add_(other, alpha=alpha)

        eager = EagerAndRecordGraphs()
        with torch._dynamo.config.patch(enable_dynamo_decompositions=True):
            x = torch.randn(4)
            other = torch.randn(4)
            alpha = torch.tensor(2.0)
            torch.compile(fn, backend=eager, fullgraph=True)(x, other, alpha)

        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))

        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4]", L_other_: "f32[4]", L_alpha_: "f32[]"):
        l_x_ = L_x_
        l_other_ = L_other_
        l_alpha_ = L_alpha_

        mul: "f32[4]" = torch.mul(l_other_, l_alpha_);  l_other_ = l_alpha_ = None
        add_: "f32[4]" = l_x_.add_(mul);  l_x_ = mul = None
        return (add_,)
""",
        )

    @skipIfCrossRef
    def test_add_inplace_with_alpha_decomposition_disabled(self):
        """With decompositions disabled, add_ with alpha should remain as the original op.

        Note: When using a tensor alpha and decompositions are disabled, there can be
        graph breaks due to item() calls. This test uses a scalar alpha to avoid that.
        """

        def fn(x, other):
            return x.add_(other, alpha=2.0)

        eager = EagerAndRecordGraphs()
        with torch._dynamo.config.patch(enable_dynamo_decompositions=False):
            x = torch.randn(4)
            other = torch.randn(4)
            torch.compile(fn, backend=eager, fullgraph=True)(x, other)

        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))

        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4]", L_other_: "f32[4]"):
        l_x_ = L_x_
        l_other_ = L_other_

        add_: "f32[4]" = l_x_.add_(l_other_, alpha = 2.0);  l_x_ = l_other_ = None
        return (add_,)
""",
        )

    @skipIfCrossRef
    def test_add_inplace_with_alpha_decomposition_disabled_capture_scalar(self):
        """With decompositions disabled and capture_scalar_outputs=True, add_ with
        scalar alpha should work without graph breaks.
        """

        def fn(x, other):
            return x.add_(other, alpha=2.0)

        eager = EagerAndRecordGraphs()
        with torch._dynamo.config.patch(
            enable_dynamo_decompositions=False, capture_scalar_outputs=True
        ):
            x = torch.randn(4)
            other = torch.randn(4)
            torch.compile(fn, backend=eager, fullgraph=True)(x, other)

        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))

        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4]", L_other_: "f32[4]"):
        l_x_ = L_x_
        l_other_ = L_other_

        add_: "f32[4]" = l_x_.add_(l_other_, alpha = 2.0);  l_x_ = l_other_ = None
        return (add_,)
""",
        )

    @skipIfCrossRef
    def test_addcdiv_inplace_decomposition_enabled(self):
        """With decompositions enabled, addcdiv_ should decompose into div, mul, and add_."""

        def fn(x, tensor1, tensor2, value):
            return x.addcdiv_(tensor1, tensor2, value=value)

        eager = EagerAndRecordGraphs()
        with torch._dynamo.config.patch(enable_dynamo_decompositions=True):
            x = torch.randn(4)
            tensor1 = torch.randn(4)
            tensor2 = torch.randn(4) + 0.1  # avoid div by zero
            value = torch.tensor(0.5)
            torch.compile(fn, backend=eager, fullgraph=True)(x, tensor1, tensor2, value)

        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))

        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4]", L_tensor1_: "f32[4]", L_tensor2_: "f32[4]", L_value_: "f32[]"):
        l_x_ = L_x_
        l_tensor1_ = L_tensor1_
        l_tensor2_ = L_tensor2_
        l_value_ = L_value_

        div: "f32[4]" = torch.div(l_tensor1_, l_tensor2_);  l_tensor1_ = l_tensor2_ = None
        mul: "f32[4]" = torch.mul(div, l_value_);  div = l_value_ = None
        add_: "f32[4]" = l_x_.add_(mul);  l_x_ = mul = None
        return (add_,)
""",
        )

    @skipIfCrossRef
    def test_addcdiv_inplace_decomposition_disabled(self):
        """With decompositions disabled, addcdiv_ should remain as the original op.

        Note: When using a tensor value and decompositions are disabled, there can be
        graph breaks due to item() calls. This test uses a scalar value to avoid that.
        """

        def fn(x, tensor1, tensor2):
            return x.addcdiv_(tensor1, tensor2, value=0.5)

        eager = EagerAndRecordGraphs()
        with torch._dynamo.config.patch(enable_dynamo_decompositions=False):
            x = torch.randn(4)
            tensor1 = torch.randn(4)
            tensor2 = torch.randn(4) + 0.1  # avoid div by zero
            torch.compile(fn, backend=eager, fullgraph=True)(x, tensor1, tensor2)

        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))

        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4]", L_tensor1_: "f32[4]", L_tensor2_: "f32[4]"):
        l_x_ = L_x_
        l_tensor1_ = L_tensor1_
        l_tensor2_ = L_tensor2_

        addcdiv_: "f32[4]" = l_x_.addcdiv_(l_tensor1_, l_tensor2_, value = 0.5);  l_x_ = l_tensor1_ = l_tensor2_ = None
        return (addcdiv_,)
""",
        )

    @skipIfCrossRef
    def test_addcdiv_inplace_decomposition_disabled_capture_scalar(self):
        """With decompositions disabled and capture_scalar_outputs=True, addcdiv_ with
        scalar value should work without graph breaks.
        """

        def fn(x, tensor1, tensor2):
            return x.addcdiv_(tensor1, tensor2, value=0.5)

        eager = EagerAndRecordGraphs()
        with torch._dynamo.config.patch(
            enable_dynamo_decompositions=False, capture_scalar_outputs=True
        ):
            x = torch.randn(4)
            tensor1 = torch.randn(4)
            tensor2 = torch.randn(4) + 0.1  # avoid div by zero
            torch.compile(fn, backend=eager, fullgraph=True)(x, tensor1, tensor2)

        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))

        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4]", L_tensor1_: "f32[4]", L_tensor2_: "f32[4]"):
        l_x_ = L_x_
        l_tensor1_ = L_tensor1_
        l_tensor2_ = L_tensor2_

        addcdiv_: "f32[4]" = l_x_.addcdiv_(l_tensor1_, l_tensor2_, value = 0.5);  l_x_ = l_tensor1_ = l_tensor2_ = None
        return (addcdiv_,)
""",
        )

    @skipIfCrossRef
    def test_foreach_lerp_inplace_decomposition_enabled(self):
        """With decompositions enabled, foreach_lerp_ with scalar weight should decompose."""

        def fn(tensors, end_tensors, weight):
            torch._foreach_lerp_(tensors, end_tensors, weight)
            return tensors

        eager = EagerAndRecordGraphs()
        with torch._dynamo.config.patch(enable_dynamo_decompositions=True):
            tensors = [torch.randn(4), torch.randn(4)]
            end_tensors = [torch.randn(4), torch.randn(4)]
            weight = torch.tensor(0.5)
            torch.compile(fn, backend=eager, fullgraph=True)(
                tensors, end_tensors, weight
            )

        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))
        # On Python 3.10, dynamo names the >= node after the op ("ge")
        # rather than the variable name ("mask").  Normalize so the
        # expected string works on every Python version.
        import re

        actual = re.sub(r"\bge\b", "mask", actual)

        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_weight_: "f32[]", L_end_tensors_0_: "f32[4]", L_end_tensors_1_: "f32[4]", L_tensors_0_: "f32[4]", L_tensors_1_: "f32[4]"):
        l_weight_ = L_weight_
        l_end_tensors_0_ = L_end_tensors_0_
        l_end_tensors_1_ = L_end_tensors_1_
        l_tensors_0_ = L_tensors_0_
        l_tensors_1_ = L_tensors_1_

        _foreach_sub = torch._foreach_sub([l_end_tensors_0_, l_end_tensors_1_], [l_tensors_0_, l_tensors_1_])
        getitem: "f32[4]" = _foreach_sub[0]
        getitem_1: "f32[4]" = _foreach_sub[1];  _foreach_sub = None
        abs_1: "f32[]" = l_weight_.abs()
        mask: "b8[]" = abs_1 >= 0.5;  abs_1 = None
        sub: "f32[]" = 1.0 - l_weight_
        neg_omw: "f32[]" = -sub;  sub = None
        w: "f32[]" = torch.where(mask, neg_omw, l_weight_);  neg_omw = l_weight_ = None
        b: "f32[4]" = torch.where(mask, l_end_tensors_0_, l_tensors_0_);  l_end_tensors_0_ = None
        b_1: "f32[4]" = torch.where(mask, l_end_tensors_1_, l_tensors_1_);  mask = l_end_tensors_1_ = None
        _foreach_addcmul_ = torch._foreach_addcmul_([b, b_1], [w, w], (getitem, getitem_1));  w = getitem = getitem_1 = _foreach_addcmul_ = None
        copy_: "f32[4]" = l_tensors_0_.copy_(b);  l_tensors_0_ = b = copy_ = None
        copy__1: "f32[4]" = l_tensors_1_.copy_(b_1);  l_tensors_1_ = b_1 = copy__1 = None
        return ()
""",  # noqa: B950
        )

    @skipIfCrossRef
    def test_foreach_lerp_inplace_decomposition_disabled(self):
        """With decompositions disabled, foreach_lerp_ should remain as the original op.

        Note: When using a tensor weight and decompositions are disabled, there can be
        graph breaks due to item() calls. This test uses a scalar weight to avoid that.
        """

        def fn(tensors, end_tensors):
            torch._foreach_lerp_(tensors, end_tensors, 0.5)
            return tensors

        eager = EagerAndRecordGraphs()
        with torch._dynamo.config.patch(enable_dynamo_decompositions=False):
            tensors = [torch.randn(4), torch.randn(4)]
            end_tensors = [torch.randn(4), torch.randn(4)]
            torch.compile(fn, backend=eager, fullgraph=True)(tensors, end_tensors)

        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))

        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_end_tensors_0_: "f32[4]", L_end_tensors_1_: "f32[4]", L_tensors_0_: "f32[4]", L_tensors_1_: "f32[4]"):
        l_end_tensors_0_ = L_end_tensors_0_
        l_end_tensors_1_ = L_end_tensors_1_
        l_tensors_0_ = L_tensors_0_
        l_tensors_1_ = L_tensors_1_

        _foreach_sub = torch._foreach_sub([l_end_tensors_0_, l_end_tensors_1_], [l_tensors_0_, l_tensors_1_])
        getitem: "f32[4]" = _foreach_sub[0]
        getitem_1: "f32[4]" = _foreach_sub[1];  _foreach_sub = None
        tensor: "f32[]" = torch.tensor(0.5, dtype = torch.float32, device = device(type='cpu'))
        sub: "f32[]" = 1.0 - tensor;  tensor = None
        neg_omw: "f32[]" = -sub;  sub = None
        copy_: "f32[4]" = l_tensors_0_.copy_(l_end_tensors_0_);  l_end_tensors_0_ = copy_ = None
        copy__1: "f32[4]" = l_tensors_1_.copy_(l_end_tensors_1_);  l_end_tensors_1_ = copy__1 = None
        _foreach_addcmul_ = torch._foreach_addcmul_([l_tensors_0_, l_tensors_1_], [neg_omw, neg_omw], (getitem, getitem_1));  l_tensors_0_ = l_tensors_1_ = neg_omw = getitem = getitem_1 = _foreach_addcmul_ = None
        return ()
""",  # noqa: B950
        )

    @skipIfCrossRef
    def test_foreach_lerp_inplace_decomposition_disabled_capture_scalar(self):
        """With decompositions disabled and capture_scalar_outputs=True, foreach_lerp_
        with scalar weight should work without graph breaks.
        """

        def fn(tensors, end_tensors):
            torch._foreach_lerp_(tensors, end_tensors, 0.5)
            return tensors

        eager = EagerAndRecordGraphs()
        with torch._dynamo.config.patch(
            enable_dynamo_decompositions=False, capture_scalar_outputs=True
        ):
            tensors = [torch.randn(4), torch.randn(4)]
            end_tensors = [torch.randn(4), torch.randn(4)]
            torch.compile(fn, backend=eager, fullgraph=True)(tensors, end_tensors)

        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))

        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_end_tensors_0_: "f32[4]", L_end_tensors_1_: "f32[4]", L_tensors_0_: "f32[4]", L_tensors_1_: "f32[4]"):
        l_end_tensors_0_ = L_end_tensors_0_
        l_end_tensors_1_ = L_end_tensors_1_
        l_tensors_0_ = L_tensors_0_
        l_tensors_1_ = L_tensors_1_

        _foreach_sub = torch._foreach_sub([l_end_tensors_0_, l_end_tensors_1_], [l_tensors_0_, l_tensors_1_])
        getitem: "f32[4]" = _foreach_sub[0]
        getitem_1: "f32[4]" = _foreach_sub[1];  _foreach_sub = None
        tensor: "f32[]" = torch.tensor(0.5, dtype = torch.float32, device = device(type='cpu'))
        sub: "f32[]" = 1.0 - tensor;  tensor = None
        neg_omw: "f32[]" = -sub;  sub = None
        copy_: "f32[4]" = l_tensors_0_.copy_(l_end_tensors_0_);  l_end_tensors_0_ = copy_ = None
        copy__1: "f32[4]" = l_tensors_1_.copy_(l_end_tensors_1_);  l_end_tensors_1_ = copy__1 = None
        _foreach_addcmul_ = torch._foreach_addcmul_([l_tensors_0_, l_tensors_1_], [neg_omw, neg_omw], (getitem, getitem_1));  l_tensors_0_ = l_tensors_1_ = neg_omw = getitem = getitem_1 = _foreach_addcmul_ = None
        return ()
""",  # noqa: B950
        )

    def test_foreach_pow_scalar_decomposition_enabled(self):
        """With decompositions enabled, foreach_pow with scalar base should decompose."""

        def fn(scalar, exps):
            return torch._foreach_pow(scalar, exps)

        eager = EagerAndRecordGraphs()
        with torch._dynamo.config.patch(enable_dynamo_decompositions=True):
            scalar = torch.tensor(2.0)
            exps = [torch.randn(4), torch.randn(4)]
            torch.compile(fn, backend=eager, fullgraph=True)(scalar, exps)

        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))

        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_scalar_: "f32[]", L_exps_0_: "f32[4]", L_exps_1_: "f32[4]"):
        l_scalar_ = L_scalar_
        l_exps_0_ = L_exps_0_
        l_exps_1_ = L_exps_1_

        _foreach_pow = torch._foreach_pow([l_scalar_, l_scalar_], [l_exps_0_, l_exps_1_]);  \
l_scalar_ = l_exps_0_ = l_exps_1_ = None
        getitem: "f32[4]" = _foreach_pow[0]
        getitem_1: "f32[4]" = _foreach_pow[1];  _foreach_pow = None
        return (getitem, getitem_1)
""",
        )

    def test_foreach_pow_scalar_decomposition_disabled(self):
        """With decompositions disabled, foreach_pow with scalar base should remain.

        Note: When using a tensor scalar and decompositions are disabled, there can be
        graph breaks due to item() calls. This test uses an actual float scalar to avoid that.
        """

        def fn(exps):
            return torch._foreach_pow(2.0, exps)

        eager = EagerAndRecordGraphs()
        with torch._dynamo.config.patch(enable_dynamo_decompositions=False):
            exps = [torch.randn(4), torch.randn(4)]
            torch.compile(fn, backend=eager, fullgraph=True)(exps)

        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))

        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_exps_0_: "f32[4]", L_exps_1_: "f32[4]"):
        l_exps_0_ = L_exps_0_
        l_exps_1_ = L_exps_1_

        _foreach_pow = torch._foreach_pow(2.0, [l_exps_0_, l_exps_1_]);  l_exps_0_ = l_exps_1_ = None
        getitem: "f32[4]" = _foreach_pow[0]
        getitem_1: "f32[4]" = _foreach_pow[1];  _foreach_pow = None
        return (getitem, getitem_1)
""",
        )

    def test_foreach_pow_scalar_decomposition_disabled_capture_scalar(self):
        """With decompositions disabled and capture_scalar_outputs=True, foreach_pow
        with scalar base should work without graph breaks.
        """

        def fn(exps):
            return torch._foreach_pow(2.0, exps)

        eager = EagerAndRecordGraphs()
        with torch._dynamo.config.patch(
            enable_dynamo_decompositions=False, capture_scalar_outputs=True
        ):
            exps = [torch.randn(4), torch.randn(4)]
            torch.compile(fn, backend=eager, fullgraph=True)(exps)

        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))

        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_exps_0_: "f32[4]", L_exps_1_: "f32[4]"):
        l_exps_0_ = L_exps_0_
        l_exps_1_ = L_exps_1_

        _foreach_pow = torch._foreach_pow(2.0, [l_exps_0_, l_exps_1_]);  l_exps_0_ = l_exps_1_ = None
        getitem: "f32[4]" = _foreach_pow[0]
        getitem_1: "f32[4]" = _foreach_pow[1];  _foreach_pow = None
        return (getitem, getitem_1)
""",
        )


if __name__ == "__main__":
    run_tests()
