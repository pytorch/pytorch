# Owner(s): ["module: functorch"]
from functools import wraps
import unittest

from torch.testing._internal.common_utils import (
    _TestParametrizer,
    TestCase,
    run_tests,
    instantiate_parametrized_tests,
)

from torch._C import (
    _dispatch_get_registrations_for_dispatch_key as get_registrations_for_dispatch_key,
)

xfail_functorch_batched = {
    "aten::flatten.using_ints",
    "aten::gather_backward",
    "aten::imag",
    "aten::is_nonzero",
    "aten::isfinite",
    "aten::isreal",
    "aten::item",
    "aten::linalg_matrix_power",
    "aten::linalg_matrix_rank.atol_rtol_float",
    "aten::linalg_matrix_rank.atol_rtol_tensor",
    "aten::linalg_pinv",
    "aten::linalg_pinv.atol_rtol_float",
    "aten::linalg_slogdet",
    "aten::linear",
    "aten::log_sigmoid",
    "aten::log_softmax.int",
    "aten::logdet",
    "aten::masked_select_backward",
    "aten::movedim.intlist",
    "aten::one_hot",
    "aten::real",
    "aten::relu6",
    "aten::relu6_",
    "aten::selu",
    "aten::selu_",
    "aten::silu_backward",
    "aten::special_xlogy",
    "aten::special_xlogy.other_scalar",
    "aten::special_xlogy.self_scalar",
    "aten::tensor_split.indices",
    "aten::tensor_split.sections",
    "aten::to.device",
    "aten::to.dtype",
    "aten::to.dtype_layout",
    "aten::to.other",
    "aten::upsample_bicubic2d.vec",
    "aten::upsample_bilinear2d.vec",
    "aten::upsample_linear1d.vec",
    "aten::upsample_nearest1d.vec",
    "aten::upsample_nearest2d.vec",
    "aten::upsample_nearest3d.vec",
    "aten::upsample_trilinear3d.vec",
    "aten::where",
}

xfail_functorch_batched_decomposition = {
    "aten::diagonal_copy",
    "aten::is_same_size",
    "aten::t",
    "aten::t_",
    "aten::unfold_copy",
}


class dispatch_registrations(_TestParametrizer):
    def __init__(self, dispatch_key: str, xfails: set):
        self.registrations = sorted(get_registrations_for_dispatch_key(dispatch_key))
        self.xfails = xfails

    def _parametrize_test(self, test, generic_cls, device_cls):
        for registration in self.registrations:

            @wraps(test)
            def test_wrapper(*args, **kwargs):
                return test(*args, **kwargs)

            if registration in self.xfails:
                test_wrapper = unittest.expectedFailure(test_wrapper)

            yield (test_wrapper, f"[{registration}]", {"registration": registration})


CompositeImplicitAutogradRegistrations = set(
    get_registrations_for_dispatch_key("CompositeImplicitAutograd")
)
FuncTorchBatchedRegistrations = set(
    get_registrations_for_dispatch_key("FuncTorchBatched")
)


class TestFunctorchDispatcher(TestCase):
    @dispatch_registrations("CompositeImplicitAutograd", xfail_functorch_batched)
    def test_register_a_batching_rule_for_composite_implicit_autograd(
        self, registration
    ):
        assert registration not in FuncTorchBatchedRegistrations, (
            f"You've added a batching rule for a CompositeImplicitAutograd operator {registration}. "
            "The correct way to add vmap support for it is to put it into BatchRulesDecomposition to "
            "reuse the CompositeImplicitAutograd decomposition"
        )

    @dispatch_registrations(
        "FuncTorchBatchedDecomposition", xfail_functorch_batched_decomposition
    )
    def test_register_functorch_batched_decomposition(self, registration):
        assert registration in CompositeImplicitAutogradRegistrations, (
            f"The registrations in BatchedDecompositions.cpp must be for CompositeImplicitAutograd "
            f"operations. If your operation {registration} is not CompositeImplicitAutograd, "
            "then please register it to the FuncTorchBatched key in another file."
        )


instantiate_parametrized_tests(TestFunctorchDispatcher)

if __name__ == "__main__":
    run_tests()
