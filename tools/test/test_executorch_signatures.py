import unittest

from torchgen.executorch.api.types import ExecutorchCppSignature
from torchgen.local import parametrize
from torchgen.model import Location, NativeFunction


DEFAULT_NATIVE_FUNCTION, _ = NativeFunction.from_yaml(
    {"func": "foo.out(Tensor input, *, Tensor(a!) out) -> Tensor(a!)"},
    loc=Location(__file__, 1),
    valid_tags=set(),
)


class ExecutorchCppSignatureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sig = ExecutorchCppSignature.from_native_function(DEFAULT_NATIVE_FUNCTION)

    def test_runtime_signature_contains_runtime_context(self) -> None:
        # test if `KernelRuntimeContext` argument exists in `RuntimeSignature`
        with parametrize(
            use_const_ref_for_mutable_tensors=False, use_ilistref_for_tensor_lists=False
        ):
            args = self.sig.arguments(include_context=True)
            self.assertEqual(len(args), 3)
            self.assertTrue(any(a.name == "context" for a in args))

    def test_runtime_signature_does_not_contain_runtime_context(self) -> None:
        # test if `KernelRuntimeContext` argument is missing in `RuntimeSignature`
        with parametrize(
            use_const_ref_for_mutable_tensors=False, use_ilistref_for_tensor_lists=False
        ):
            args = self.sig.arguments(include_context=False)
            self.assertEqual(len(args), 2)
            self.assertFalse(any(a.name == "context" for a in args))

    def test_runtime_signature_declaration_correct(self) -> None:
        with parametrize(
            use_const_ref_for_mutable_tensors=False, use_ilistref_for_tensor_lists=False
        ):
            decl = self.sig.decl(include_context=True)
            self.assertEqual(
                decl,
                (
                    "torch::executor::Tensor & foo_outf("
                    "torch::executor::KernelRuntimeContext & context, "
                    "const torch::executor::Tensor & input, "
                    "torch::executor::Tensor & out)"
                ),
            )
            no_context_decl = self.sig.decl(include_context=False)
            self.assertEqual(
                no_context_decl,
                (
                    "torch::executor::Tensor & foo_outf("
                    "const torch::executor::Tensor & input, "
                    "torch::executor::Tensor & out)"
                ),
            )
