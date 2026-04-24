# Owner(s): ["module: custom-operators"]
import torch
from torch._library.utils import is_inplace
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


@skipIfTorchDynamo("custom operator tests not applicable to dynamo")
class TestInplaceTag(TestCase):
    def setUp(self):
        self.lib = torch.library.Library("_TestInplaceTag", "FRAGMENT")  # noqa: TOR901

    def tearDown(self):
        self.lib._destroy()
        super().tearDown()

    def test_basic_inplace(self):
        self.lib.define(
            "add_(Tensor(a!) self, Tensor other) -> Tensor(a!)",
            tags=[torch.Tag.inplace],
        )
        self.assertTrue(is_inplace(torch.ops._TestInplaceTag.add_.default))

    def test_is_inplace_native(self):
        # Hand-written inplace op
        self.assertTrue(is_inplace(torch.ops.aten.abs_.default))
        self.assertFalse(is_inplace(torch.ops.aten.abs.default))
        # Out op is not inplace
        self.assertFalse(is_inplace(torch.ops.aten.abs.out))
        # Functional op is not inplace
        self.assertFalse(is_inplace(torch.ops.aten.add.Tensor))

    def test_no_positional_args(self):
        with self.assertRaisesRegex(ValueError, "at least one positional argument"):
            self.lib.define(
                "no_args(*, Tensor(a!) out) -> Tensor(a!)",
                tags=[torch.Tag.inplace],
            )

    def test_first_arg_not_mutable(self):
        with self.assertRaisesRegex(
            ValueError, "first positional argument to be mutable"
        ):
            self.lib.define(
                "not_mutable(Tensor self, Tensor other) -> Tensor",
                tags=[torch.Tag.inplace],
            )

    def test_first_arg_not_tensor(self):
        with self.assertRaisesRegex(
            ValueError, "first positional argument to be a Tensor"
        ):
            self.lib.define(
                "not_tensor(Tensor(a!)[] self) -> ()",
                tags=[torch.Tag.inplace],
            )

    def test_wrong_return_count(self):
        with self.assertRaisesRegex(ValueError, "must return exactly one value"):
            self.lib.define(
                "no_return(Tensor(a!) self) -> ()",
                tags=[torch.Tag.inplace],
            )

    def test_wrong_return_count_multiple(self):
        with self.assertRaisesRegex(ValueError, "must return exactly one value"):
            self.lib.define(
                "multi_return(Tensor(a!) self) -> (Tensor(a!), Tensor)",
                tags=[torch.Tag.inplace],
            )

    def test_return_not_aliased(self):
        with self.assertRaisesRegex(ValueError, "return the first mutable argument"):
            self.lib.define(
                "bad_alias(Tensor(a!) self) -> Tensor",
                tags=[torch.Tag.inplace],
            )

    def test_return_wrong_alias(self):
        with self.assertRaisesRegex(ValueError, "return the first mutable argument"):
            self.lib.define(
                "wrong_alias(Tensor(a!) self, Tensor(b!) other) -> Tensor(b!)",
                tags=[torch.Tag.inplace],
            )

    def test_additional_mutable_positional_arg(self):
        with self.assertRaisesRegex(
            ValueError, "must only mutate the first positional"
        ):
            self.lib.define(
                "extra_mut(Tensor(a!) self, Tensor(b!) other) -> Tensor(a!)",
                tags=[torch.Tag.inplace],
            )

    def test_additional_mutable_kwarg(self):
        with self.assertRaisesRegex(
            ValueError, "must only mutate the first positional"
        ):
            self.lib.define(
                "extra_kwarg(Tensor(a!) self, *, Tensor(b!) out) -> Tensor(a!)",
                tags=[torch.Tag.inplace],
            )


if __name__ == "__main__":
    run_tests()
