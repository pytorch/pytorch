import unittest
from types import ModuleType

from torchgen import local
from torchgen.api import cpp as aten_cpp, types as aten_types
from torchgen.api.types import (
    ArgName,
    BaseCType,
    ConstRefCType,
    MutRefCType,
    NamedCType,
)
from torchgen.executorch.api import et_cpp as et_cpp, types as et_types
from torchgen.executorch.api.unboxing import Unboxing
from torchgen.model import BaseTy, BaseType, ListType, OptionalType, Type


def aten_argumenttype_type_wrapper(
    t: Type, *, mutable: bool, binds: ArgName, remove_non_owning_ref_types: bool = False
) -> NamedCType:
    return aten_cpp.argumenttype_type(
        t,
        mutable=mutable,
        binds=binds,
        remove_non_owning_ref_types=remove_non_owning_ref_types,
    )


ATEN_UNBOXING = Unboxing(argument_type_gen=aten_argumenttype_type_wrapper)
ET_UNBOXING = Unboxing(argument_type_gen=et_cpp.argumenttype_type)


class TestUnboxing(unittest.TestCase):
    """
    Could use torch.testing._internal.common_utils to reduce boilerplate.
    GH CI job doesn't build torch before running tools unit tests, hence
    manually adding these parametrized tests.
    """

    @local.parametrize(
        use_const_ref_for_mutable_tensors=False, use_ilistref_for_tensor_lists=False
    )
    def test_symint_argument_translate_ctype_aten(self) -> None:
        # test if `SymInt[]` JIT argument can be translated into C++ argument correctly.
        # should be `IntArrayRef` due to the fact that Executorch doesn't use symint sig.

        # pyre-fixme[16]: `enum.Enum` has no attribute `SymInt`
        # pyre-fixme[19]: Call `BaseType.__init__` expects 0 positional arguments, 1 was provided.
        symint_list_type = ListType(elem=BaseType(BaseTy.SymInt), size=None)

        out_name, ctype, _, _ = ATEN_UNBOXING.argumenttype_evalue_convert(
            t=symint_list_type, arg_name="size", mutable=False
        )

        self.assertEqual(out_name, "size_list_out")
        self.assertIsInstance(ctype, BaseCType)
        # pyre-fixme[16]:
        self.assertEqual(ctype, aten_types.BaseCType(aten_types.intArrayRefT))

    @local.parametrize(
        use_const_ref_for_mutable_tensors=False, use_ilistref_for_tensor_lists=False
    )
    def test_symint_argument_translate_ctype_executorch(self) -> None:
        # test if `SymInt[]` JIT argument can be translated into C++ argument correctly.
        # should be `IntArrayRef` due to the fact that Executorch doesn't use symint sig.

        # pyre-fixme[16]: `enum.Enum` has no attribute `SymInt`
        # pyre-fixme[19]: Call `BaseType.__init__` expects 0 positional arguments, 1 was provided.
        symint_list_type = ListType(elem=BaseType(BaseTy.SymInt), size=None)

        out_name, ctype, _, _ = ET_UNBOXING.argumenttype_evalue_convert(
            t=symint_list_type, arg_name="size", mutable=False
        )

        self.assertEqual(out_name, "size_list_out")
        self.assertIsInstance(ctype, et_types.ArrayRefCType)
        # pyre-fixme[16]:
        self.assertEqual(
            ctype, et_types.ArrayRefCType(elem=BaseCType(aten_types.longT))
        )

    @local.parametrize(
        use_const_ref_for_mutable_tensors=False, use_ilistref_for_tensor_lists=False
    )
    def _test_const_tensor_argument_translate_ctype(
        self, unboxing: Unboxing, types: ModuleType
    ) -> None:
        # pyre-fixme[16]: `enum.Enum` has no attribute `Tensor`
        # pyre-fixme[19]: Call `BaseType.__init__` expects 0 positional arguments, 1 was provided.
        tensor_type = BaseType(BaseTy.Tensor)

        out_name, ctype, _, _ = unboxing.argumenttype_evalue_convert(
            t=tensor_type, arg_name="self", mutable=False
        )

        self.assertEqual(out_name, "self_base")
        # pyre-fixme[16]:
        self.assertEqual(ctype, ConstRefCType(BaseCType(types.tensorT)))

    def test_const_tensor_argument_translate_ctype_aten(self) -> None:
        self._test_const_tensor_argument_translate_ctype(ATEN_UNBOXING, aten_types)

    def test_const_tensor_argument_translate_ctype_executorch(self) -> None:
        self._test_const_tensor_argument_translate_ctype(ET_UNBOXING, et_types)

    @local.parametrize(
        use_const_ref_for_mutable_tensors=False, use_ilistref_for_tensor_lists=False
    )
    def _test_mutable_tensor_argument_translate_ctype(
        self, unboxing: Unboxing, types: ModuleType
    ) -> None:
        # pyre-fixme[16]: `enum.Enum` has no attribute `Tensor`
        # pyre-fixme[19]: Call `BaseType.__init__` expects 0 positional arguments, 1 was provided.
        tensor_type = BaseType(BaseTy.Tensor)

        out_name, ctype, _, _ = unboxing.argumenttype_evalue_convert(
            t=tensor_type, arg_name="out", mutable=True
        )

        self.assertEqual(out_name, "out_base")
        # pyre-fixme[16]:
        self.assertEqual(ctype, MutRefCType(BaseCType(types.tensorT)))

    def test_mutable_tensor_argument_translate_ctype_aten(self) -> None:
        self._test_mutable_tensor_argument_translate_ctype(ATEN_UNBOXING, aten_types)

    def test_mutable_tensor_argument_translate_ctype_executorch(self) -> None:
        self._test_mutable_tensor_argument_translate_ctype(ET_UNBOXING, et_types)

    @local.parametrize(
        use_const_ref_for_mutable_tensors=False, use_ilistref_for_tensor_lists=False
    )
    def _test_tensor_list_argument_translate_ctype(
        self, unboxing: Unboxing, types: ModuleType
    ) -> None:
        # pyre-fixme[16]: `enum.Enum` has no attribute `Tensor`
        # pyre-fixme[19]: Call `BaseType.__init__` expects 0 positional arguments, 1 was provided.
        tensor_list_type = ListType(elem=BaseType(BaseTy.Tensor), size=None)

        out_name, ctype, _, _ = unboxing.argumenttype_evalue_convert(
            t=tensor_list_type, arg_name="out", mutable=True
        )

        self.assertEqual(out_name, "out_list_out")
        # pyre-fixme[16]:
        self.assertEqual(ctype, BaseCType(types.tensorListT))

    def test_tensor_list_argument_translate_ctype_aten(self) -> None:
        self._test_tensor_list_argument_translate_ctype(ATEN_UNBOXING, aten_types)

    def test_tensor_list_argument_translate_ctype_executorch(self) -> None:
        self._test_tensor_list_argument_translate_ctype(ET_UNBOXING, et_types)

    @local.parametrize(
        use_const_ref_for_mutable_tensors=False, use_ilistref_for_tensor_lists=False
    )
    def _test_optional_int_argument_translate_ctype(
        self, unboxing: Unboxing, types: ModuleType
    ) -> None:
        # pyre-fixme[16]: `enum.Enum` has no attribute `Tensor`
        # pyre-fixme[19]: Call `BaseType.__init__` expects 0 positional arguments, 1 was provided.
        optional_int_type = OptionalType(elem=BaseType(BaseTy.int))

        out_name, ctype, _, _ = unboxing.argumenttype_evalue_convert(
            t=optional_int_type, arg_name="something", mutable=True
        )

        self.assertEqual(out_name, "something_opt_out")
        # pyre-fixme[16]:
        self.assertEqual(ctype, types.OptionalCType(BaseCType(types.longT)))

    def test_optional_int_argument_translate_ctype_aten(self) -> None:
        self._test_optional_int_argument_translate_ctype(ATEN_UNBOXING, aten_types)

    def test_optional_int_argument_translate_ctype_executorch(self) -> None:
        self._test_optional_int_argument_translate_ctype(ET_UNBOXING, et_types)
