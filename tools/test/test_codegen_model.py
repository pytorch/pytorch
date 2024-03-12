# Owner(s): ["module: codegen"]

import textwrap
import unittest
from typing import cast

import expecttest

import torchgen.dest as dest
import torchgen.gen as gen
import yaml
from torchgen.gen import LineLoader, parse_native_yaml_struct
from torchgen.model import (
    Annotation,
    CustomClassType,
    DispatchKey,
    NativeFunctionsGroup,
    Type,
)


class TestCodegenModel(expecttest.TestCase):
    def assertParseErrorInline(self, yaml_str: str, expect: str) -> None:
        es = yaml.load(yaml_str, Loader=LineLoader)
        try:
            parse_native_yaml_struct(es, set())
        except AssertionError as e:
            # hack to strip out the context
            msg, _ = str(e).split("  in ", 2)
            self.assertExpectedInline("\n".join(textwrap.wrap(msg)), expect, skip=1)
            return
        self.fail(msg="Did not raise when expected to")

    def assertUfuncErrorInline(self, yaml_str: str, expect: str) -> None:
        # parse a single structured group out of the yaml to g
        es = yaml.load(yaml_str, Loader=LineLoader)
        parsed_yaml = parse_native_yaml_struct(es, set())
        native_functions, backend_indices = (
            parsed_yaml.native_functions,
            parsed_yaml.backend_indices,
        )
        grouped_native_functions = gen.get_grouped_native_functions(native_functions)
        assert len(grouped_native_functions) == 1
        g = grouped_native_functions[0]
        assert isinstance(g, NativeFunctionsGroup)
        assert g.out.ufunc_inner_loop
        # this is not ufunc codegen per se, but it does some basic sanity tests for
        # ufunc generation
        gen.compute_meta_function_declaration(g)
        dest.compute_native_function_declaration(g, backend_indices[DispatchKey.CPU])
        dest.compute_native_function_declaration(g, backend_indices[DispatchKey.CUDA])
        try:
            # the real kahuna
            dest.compute_ufunc_cpu(g)
            dest.compute_ufunc_cpu_kernel(g)
            dest.compute_ufunc_cuda(g)
        except AssertionError as e:
            # hack to strip out the context
            msg, _ = str(e).split("  in ", 2)
            self.assertExpectedInline("\n".join(textwrap.wrap(msg)), expect, skip=1)
            return
        self.fail(msg="Did not raise when expected to")

    # NB: indent is hardcoded to be two here, so format your yaml accordingly
    binop_out = (
        "func: binop.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"
    )
    ti_binop_out = f"""{binop_out}
  structured: True
  structured_inherits: TensorIteratorBase"""
    ti_binop = """func: binop(Tensor self, Tensor other) -> Tensor
  structured_delegate: binop.out
"""

    ti_unop_out = """func: unop.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
  structured: True
  structured_inherits: TensorIteratorBase"""
    ti_unop = """func: unop(Tensor self) -> Tensor
  structured_delegate: unop.out
"""

    def test_nonstructured_ufunc(self) -> None:
        yaml_str = f"""\
- {self.binop_out}
  ufunc_inner_loop:
    Generic: binop (Bool)
"""
        self.assertParseErrorInline(
            yaml_str,
            """\
ufunc must be structured""",
        )

    def test_overlapping_ufunc_and_dispatch(self) -> None:
        yaml_str = f"""\
- {self.ti_binop_out}
  ufunc_inner_loop:
    Generic: binop (Bool)
  dispatch:
    CPU: binop_cpu
"""
        self.assertParseErrorInline(
            yaml_str,
            """\
ufunc should not have explicit dispatch entry for CPU""",
        )

    # See https://github.com/pytorch/pytorch/pull/65851#discussion_r810238456
    @unittest.expectedFailure
    def test_scalaronly_shadowed(self) -> None:
        yaml_str = f"""\
- {self.ti_binop_out}
  ufunc_inner_loop:
    Generic: binop (Bool)
    ScalarOnly: binop (Bool)
"""
        self.assertParseErrorInline(
            yaml_str,
            """\
""",
        )

    def test_conflicting_ufunc(self) -> None:
        yaml_str = f"""\
- {self.ti_binop_out}
  ufunc_inner_loop:
    Generic: binop (Bool)
    ScalarOnly: binop_scalar (Bool)
- {self.ti_binop}
"""
        self.assertUfuncErrorInline(
            yaml_str,
            """\
ScalarOnly and Generic must have same ufunc name""",
        )

    def test_invalid_cudafunctoronself_for_binary_op(self) -> None:
        yaml_str = f"""\
- {self.ti_unop_out}
  ufunc_inner_loop:
    Generic: unop (All)
    CUDAFunctorOnSelf: unop_self_cuda (All)
- {self.ti_unop}
"""
        self.assertUfuncErrorInline(
            yaml_str,
            """\
cannot use CUDAFunctorOnSelf on non-binary function""",
        )

    def test_parse_custom_class_type(self) -> None:
        custom_class_name = "namespace_foo.class_bar"
        custom_class_name_with_prefix = f"__torch__.torch.classes.{custom_class_name}"
        custom_class_type = cast(
            CustomClassType, Type.parse(custom_class_name_with_prefix)
        )
        self.assertTrue(isinstance(custom_class_type, CustomClassType))
        self.assertEqual(custom_class_name, custom_class_type.class_name)
        self.assertEqual(custom_class_name_with_prefix, str(custom_class_type))


class TestAnnotation(expecttest.TestCase):
    def test_single_alias_no_write(self) -> None:
        a = Annotation.parse("a")
        self.assertEqual(a.alias_set, tuple("a"))
        self.assertFalse(a.is_write)
        self.assertEqual(a.alias_set_after, tuple())

    def test_single_alias_is_write(self) -> None:
        a = Annotation.parse("a!")
        self.assertEqual(a.alias_set, tuple("a"))
        self.assertTrue(a.is_write)
        self.assertEqual(a.alias_set_after, tuple())

    def test_single_alias_is_write_to_wildcard(self) -> None:
        a = Annotation.parse("a! -> *")
        self.assertEqual(a.alias_set, tuple("a"))
        self.assertTrue(a.is_write)
        self.assertEqual(a.alias_set_after, tuple("*"))

    def test_alias_set(self) -> None:
        a = Annotation.parse("a|b")
        self.assertEqual(a.alias_set, ("a", "b"))

    def test_alias_set_is_write_raises_exception(self) -> None:
        with self.assertRaisesRegex(
            AssertionError, r"alias set larger than 1 is not mutable"
        ):
            Annotation.parse("a|b!")

    def test_single_alias_is_write_to_alias_set(self) -> None:
        a = Annotation.parse("a! -> a|b")
        self.assertEqual(a.alias_set, tuple("a"))
        self.assertTrue(a.is_write)
        self.assertEqual(a.alias_set_after, ("a", "b"))

    def test_before_and_after_alias_set_larger_than_1_raises_exception(self) -> None:
        with self.assertRaisesRegex(
            AssertionError,
            r"before alias set and after alias set cannot be larger than 1 at the same time",
        ):
            Annotation.parse("a|b -> c|d")


if __name__ == "__main__":
    unittest.main()
