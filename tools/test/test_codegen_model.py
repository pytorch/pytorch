# Owner(s): ["module: codegen"]

import expecttest
import unittest
import yaml
import textwrap

from torchgen.model import NativeFunctionsGroup, DispatchKey
import torchgen.dest as dest
import torchgen.gen as gen
from torchgen.gen import LineLoader, parse_native_yaml_struct


class TestCodegenModel(expecttest.TestCase):
    def assertParseErrorInline(self, yaml_str: str, expect: str) -> None:
        es = yaml.load(yaml_str, Loader=LineLoader)
        try:
            parse_native_yaml_struct(es)
        except AssertionError as e:
            # hack to strip out the context
            msg, _ = str(e).split("  in ", 2)
            self.assertExpectedInline("\n".join(textwrap.wrap(msg)), expect, skip=1)
            return
        self.fail(msg="Did not raise when expected to")

    def assertUfuncErrorInline(self, yaml_str: str, expect: str) -> None:
        # parse a single structured group out of the yaml to g
        es = yaml.load(yaml_str, Loader=LineLoader)
        parsed_yaml = parse_native_yaml_struct(es)
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


if __name__ == "__main__":
    unittest.main()
