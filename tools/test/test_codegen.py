from __future__ import annotations

import dataclasses
import tempfile
import typing
import unittest
from collections import defaultdict
from pathlib import Path

import yaml
from tools.autograd import gen_autograd_functions, load_derivatives
from tools.autograd.gen_python_functions import (
    group_alias_overloads,
    load_python_module_aliases,
    load_signatures,
    method_impl,
    should_generate_py_binding,
)
from tools.pyi.gen_pyi import generate_type_hints

from torchgen import dest
from torchgen.api.python import PythonSignatureAliased, PythonSignatureGroup, signature
from torchgen.api.types import CppSignatureGroup, DispatcherSignature
from torchgen.context import native_function_manager
from torchgen.gen import (
    get_native_function_declarations,
    get_native_function_schema_registrations,
    LineLoader,
    parse_native_yaml,
    static_dispatch,
)
from torchgen.model import (
    BackendIndex,
    BackendMetadata,
    DispatchKey,
    FunctionSchema,
    Location,
    NativeFunction,
    OperatorName,
)
from torchgen.native_function_generation import add_generated_native_functions
from torchgen.selective_build.selector import SelectiveBuilder


class TestGenPyi(unittest.TestCase):
    def _foreach_signatures(self):
        native_functions = parse_native_yaml(
            "aten/src/ATen/native/native_functions.yaml",
            "aten/src/ATen/native/tags.yaml",
        ).native_functions
        native_functions = list(filter(should_generate_py_binding, native_functions))
        return load_signatures(
            native_functions,
            "tools/autograd/deprecated.yaml",
            method=False,
        )

    def test_inplace_foreach_returns_input_container(self) -> None:
        native_function, _ = NativeFunction.from_yaml(
            {
                "func": "_foreach_add_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()",
                "variants": "function",
                "device_check": "NoCheck",
            },
            loc=Location(__file__, 1),
            valid_tags=set(),
        )
        group = PythonSignatureGroup(
            signature=signature(native_function, pyi=True),
            base=native_function,
            outplace=None,
        )

        hints = generate_type_hints(group)

        self.assertEqual(len(hints), 1)
        self.assertIn(
            ") -> tuple[Tensor, ...] | list[Tensor]: ...",
            hints[0],
        )

    def test_public_foreach_alias_manifest(self) -> None:
        signatures = self._foreach_signatures()
        aliases = load_python_module_aliases(
            signatures,
            "tools/autograd/python_aliases.yaml",
            module="foreach",
        )

        groups = group_alias_overloads(aliases)
        group_names = {str(name) for name in groups}
        self.assertEqual(len(aliases), 145)
        self.assertEqual(len(groups), 90)
        self.assertNotIn("powsum", {pair.signature.name for pair in aliases})
        self.assertNotIn("copy", group_names)
        self.assertNotIn("zero", group_names)
        self.assertIn("copy_", group_names)
        self.assertIn("zero_", group_names)

        add_signatures = {
            pair.signature.signature_str()
            for pair in aliases
            if pair.signature.name == "add"
        }
        self.assertIn(
            "add(TensorList inputs, TensorList other, *, Scalar alpha=1)",
            add_signatures,
        )
        norm = next(pair for pair in aliases if pair.signature.name == "norm")
        self.assertEqual(
            norm.signature.signature_str(),
            "norm(TensorList inputs, Scalar ord=2, *, ScalarType? dtype=None)",
        )
        clamp_min = {
            pair.signature.signature_str()
            for pair in aliases
            if pair.signature.name == "clamp_min"
        }
        self.assertIn("clamp_min(TensorList inputs, Scalar min)", clamp_min)
        pow_inplace = {
            pair.signature.signature_str()
            for pair in aliases
            if pair.signature.name == "pow_"
        }
        self.assertIn("pow_(TensorList inputs, Scalar exponent)", pow_inplace)

        for pair in aliases:
            self.assertIsInstance(pair.signature, PythonSignatureAliased)
            self.assertEqual(
                pair.signature.return_arg_index,
                0 if pair.signature.name.endswith("_") else None,
            )

        add_name = next(name for name in groups if str(name) == "add")
        add_impl = method_impl(
            add_name,
            "torch.foreach",
            groups[add_name],
            method=False,
        )
        self.assertIn("THPForeachVariableFunctionsModule", add_impl)
        self.assertIn("return at::_foreach_add", add_impl)

        add_inplace_name = next(name for name in groups if str(name) == "add_")
        add_inplace_impl = method_impl(
            add_inplace_name,
            "torch.foreach",
            groups[add_inplace_name],
            method=False,
        )
        self.assertIn("Python alias add_.Scalar", add_inplace_impl)
        self.assertIn("return self_tensorlist;", add_inplace_impl)

    def test_public_foreach_alias_tracks_reordered_return_argument(self) -> None:
        alias = [
            {
                "module": "foreach",
                "name": "add_.Scalar(Scalar other, Tensor(a!)[] inputs) -> ()",
                "aten": "_foreach_add_(inputs, other)",
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "python_aliases.yaml"
            path.write_text(yaml.safe_dump(alias))
            pairs = load_python_module_aliases(
                self._foreach_signatures(),
                str(path),
                module="foreach",
            )

        self.assertEqual(len(pairs), 1)
        self.assertIsInstance(pairs[0].signature, PythonSignatureAliased)
        self.assertEqual(pairs[0].signature.return_arg_index, 1)
        groups = group_alias_overloads(pairs)
        impl = method_impl(
            next(iter(groups)),
            "torch.foreach",
            next(iter(groups.values())),
            method=False,
        )
        self.assertIn("self_tensorlist = _r.args[1]", impl)

    def test_public_foreach_alias_rejects_unused_arguments(self) -> None:
        alias = [
            {
                "module": "foreach",
                "name": (
                    "add.Scalar(Tensor[] inputs, Scalar other, "
                    "bool ignored=False) -> Tensor[]"
                ),
                "aten": "_foreach_add(inputs, other)",
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "python_aliases.yaml"
            path.write_text(yaml.safe_dump(alias))
            with self.assertRaisesRegex(
                RuntimeError, "unused arguments: \\['ignored'\\]"
            ):
                load_python_module_aliases(
                    self._foreach_signatures(),
                    str(path),
                    module="foreach",
                )

    def test_public_foreach_alias_rejects_duplicate_arguments(self) -> None:
        alias = [
            {
                "module": "foreach",
                "name": "add.Scalar(Tensor[] inputs, Scalar other) -> Tensor[]",
                "aten": "_foreach_add(inputs, inputs)",
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "python_aliases.yaml"
            path.write_text(yaml.safe_dump(alias))
            with self.assertRaisesRegex(
                RuntimeError, r"duplicate arguments: \['inputs'\]"
            ):
                load_python_module_aliases(
                    self._foreach_signatures(),
                    str(path),
                    module="foreach",
                )


class TestCreateDerivative(unittest.TestCase):
    def test_named_grads(self) -> None:
        schema = FunctionSchema.parse(
            "func(Tensor a, Tensor b) -> (Tensor x, Tensor y)"
        )
        native_function = dataclasses.replace(DEFAULT_NATIVE_FUNCTION, func=schema)

        derivative = load_derivatives.create_derivative(
            native_function,
            formula="func_backward(grad_x, grad_y)",
            var_names=(),
            available_named_gradients=["grad_x", "grad_y"],
        )
        self.assertSetEqual(derivative.named_gradients, {"grad_x", "grad_y"})

    def test_non_differentiable_output(self) -> None:
        specification = "func(Tensor a, Tensor b) -> (Tensor x, bool y, Tensor z)"
        schema = FunctionSchema.parse(specification)
        native_function = dataclasses.replace(DEFAULT_NATIVE_FUNCTION, func=schema)

        _, differentiability_info = load_derivatives.create_differentiability_info(
            defn_dict={
                "name": specification,
                "dispatch": {"Default": {"a": "grads[0]", "b": "grads[2]"}},
            },
            functions_by_signature={schema.signature(): [native_function]},
            functions_by_schema={specification: native_function},
            op_counter=typing.Counter[str](),
            used_dispatch_keys=set(),
        )

        self.assertSequenceEqual(
            differentiability_info["Default"].available_named_gradients,
            # grad_y is not present because y is a
            # bool and thus not differentiable.
            ["grad_x", "grad_z"],
        )

    def test_indexed_grads(self) -> None:
        schema = FunctionSchema.parse(
            "func(Tensor a, Tensor b) -> (Tensor x, Tensor y)"
        )
        native_function = dataclasses.replace(DEFAULT_NATIVE_FUNCTION, func=schema)

        derivative = load_derivatives.create_derivative(
            native_function,
            formula="func_backward(grads[0], grads[1])",
            var_names=(),
            available_named_gradients=["grad_x", "grad_y"],
        )
        self.assertSetEqual(derivative.named_gradients, set())

    def test_named_grads_and_indexed_grads(self) -> None:
        specification = "func(Tensor a, Tensor b) -> (Tensor x, Tensor y)"
        schema = FunctionSchema.parse(specification)
        native_function = dataclasses.replace(DEFAULT_NATIVE_FUNCTION, func=schema)

        with self.assertRaisesRegex(
            RuntimeError, 'illegally mixes use of "grad_RETURN_NAME"'
        ):
            load_derivatives.create_differentiability_info(
                defn_dict={
                    "name": specification,
                    # Uh-oh, the derivatives reference gradients by
                    # name and by index.
                    "dispatch": {
                        "Default": {
                            "a": "grad_x",
                            "b": "grads[1]",
                        }
                    },
                },
                functions_by_signature={schema.signature(): [native_function]},
                functions_by_schema={specification: native_function},
                op_counter=typing.Counter[str](),
                used_dispatch_keys=set(),
            )


class TestGenAutogradFunctions(unittest.TestCase):
    def test_non_differentiable_output_invalid_type(self) -> None:
        specification = "func(Tensor a, Tensor b) -> (Tensor x, bool y, Tensor z)"
        schema = FunctionSchema.parse(specification)
        native_function = dataclasses.replace(DEFAULT_NATIVE_FUNCTION, func=schema)

        _, differentiability_info = load_derivatives.create_differentiability_info(
            defn_dict={
                "name": specification,
                "dispatch": {
                    "Default": {
                        "a": "grad_x",
                        "b": "grad_z",
                    }
                },
            },
            functions_by_signature={schema.signature(): [native_function]},
            functions_by_schema={specification: native_function},
            op_counter=typing.Counter[str](),
            used_dispatch_keys=set(),
        )
        definition = gen_autograd_functions.process_function(
            differentiability_info["Default"],
            gen_autograd_functions.FUNCTION_DEFINITION,
        )
        # grad_z should map to grads[1], not grads[2] because output 1
        # (y) is not differentiable.
        if "grad_z = grads[2]" in definition:
            raise AssertionError("grad_z should not map to grads[2]")
        if "grad_z = grads[1]" not in definition:
            raise AssertionError("grad_z should map to grads[1]")

    def test_non_differentiable_output_output_differentiability(self) -> None:
        specification = "func(Tensor a, Tensor b) -> (Tensor x, Tensor y, Tensor z)"
        schema = FunctionSchema.parse(specification)
        native_function = dataclasses.replace(DEFAULT_NATIVE_FUNCTION, func=schema)

        _, differentiability_info = load_derivatives.create_differentiability_info(
            defn_dict={
                "name": specification,
                "dispatch": {
                    "Default": {
                        "a": "grad_x",
                        "b": "grad_z",
                    },
                    "AutogradNestedTensor": {
                        "a": "grad_z",
                        "b": "grad_x",
                    },
                },
                "output_differentiability": [True, False, True],
            },
            functions_by_signature={schema.signature(): [native_function]},
            functions_by_schema={specification: native_function},
            op_counter=typing.Counter[str](),
            used_dispatch_keys=set(),
        )
        default_definition = gen_autograd_functions.process_function(
            differentiability_info["Default"],
            gen_autograd_functions.FUNCTION_DEFINITION,
        )
        # grad_z should map to grads[1], not grads[2] because output 1
        # (y) is not differentiable.
        if "grad_z = grads[2]" in default_definition:
            raise AssertionError(
                "grad_z should not map to grads[2] in default_definition"
            )
        if "grad_z = grads[1]" not in default_definition:
            raise AssertionError("grad_z should map to grads[1] in default_definition")

        nested_tensor_definition = gen_autograd_functions.process_function(
            differentiability_info["AutogradNestedTensor"],
            gen_autograd_functions.FUNCTION_DEFINITION,
        )
        if "grad_z = grads[2]" in nested_tensor_definition:
            raise AssertionError(
                "grad_z should not map to grads[2] in nested_tensor_definition"
            )
        if "grad_z = grads[1]" not in nested_tensor_definition:
            raise AssertionError(
                "grad_z should map to grads[1] in nested_tensor_definition"
            )

    def test_register_bogus_dispatch_key(self) -> None:
        specification = "func(Tensor a, Tensor b) -> (Tensor x, bool y, Tensor z)"
        schema = FunctionSchema.parse(specification)
        native_function = dataclasses.replace(DEFAULT_NATIVE_FUNCTION, func=schema)

        with self.assertRaisesRegex(
            RuntimeError,
            "Invalid dispatch key AutogradRandomTensor in derivatives.yaml for",
        ):
            load_derivatives.create_differentiability_info(
                defn_dict={
                    "name": specification,
                    "dispatch": {
                        "Default": {
                            "a": "grad_x",
                            "b": "grad_z",
                        },
                        "AutogradRandomTensor": {
                            "a": "grad_x",
                            "b": "grad_z",
                        },
                    },
                },
                functions_by_signature={schema.signature(): [native_function]},
                functions_by_schema={specification: native_function},
                op_counter=typing.Counter[str](),
                used_dispatch_keys=set(),
            )


class TestGenSchemaRegistration(unittest.TestCase):
    def setUp(self) -> None:
        self.selector = SelectiveBuilder.get_nop_selector()
        self.custom_native_function, _ = NativeFunction.from_yaml(
            {"func": "custom::func() -> bool"},
            loc=Location(__file__, 1),
            valid_tags=set(),
        )
        (
            self.fragment_custom_native_function,
            _,
        ) = NativeFunction.from_yaml(
            {"func": "quantized_decomposed::func() -> bool"},
            loc=Location(__file__, 1),
            valid_tags=set(),
        )

    def test_default_namespace_schema_registration_code_valid(self) -> None:
        native_functions = [DEFAULT_NATIVE_FUNCTION]
        registrations, _ = get_native_function_schema_registrations(
            native_functions=native_functions,
            schema_selector=self.selector,
        )
        self.assertEqual(registrations, ['m.def("func() -> bool", {});\n'])

    def test_custom_namespace_schema_registration_code_valid(self) -> None:
        _, registrations = get_native_function_schema_registrations(
            native_functions=[self.custom_native_function],
            schema_selector=self.selector,
        )
        self.assertEqual(
            registrations,
            """
TORCH_LIBRARY(custom, m) {
  m.def("func() -> bool", {});

};""",
        )

    def test_fragment_custom_namespace_schema_registration_code_valid(self) -> None:
        """Sometimes we want to extend an existing namespace, for example quantized
        namespace, which is already defined in native/quantized/library.cpp
        """
        _, registrations = get_native_function_schema_registrations(
            native_functions=[self.fragment_custom_native_function],
            schema_selector=self.selector,
        )
        self.assertEqual(
            registrations,
            """
TORCH_LIBRARY_FRAGMENT(quantized_decomposed, m) {
  m.def("func() -> bool", {});

};""",
        )

    def test_mixed_namespace_schema_registration_code_valid(self) -> None:
        (
            aten_registrations,
            custom_registrations,
        ) = get_native_function_schema_registrations(
            native_functions=[DEFAULT_NATIVE_FUNCTION, self.custom_native_function],
            schema_selector=self.selector,
        )
        self.assertEqual(aten_registrations, ['m.def("func() -> bool", {});\n'])
        self.assertEqual(
            custom_registrations,
            """
TORCH_LIBRARY(custom, m) {
  m.def("func() -> bool", {});

};""",
        )

    def test_3_namespaces_schema_registration_code_valid(self) -> None:
        custom2_native_function, _ = NativeFunction.from_yaml(
            {"func": "custom2::func() -> bool"},
            loc=Location(__file__, 1),
            valid_tags=set(),
        )
        (
            aten_registrations,
            custom_registrations,
        ) = get_native_function_schema_registrations(
            native_functions=[
                DEFAULT_NATIVE_FUNCTION,
                self.custom_native_function,
                custom2_native_function,
            ],
            schema_selector=self.selector,
        )
        self.assertEqual(aten_registrations, ['m.def("func() -> bool", {});\n'])
        self.assertEqual(
            custom_registrations,
            """
TORCH_LIBRARY(custom, m) {
  m.def("func() -> bool", {});

};
TORCH_LIBRARY(custom2, m) {
  m.def("func() -> bool", {});

};""",
        )


class TestGenNativeFunctionDeclaration(unittest.TestCase):
    def setUp(self) -> None:
        self.op_1_native_function, op_1_backend_index = NativeFunction.from_yaml(
            {"func": "op_1() -> bool", "dispatch": {"CPU": "kernel_1"}},
            loc=Location(__file__, 1),
            valid_tags=set(),
        )
        self.op_2_native_function, op_2_backend_index = NativeFunction.from_yaml(
            {
                "func": "op_2() -> bool",
                "dispatch": {"CPU": "kernel_2", "QuantizedCPU": "custom::kernel_3"},
            },
            loc=Location(__file__, 1),
            valid_tags=set(),
        )

        backend_indices: dict[DispatchKey, dict[OperatorName, BackendMetadata]] = {
            DispatchKey.CPU: {},
            DispatchKey.QuantizedCPU: {},
        }
        BackendIndex.grow_index(backend_indices, op_1_backend_index)
        BackendIndex.grow_index(backend_indices, op_2_backend_index)
        self.backend_indices = {
            k: BackendIndex(
                dispatch_key=k,
                use_out_as_primary=True,
                external=False,
                device_guard=False,
                index=backend_indices[k],
            )
            for k in backend_indices
        }

    def test_native_function_declaration_1_op_2_ns_error(self) -> None:
        with self.assertRaises(AssertionError):
            get_native_function_declarations(
                grouped_native_functions=[
                    self.op_1_native_function,
                    self.op_2_native_function,
                ],
                backend_indices=self.backend_indices,
                native_function_decl_gen=dest.compute_native_function_declaration,
            )

    def test_native_function_declaration_1_op_1_ns_valid(self) -> None:
        self.assertIsInstance(self.op_1_native_function, NativeFunction)
        declaration = get_native_function_declarations(
            grouped_native_functions=[
                self.op_1_native_function,
            ],
            backend_indices=self.backend_indices,
            native_function_decl_gen=dest.compute_native_function_declaration,
        )
        target = """
namespace at {
namespace native {
TORCH_API bool kernel_1();
} // namespace native
} // namespace at
        """
        self.assertEqual("\n".join(declaration), target)


# Test for native_function_generation
class TestNativeFunctionGeneratrion(unittest.TestCase):
    def setUp(self) -> None:
        self.native_functions: list[NativeFunction] = []
        self.backend_indices: dict[DispatchKey, dict[OperatorName, BackendMetadata]] = (
            defaultdict(dict)
        )
        yaml_entry = """
- func: op(Tensor self) -> Tensor
  dispatch:
    CompositeExplicitAutograd: op
  autogen: op.out
        """
        es = yaml.load(yaml_entry, Loader=LineLoader)
        self.one_return_func, m = NativeFunction.from_yaml(
            es[0], loc=Location(__file__, 1), valid_tags=set()
        )

        BackendIndex.grow_index(self.backend_indices, m)

        self.two_returns_func, two_returns_backend_index = NativeFunction.from_yaml(
            {
                "func": "op_2() -> (Tensor, Tensor)",
                "dispatch": {"CPU": "kernel_1"},
                "autogen": "op_2.out",
            },
            loc=Location(__file__, 1),
            valid_tags=set(),
        )
        BackendIndex.grow_index(self.backend_indices, two_returns_backend_index)

        self.core_func, core_func_index = NativeFunction.from_yaml(
            {
                "func": "op_3.vec(Tensor input, SymInt[]? output_size, float[]? scale_factors) -> Tensor",
                "autogen": "op_3.vec_out",
                "tags": ["core"],
            },
            loc=Location(__file__, 1),
            valid_tags={"core"},
        )
        BackendIndex.grow_index(self.backend_indices, core_func_index)

    def test_functional_variant_autogen_out_variant(self) -> None:
        native_functions = [self.one_return_func]
        add_generated_native_functions(native_functions, self.backend_indices)
        self.assertEqual(len(native_functions), 2)
        self.assertEqual(
            str(native_functions[1].func),
            "op.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
        )
        op_name = native_functions[1].func.name
        backend_metadata = self.backend_indices[DispatchKey.CompositeExplicitAutograd][
            op_name
        ]
        self.assertEqual(backend_metadata.kernel, "op_out")

    def test_functional_variant_autogen_out_variant_two_returns(self) -> None:
        native_functions = [self.two_returns_func]
        add_generated_native_functions(native_functions, self.backend_indices)
        self.assertEqual(len(native_functions), 2)
        self.assertEqual(
            str(native_functions[1].func),
            "op_2.out(*, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))",
        )
        op_name = native_functions[1].func.name
        backend_metadata = self.backend_indices[DispatchKey.CompositeExplicitAutograd][
            op_name
        ]
        self.assertEqual(backend_metadata.kernel, "op_2_out")

    def test_functional_variant_autogen_out_variant_core(self) -> None:
        """
        Tests autogen of out variants for core-tageed ops that are CompositeImplicitAutograd.
        """
        native_functions = [self.core_func]
        add_generated_native_functions(native_functions, self.backend_indices)
        print(native_functions)
        self.assertEqual(len(native_functions), 2)
        self.assertEqual(
            str(native_functions[1].func),
            "op_3.vec_out(Tensor input, SymInt[]? output_size, float[]? scale_factors, *, Tensor(a!) out) -> Tensor(a!)",
        )


# Test for static_dispatch
class TestStaticDispatchGeneratrion(unittest.TestCase):
    def setUp(self) -> None:
        self.backend_indices: dict[DispatchKey, dict[OperatorName, BackendMetadata]] = (
            defaultdict(dict)
        )
        yaml_entry = """
- func: op.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
  dispatch:
    CompositeExplicitAutograd: op
        """
        es = yaml.load(yaml_entry, Loader=LineLoader)
        self.one_return_func, m = NativeFunction.from_yaml(
            es[0], loc=Location(__file__, 1), valid_tags=set()
        )

        BackendIndex.grow_index(self.backend_indices, m)
        dispatch_key = DispatchKey.CompositeExplicitAutograd
        self.assertTrue(dispatch_key in self.backend_indices)
        self.indices = [
            BackendIndex(
                dispatch_key=dispatch_key,
                use_out_as_primary=True,
                external=False,
                device_guard=False,
                index=self.backend_indices[dispatch_key],
            )
        ]

    def test_op_with_1_backend_generates_static_dispatch(self) -> None:
        disp_sig = DispatcherSignature.from_schema(self.one_return_func.func)
        with native_function_manager(self.one_return_func):
            out = static_dispatch(
                sig=disp_sig,
                f=self.one_return_func,
                backend_indices=self.indices,
            )
        self.assertEqual(
            out, "return at::compositeexplicitautograd::op_out(out, self);"
        )

    def test_op_with_cpp_sig_generates_static_dispatch(self) -> None:
        sig_group = CppSignatureGroup.from_native_function(
            self.one_return_func,
            method=False,
            fallback_binding=self.one_return_func.manual_cpp_binding,
        )
        # cpp signature puts out at the front
        with native_function_manager(self.one_return_func):
            out = static_dispatch(
                sig=sig_group.signature,
                f=self.one_return_func,
                backend_indices=self.indices,
            )
        self.assertEqual(
            out, "return at::compositeexplicitautograd::op_out(out, self);"
        )


# Represents the most basic NativeFunction. Use dataclasses.replace()
# to edit for use.
DEFAULT_NATIVE_FUNCTION, _ = NativeFunction.from_yaml(
    {"func": "func() -> bool"},
    loc=Location(__file__, 1),
    valid_tags=set(),
)


if __name__ == "__main__":
    unittest.main()
