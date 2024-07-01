from __future__ import annotations

import dataclasses
import typing
import unittest
from collections import defaultdict

import yaml
from tools.autograd import gen_autograd_functions, load_derivatives

from torchgen import dest
from torchgen.api.types import CppSignatureGroup, DispatcherSignature
from torchgen.context import native_function_manager
from torchgen.gen import (
    get_native_function_declarations,
    get_native_function_schema_registrations,
    LineLoader,
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
        assert "grad_z = grads[2]" not in definition
        assert "grad_z = grads[1]" in definition

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
        assert "grad_z = grads[2]" not in default_definition
        assert "grad_z = grads[1]" in default_definition

        nested_tensor_definition = gen_autograd_functions.process_function(
            differentiability_info["AutogradNestedTensor"],
            gen_autograd_functions.FUNCTION_DEFINITION,
        )
        assert "grad_z = grads[2]" not in nested_tensor_definition
        assert "grad_z = grads[1]" in nested_tensor_definition

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
