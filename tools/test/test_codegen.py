from __future__ import annotations

import dataclasses
import typing
import unittest
import unittest.mock
from collections import defaultdict

import yaml
from tools.autograd import gen_autograd_functions, load_derivatives
from tools.pyi.gen_pyi import generate_type_hints

from torchgen import dest
from torchgen.aoti.fallback_ops import inductor_fallback_ops
from torchgen.api.python import PythonSignatureGroup, signature
from torchgen.api.types import CppSignatureGroup, DispatcherSignature
from torchgen.context import native_function_manager
from torchgen.dest import native_functions as native_functions_dest
from torchgen.dest.native_functions import (
    dll_export_macro_for_kernel,
    validate_cpu_dll_cuda_kernels,
)
from torchgen.gen import (
    get_native_function_declarations,
    get_native_function_schema_registrations,
    LineLoader,
    static_dispatch,
)
from torchgen.gen_aoti_c_shim import (
    _get_earliest_torch_version_for_op_variant,
    declaration_definition_cache,
    gen_declaration_and_definition,
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
from torchgen.utils import Target


class TestAOTIShimVersions(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        declaration_definition_cache.clear()
        self.addCleanup(declaration_definition_cache.clear)

    def test_histogram_backend_version(self) -> None:
        schema = FunctionSchema.parse("histogram(Tensor self) -> Tensor")
        for op in ["aten.histogram.bin_ct", "aten._histogramdd_from_bin_cts.default"]:
            for device in ["cpu", "cuda"]:
                with self.subTest(op=op, device=device):
                    declaration_definition_cache.clear()
                    declaration, definition = gen_declaration_and_definition(
                        schema, device, "at::histogram", inductor_fallback_ops[op]
                    )
                    self.assertEqual(
                        "#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_16_0"
                        in declaration,
                        device == "cuda",
                    )
                    self.assertNotIn("#if", definition)
                    self.assertIn(f"aoti_torch_{device}_histogram", definition)

    def test_global_and_variant_versions(self) -> None:
        schema = FunctionSchema.parse("op(Tensor self, int x=0) -> Tensor")
        metadata = {
            "since": "TORCH_VERSION_2_10_0",
            "since_cuda": "TORCH_VERSION_2_15_0",
            "v2": {
                "new_args": ["x"],
                "since": "TORCH_VERSION_2_12_0",
                "since_cuda": "TORCH_VERSION_2_16_0",
            },
        }
        for device, versions in [
            ("cpu", ["2_10_0", "2_12_0"]),
            ("cuda", ["2_15_0", "2_16_0"]),
        ]:
            with self.subTest(device=device):
                declaration, _ = gen_declaration_and_definition(
                    schema, device, "at::op", metadata
                )
                for version in versions:
                    self.assertIn(
                        f"#if TORCH_FEATURE_VERSION >= TORCH_VERSION_{version}",
                        declaration,
                    )
                self.assertIn(f"aoti_torch_{device}_op_v2", declaration)
                self.assertEqual(declaration.count("#if"), 2)

    def test_backend_floor_for_variants(self) -> None:
        schema = FunctionSchema.parse("op(Tensor self, int x=0) -> Tensor")
        cases: list[tuple[str, dict[str, str], str, str | None]] = [
            ("2_16_0", {}, "2_16_0", None),
            ("2_16_0", {"since": "2_12_0"}, "2_16_0", "2_12_0"),
            ("2_16_0", {"since": "2_17_0"}, "2_17_0", "2_17_0"),
            ("2_16_0", {"since_cuda": "2_12_0"}, "2_16_0", None),
            ("2_16_0", {"since_cuda": "2_17_0"}, "2_17_0", None),
            (
                "2_16_0",
                {"since": "2_17_0", "since_cuda": "2_15_0"},
                "2_16_0",
                "2_17_0",
            ),
            ("2_9_0", {"since": "2_12_0"}, "2_12_0", "2_12_0"),
            ("2_12_0", {"since": "2_9_0"}, "2_12_0", "2_9_0"),
            ("2_16_2", {"since": "2_16_10"}, "2_16_10", "2_16_10"),
            ("2_16_10", {"since": "2_16_2"}, "2_16_10", "2_16_2"),
            ("3_0_0", {"since": "2_16_0"}, "3_0_0", "2_16_0"),
            ("2_16_0", {"since": "3_0_0"}, "3_0_0", "3_0_0"),
        ]
        for base, overrides, cuda_version, cpu_version in cases:
            variant: dict[str, list[str] | str] = {"new_args": ["x"]}
            variant.update({k: f"TORCH_VERSION_{v}" for k, v in overrides.items()})
            metadata: dict[str, str | dict[str, list[str] | str]] = {
                "since_cuda": f"TORCH_VERSION_{base}",
                "v2": variant,
            }
            for device, version in [("cuda", cuda_version), ("cpu", cpu_version)]:
                with self.subTest(base=base, overrides=overrides, device=device):
                    expected = f"TORCH_VERSION_{version}" if version else None
                    self.assertEqual(
                        _get_earliest_torch_version_for_op_variant(metadata, 2, device),
                        expected,
                    )
                    declaration_definition_cache.clear()
                    declaration, definition = gen_declaration_and_definition(
                        schema, device, "at::op", metadata
                    )
                    variant_decl = (
                        f"AOTI_TORCH_EXPORT AOTITorchError aoti_torch_{device}_op_v2("
                    )
                    if expected:
                        self.assertIn(
                            f"#if TORCH_FEATURE_VERSION >= {expected}\n{variant_decl}",
                            declaration,
                        )
                    self.assertEqual(
                        declaration.count("#if"),
                        int(device == "cuda") + int(expected is not None),
                    )
                    self.assertNotIn("#if", definition)

    def test_backend_floor_does_not_inherit_sibling_versions(self) -> None:
        metadata: dict[str, str | dict[str, list[str] | str]] = {
            "since_cuda": "TORCH_VERSION_2_16_0",
            "v2": {"new_args": ["x"], "since_cuda": "TORCH_VERSION_2_18_0"},
            "v3": {"new_args": ["y"], "since": "TORCH_VERSION_2_17_0"},
        }
        for device, versions in [
            ("cuda", ["2_16_0", "2_18_0", "2_17_0"]),
            ("cpu", [None, None, "2_17_0"]),
        ]:
            for op_version, version in enumerate(versions, 1):
                with self.subTest(device=device, op_version=op_version):
                    self.assertEqual(
                        _get_earliest_torch_version_for_op_variant(
                            metadata, op_version, device
                        ),
                        f"TORCH_VERSION_{version}" if version else None,
                    )
        self.assertIsNone(_get_earliest_torch_version_for_op_variant({}, 1, "cuda"))

    def test_global_version_is_not_inherited_by_variants(self) -> None:
        for version in [None, "TORCH_VERSION_2_12_0"]:
            variant: dict[str, list[str] | str] = {"new_args": ["x"]}
            if version is not None:
                variant["since"] = version
            metadata: dict[str, str | dict[str, list[str] | str]] = {
                "since": "TORCH_VERSION_2_17_0",
                "v2": variant,
            }
            for device in ["cpu", "cuda"]:
                with self.subTest(version=version, device=device):
                    self.assertEqual(
                        _get_earliest_torch_version_for_op_variant(metadata, 2, device),
                        version,
                    )

    def test_invalid_backend_version(self) -> None:
        cases: list[dict[str, str | dict[str, list[str] | str]]] = [
            {"since_cdua": "TORCH_VERSION_2_16_0"},
            {"since_cuda": "2.16"},
            {"since_cuda": "2.16", "v2": {"new_args": ["x"]}},
            {"since_cdua": "TORCH_VERSION_2_16_0", "v2": {"new_args": ["x"]}},
            {"v2": {"new_args": ["x"], "since_cdua": "TORCH_VERSION_2_16_0"}},
        ]
        for metadata in cases:
            with self.subTest(metadata=metadata), self.assertRaises(AssertionError):
                _get_earliest_torch_version_for_op_variant(
                    metadata, 2 if "v2" in metadata else 1, "cpu"
                )


class TestGenPyi(unittest.TestCase):
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


def _build_backend_indices_from_yaml(
    yaml_entries: list[dict],
) -> tuple[list[NativeFunction], dict[DispatchKey, BackendIndex]]:
    raw_indices: dict[DispatchKey, dict[OperatorName, BackendMetadata]] = defaultdict(
        dict
    )
    native_functions: list[NativeFunction] = []
    for entry in yaml_entries:
        func, indices = NativeFunction.from_yaml(
            entry,
            loc=Location(__file__, 1),
            valid_tags=set(),
        )
        native_functions.append(func)
        BackendIndex.grow_index(raw_indices, indices)
    backend_indices = {
        k: BackendIndex(
            dispatch_key=k,
            use_out_as_primary=True,
            external=False,
            device_guard=False,
            index=raw_indices[k],
        )
        for k in raw_indices
    }
    return native_functions, backend_indices


class TestNativeDeclDllExportMacro(unittest.TestCase):
    def test_cuda_distinct_kernel_uses_torch_cuda_cpp_api(self) -> None:
        native_function, backend_indices = _build_backend_indices_from_yaml(
            [
                {
                    "func": "cuda_only_op(Tensor self) -> Tensor",
                    "dispatch": {"CUDA": "cuda_only_kernel"},
                }
            ]
        )
        decl = dest.compute_native_function_declaration(
            native_function[0], backend_indices[DispatchKey.CUDA]
        )
        self.assertEqual(len(decl), 1)
        self.assertTrue(decl[0].startswith("TORCH_CUDA_CPP_API "))
        self.assertIn("cuda_only_kernel(", decl[0])

    def test_allowlisted_cuda_kernel_uses_torch_api(self) -> None:
        native_function, backend_indices = _build_backend_indices_from_yaml(
            [
                {
                    "func": "count_nonzero_cuda(Tensor self) -> Tensor",
                    "dispatch": {"CUDA": "count_nonzero_cuda"},
                }
            ]
        )
        decl = dest.compute_native_function_declaration(
            native_function[0], backend_indices[DispatchKey.CUDA]
        )
        self.assertEqual(len(decl), 1)
        self.assertTrue(decl[0].startswith("TORCH_API "))
        self.assertIn("count_nonzero_cuda(", decl[0])

    def test_shared_cpu_cuda_kernel_merges_to_single_torch_api_decl(self) -> None:
        native_functions, backend_indices = _build_backend_indices_from_yaml(
            [
                {
                    "func": "shared_op(Tensor self) -> Tensor",
                    "dispatch": {"CPU": "shared_op", "CUDA": "shared_op"},
                }
            ]
        )
        declaration = get_native_function_declarations(
            grouped_native_functions=native_functions,
            backend_indices=backend_indices,
            native_function_decl_gen=dest.compute_native_function_declaration,
        )
        joined = "\n".join(declaration)
        self.assertEqual(joined.count("shared_op("), 1)
        self.assertIn("TORCH_API ", joined)
        self.assertIn("shared_op(", joined)
        self.assertNotIn("TORCH_CUDA_CPP_API", joined)

    def test_allowlisted_quantized_cuda_kernel_overrides_only_quantized_cuda(
        self,
    ) -> None:
        native_function, backend_indices = _build_backend_indices_from_yaml(
            [
                {
                    "func": "masked_fill_.Scalar(Tensor(a!) self, Tensor mask, Scalar value) -> Tensor(a!)",
                    "dispatch": {
                        "CPU": "masked_fill__cpu",
                        "CUDA": "masked_fill__cuda",
                        "QuantizedCUDA": "masked_fill__quantized_cuda",
                    },
                }
            ]
        )
        func = native_function[0]
        self.assertEqual(
            dll_export_macro_for_kernel(backend_indices[DispatchKey.CPU], func),
            "TORCH_API",
        )
        self.assertEqual(
            dll_export_macro_for_kernel(backend_indices[DispatchKey.CUDA], func),
            "TORCH_CUDA_CPP_API",
        )
        self.assertEqual(
            dll_export_macro_for_kernel(
                backend_indices[DispatchKey.QuantizedCUDA], func
            ),
            "TORCH_API",
        )

    def test_index_put_impl_quantized_cuda_kernel_uses_torch_api(self) -> None:
        native_function, backend_indices = _build_backend_indices_from_yaml(
            [
                {
                    "func": "_index_put_impl_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False, bool unsafe=False) -> Tensor(a!)",
                    "dispatch": {
                        "CPU": "_index_put_impl_",
                        "CUDA": "_index_put_impl_",
                        "QuantizedCUDA": "_index_put_impl_quantized_cuda_",
                    },
                }
            ]
        )
        func = native_function[0]
        decl = dest.compute_native_function_declaration(
            func, backend_indices[DispatchKey.QuantizedCUDA]
        )
        self.assertEqual(len(decl), 1)
        self.assertTrue(decl[0].startswith("TORCH_API "))
        self.assertIn("_index_put_impl_quantized_cuda_(", decl[0])

    def test_xpu_kernel_uses_torch_xpu_api(self) -> None:
        native_function, backend_indices = _build_backend_indices_from_yaml(
            [
                {
                    "func": "xpu_only_op(Tensor self) -> Tensor",
                    "dispatch": {"XPU": "xpu_only_kernel"},
                }
            ]
        )
        decl = dest.compute_native_function_declaration(
            native_function[0], backend_indices[DispatchKey.XPU]
        )
        self.assertEqual(len(decl), 1)
        self.assertTrue(decl[0].startswith("TORCH_XPU_API "))
        self.assertIn("xpu_only_kernel(", decl[0])

    def test_cuda_namespaced_declaration_uses_torch_cuda_cpp_api(self) -> None:
        native_function, backend_indices = _build_backend_indices_from_yaml(
            [
                {
                    "func": "cuda_only_op(Tensor self) -> Tensor",
                    "dispatch": {"CUDA": "cuda_only_kernel"},
                }
            ]
        )
        decl = dest.RegisterDispatchKey(
            backend_index=backend_indices[DispatchKey.CUDA],
            target=Target.NAMESPACED_DECLARATION,
            selector=SelectiveBuilder.get_nop_selector(),
            rocm=False,
            symint=True,
            class_method_name=None,
            skip_dispatcher_op_registration=False,
        )(native_function[0])
        joined = "".join(decl)
        self.assertIn("TORCH_CUDA_CPP_API ", joined)
        self.assertNotIn("TORCH_API at::Tensor cuda_only_op", joined)

    def test_validate_cpu_dll_cuda_kernels_accepts_reachable_entry(self) -> None:
        _, backend_indices = _build_backend_indices_from_yaml(
            [
                {
                    "func": "count_nonzero_cuda(Tensor self) -> Tensor",
                    "dispatch": {"CUDA": "count_nonzero_cuda"},
                }
            ]
        )
        with unittest.mock.patch.object(
            native_functions_dest,
            "_CPU_DLL_CUDA_KERNELS",
            frozenset({"count_nonzero_cuda"}),
        ):
            validate_cpu_dll_cuda_kernels(backend_indices)

    def test_validate_cpu_dll_cuda_kernels_rejects_stale_entry(self) -> None:
        _, backend_indices = _build_backend_indices_from_yaml(
            [
                {
                    "func": "count_nonzero_cuda(Tensor self) -> Tensor",
                    "dispatch": {"CUDA": "count_nonzero_cuda"},
                }
            ]
        )
        with unittest.mock.patch.object(
            native_functions_dest,
            "_CPU_DLL_CUDA_KERNELS",
            frozenset({"kernel_removed_from_native_functions_yaml"}),
        ):
            with self.assertRaisesRegex(
                AssertionError, "kernel_removed_from_native_functions_yaml"
            ):
                validate_cpu_dll_cuda_kernels(backend_indices)


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
