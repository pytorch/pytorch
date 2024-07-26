from __future__ import annotations

import os
import tempfile
import unittest

import yaml

from torchgen.executorch.model import ETKernelIndex, ETKernelKey
from torchgen.gen import LineLoader
from torchgen.gen_executorch import (
    ComputeCodegenUnboxedKernels,
    gen_functions_declarations,
    parse_yaml_files,
    translate_native_yaml,
)
from torchgen.model import (
    BackendIndex,
    BackendMetadata,
    DispatchKey,
    Location,
    NativeFunction,
    OperatorName,
)
from torchgen.selective_build.selector import SelectiveBuilder


TEST_YAML = """
- func: add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured: True
  structured_inherits: TensorIteratorBase
  ufunc_inner_loop:
    Generic: add (AllAndComplex, BFloat16, Half, ComplexHalf)
    ScalarOnly: add (Bool)
  dispatch:
    SparseCPU: add_out_sparse_cpu
    SparseCUDA: add_out_sparse_cuda
    SparseCsrCPU: add_out_sparse_csr_cpu
    SparseCsrCUDA: add_out_sparse_csr_cuda
    MkldnnCPU: mkldnn_add_out
    MPS: add_out_mps

- func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  device_check: NoCheck   # TensorIterator
  structured_delegate: add.out
  variants: function, method
  dispatch:
    SparseCPU, SparseCUDA: add_sparse
    SparseCsrCPU, SparseCsrCUDA: add_sparse_csr
    MkldnnCPU: mkldnn_add
    ZeroTensor: add_zerotensor
    NestedTensorCPU, NestedTensorCUDA: NestedTensor_add_Tensor
  tags: core

- func: mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured: True
  structured_inherits: TensorIteratorBase
  dispatch:
    CPU, CUDA: mul_out
    MPS: mul_out_mps
    SparseCPU: mul_out_sparse_cpu
    SparseCUDA: mul_out_sparse_cuda
    SparseCsrCPU, SparseCsrCUDA: mul_out_sparse_csr
    MkldnnCPU: mkldnn_mul_out

- func: mul.Tensor(Tensor self, Tensor other) -> Tensor
  device_check: NoCheck   # TensorIterator
  structured_delegate: mul.out
  variants: function, method
  dispatch:
    SparseCPU, SparseCUDA: mul_sparse
    SparseCsrCPU, SparseCsrCUDA: mul_sparse_csr
    MkldnnCPU: mkldnn_mul
    ZeroTensor: mul_zerotensor
    NestedTensorCPU, NestedTensorCUDA: NestedTensor_mul_Tensor
  tags: core

"""


TEST_KERNEL_YAML = """
- func: add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured: True
  structured_inherits: TensorIteratorBase
  ufunc_inner_loop:
    Generic: add (AllAndComplex, BFloat16, Half, ComplexHalf)
    ScalarOnly: add (Bool)
  type_alias:
    T0: [Float, Double]
    T1: [Double, Int]
  dim_order_alias:
    D0: [0, 1, 2, 3]
    D1: [0, 3, 2, 1]
  kernels:
    - arg_meta: null
      kernel_name: default_impl
    - arg_meta:
        self: [T0, D0]
        other: [T1, D0]
        out: [T0, D0]
      kernel_name: test_impl
    - arg_meta:
        self: [T1, D0]
        other: [T1, D1]
        out: [T0, D1]
      kernel_name: test_impl_2

- func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  device_check: NoCheck   # TensorIterator
  structured_delegate: add.out
  variants: function, method
  tags: core

- func: mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured: True
  structured_inherits: TensorIteratorBase
  type_alias:
    T0: [Float]
    T1: [Double]
  dim_order_alias:
    D0: [0, 1, 2, 3]
  kernels:
    - arg_meta: null
      kernel_name: default_impl
    - arg_meta:
        self: [T0, D0]
        other: [T1, D0]
        out: [T0, D0]
      kernel_name: test_impl

- func: mul.Tensor(Tensor self, Tensor other) -> Tensor
  device_check: NoCheck   # TensorIterator
  structured_delegate: mul.out
  variants: function, method
  tags: core

"""


class TestParseNativeYaml(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

        self.aten_yaml_path = os.path.join(self.temp_dir, "test_native_functions.yaml")
        with open(self.aten_yaml_path, "w") as f:
            f.write(TEST_YAML)
        self.ops_yaml_path = os.path.join(self.temp_dir, "test.yaml")
        self.tags_yaml_path = os.path.join(self.temp_dir, "tags.yaml")
        with open(self.tags_yaml_path, "w") as f:
            f.write(
                """
- tag: core
  desc: test
            """
            )
        with open(self.ops_yaml_path, "w") as f:
            f.write(
                """
- op: add.out
  device_check: NoCheck   # TensorIterator
  dispatch:
    CPU: torch::executor::add_out_kernel

- op: mul.out
  device_check: NoCheck   # TensorIterator
  dispatch:
    CPU: torch::executor::mul_out_kernel
                """
            )

    def test_translate_native_yaml_writes_correct_data(self) -> None:
        out_yaml_path = os.path.join(self.temp_dir, "out.yaml")
        with open(out_yaml_path, "w") as out_file:
            translate_native_yaml(
                tags_yaml_path=self.tags_yaml_path,
                aten_yaml_path=self.aten_yaml_path,
                native_yaml_path=self.ops_yaml_path,
                use_aten_lib=False,
                out_file=out_file,
            )
        with open(out_yaml_path) as out_file:
            es = yaml.load(out_file, Loader=LineLoader)
        self.assertTrue(all("func" in e for e in es))
        self.assertTrue(all(e.get("variants") == "function" for e in es))

        # Check that kernel fields aren't introduced in yaml
        for e in es:
            self.assertFalse({"kernels", "type_alias", "dim_order_alias"} < e.keys())

    def test_parse_yaml_files(self) -> None:
        custom_ops_yaml_path = None
        selector = SelectiveBuilder.get_nop_selector()
        use_aten_lib = False

        parsed_yaml, custom_ops_parsed_yaml = parse_yaml_files(
            aten_yaml_path=self.aten_yaml_path,
            tags_yaml_path=self.tags_yaml_path,
            native_yaml_path=self.ops_yaml_path,
            custom_ops_yaml_path=custom_ops_yaml_path,
            selector=selector,
            use_aten_lib=use_aten_lib,
        )

        # Just the default kernel entry
        expected_kernel_entry = {"add.out": 1, "mul.out": 1}
        self.assertTrue(len(parsed_yaml.native_functions) == len(expected_kernel_entry))

        op_entries = parsed_yaml.kernel_index.index
        for op_name, kernel_mapping in op_entries.items():
            self.assertTrue(
                len(kernel_mapping) == expected_kernel_entry.pop(str(op_name))
            )

        self.assertTrue(len(expected_kernel_entry) == 0)

    def tearDown(self) -> None:
        import shutil

        try:
            shutil.rmtree(self.temp_dir)
        except OSError:
            pass


class TestParseKernelYamlFiles(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

        self.aten_kernel_yaml_path = os.path.join(
            self.temp_dir, "test_kernel_native_functions.yaml"
        )
        with open(self.aten_kernel_yaml_path, "w") as f:
            f.write(TEST_KERNEL_YAML)
        self.ops_yaml_path = os.path.join(self.temp_dir, "test.yaml")
        self.tags_yaml_path = os.path.join(self.temp_dir, "tags.yaml")
        with open(self.tags_yaml_path, "w") as f:
            f.write(
                """
- tag: core
  desc: test
            """
            )
        with open(self.ops_yaml_path, "w") as f:
            f.write(
                """
- op: add.out
  device_check: NoCheck   # TensorIterator
  dispatch:
    CPU: torch::executor::add_out_kernel

- op: mul.out
  device_check: NoCheck   # TensorIterator
  dispatch:
    CPU: torch::executor::mul_out_kernel
                """
            )

    def test_translate_kernel_native_yaml_writes_correct_data(self) -> None:
        out_yaml_path = os.path.join(self.temp_dir, "out2.yaml")
        with open(out_yaml_path, "w") as out_file:
            translate_native_yaml(
                tags_yaml_path=self.tags_yaml_path,
                aten_yaml_path=self.aten_kernel_yaml_path,
                native_yaml_path=self.ops_yaml_path,
                use_aten_lib=False,
                out_file=out_file,
            )
        with open(out_yaml_path) as out_file:
            es = yaml.load(out_file, Loader=LineLoader)
        self.assertTrue(all("func" in e for e in es))
        self.assertTrue(all(e.get("variants") == "function" for e in es))

        # Check persistence of kernel fields in yaml
        for e in es:
            self.assertTrue({"kernels", "type_alias", "dim_order_alias"} < e.keys())

    def test_parse_yaml_files(self) -> None:
        custom_ops_yaml_path = None
        selector = SelectiveBuilder.get_nop_selector()
        use_aten_lib = False

        parsed_yaml, custom_ops_parsed_yaml = parse_yaml_files(
            aten_yaml_path=self.aten_kernel_yaml_path,
            tags_yaml_path=self.tags_yaml_path,
            native_yaml_path=self.ops_yaml_path,
            custom_ops_yaml_path=custom_ops_yaml_path,
            selector=selector,
            use_aten_lib=use_aten_lib,
        )

        expected_kernel_entry = {"add.out": 9, "mul.out": 2}
        self.assertTrue(len(parsed_yaml.native_functions) == len(expected_kernel_entry))

        op_entries = parsed_yaml.kernel_index.index
        for op_name, kernel_mapping in op_entries.items():
            self.assertTrue(
                len(kernel_mapping) == expected_kernel_entry.pop(str(op_name))
            )

        self.assertTrue(len(expected_kernel_entry) == 0)

    def tearDown(self) -> None:
        import shutil

        try:
            shutil.rmtree(self.temp_dir)
        except OSError:
            pass


class TestGenFunctionsDeclarations(unittest.TestCase):
    def setUp(self) -> None:
        (
            self.custom_1_native_function,
            custom_1_backend_index,
        ) = NativeFunction.from_yaml(
            {"func": "custom_1::op_1() -> bool", "dispatch": {"CPU": "kernel_1"}},
            loc=Location(__file__, 1),
            valid_tags=set(),
        )
        (
            self.custom_2_native_function,
            custom_2_backend_index,
        ) = NativeFunction.from_yaml(
            {
                "func": "custom_2::op_2() -> bool",
                "dispatch": {"CPU": "kernel_2"},
            },
            loc=Location(__file__, 1),
            valid_tags=set(),
        )
        (
            self.custom_3_native_function,
            custom_3_backend_index,
        ) = NativeFunction.from_yaml(
            {
                "func": "custom_3::op_3(Tensor(a!) self, Tensor x) -> Tensor(a!)",
                "dispatch": {"CPU": "kernel_3"},
                "variants": "method",
            },
            loc=Location(__file__, 1),
            valid_tags=set(),
        )

        backend_indices: dict[DispatchKey, dict[OperatorName, BackendMetadata]] = {
            DispatchKey.CPU: {},
            DispatchKey.QuantizedCPU: {},
        }
        BackendIndex.grow_index(backend_indices, custom_1_backend_index)
        BackendIndex.grow_index(backend_indices, custom_2_backend_index)
        self.static_dispatch_idx = [
            BackendIndex(
                dispatch_key=k,
                use_out_as_primary=True,
                external=False,
                device_guard=False,
                index=backend_indices[k],
            )
            for k in backend_indices
        ]
        self.kernel_index = ETKernelIndex.from_backend_indices(backend_indices)

    def test_operators_with_different_namespaces_are_grouped_correctly(self) -> None:
        declarations = gen_functions_declarations(
            native_functions=[
                self.custom_1_native_function,
                self.custom_2_native_function,
            ],
            kernel_index=self.kernel_index,
            selector=SelectiveBuilder.get_nop_selector(),
            use_aten_lib=False,
        )
        self.assertTrue(
            """
namespace custom_1 {

// custom_1::op_1() -> bool
TORCH_API inline bool op_1(torch::executor::KernelRuntimeContext & context) {
    return ::at::native::kernel_1(context);
}

} // namespace custom_1
"""
            in declarations
        )

        self.assertTrue(
            """
namespace custom_2 {

// custom_2::op_2() -> bool
TORCH_API inline bool op_2(torch::executor::KernelRuntimeContext & context) {
    return ::at::native::kernel_2(context);
}

} // namespace custom_2
        """
            in declarations
        )

    def test_aten_lib_has_context_arg(self) -> None:
        declarations = gen_functions_declarations(
            native_functions=[
                self.custom_1_native_function,
            ],
            kernel_index=self.kernel_index,
            selector=SelectiveBuilder.get_nop_selector(),
            use_aten_lib=True,
        )
        self.assertTrue(
            """
namespace custom_1 {

// custom_1::op_1() -> bool
TORCH_API inline bool op_1(torch::executor::KernelRuntimeContext & context) {
    return at::op_1();
}

} // namespace custom_1
        """
            in declarations
        )

    def test_aten_lib_method_variant(self) -> None:
        declarations = gen_functions_declarations(
            native_functions=[
                self.custom_3_native_function,
            ],
            kernel_index=self.kernel_index,
            selector=SelectiveBuilder.get_nop_selector(),
            use_aten_lib=True,
        )
        self.assertTrue(
            """
namespace custom_3 {

// custom_3::op_3(Tensor(a!) self, Tensor x) -> Tensor(a!)
TORCH_API inline at::Tensor & op_3(torch::executor::KernelRuntimeContext & context, at::Tensor & self, const at::Tensor & x) {
    return self.op_3(x);
}

} // namespace custom_3
        """
            in declarations
        )


class TestComputeCodegenUnboxedKernels(unittest.TestCase):
    def setUp(self) -> None:
        (
            self.native_function_no_kern,
            _,
        ) = NativeFunction.from_yaml(
            {
                "func": "custom_1::op_1() -> bool",
                "dispatch": {"CPU": "unused_kernel_1"},
            },
            loc=Location(__file__, 1),
            valid_tags=set(),
        )

        self.default_kernel_key = ETKernelKey(default=True)
        self.default_backend_metadata = BackendMetadata(
            "default_kernel", False, "at::native"
        )
        self.default_kernel_entry = (
            [self.default_kernel_key],
            self.default_backend_metadata,
        )

    def test_codegen_unboxed_specialized(self) -> None:
        specialized_kernel_key = ETKernelKey.gen_from_yaml(
            {"self": ("T0", "D0"), "other": ("T0", "D0"), "out": ("T0", "D0")},
            {"T0": ["Double"]},
            {"D0": [0, 1, 2, 3]},
        )
        selector = SelectiveBuilder.from_yaml_dict(
            {
                "include_all_operators": True,
                "et_kernel_metadata": {
                    "custom_1::op_1": ["v1/7;0,1,2,3|7;0,1,2,3|7;0,1,2,3"]
                },
            }
        )
        use_aten_lib = False
        entry = (
            self.native_function_no_kern,
            (specialized_kernel_key, self.default_backend_metadata),
        )

        result = ComputeCodegenUnboxedKernels(selector, use_aten_lib)(entry)
        # Concat used to prevent whitespace stripping
        expected_str = (
            """
Kernel(
    "custom_1::op_1",
    "v1/7;0,1,2,3|7;0,1,2,3|7;0,1,2,3",
    [](torch::executor::KernelRuntimeContext & context, EValue** stack) {
        """
            + """

        internal::EventTracerProfileScope event_tracer_scope(context.internal_event_tracer(), "native_call_op_1");
        EXECUTORCH_SCOPE_PROF("native_call_op_1");
        bool result_ = at::native::default_kernel(context, );
        internal::event_tracer_log_evalue(context.internal_event_tracer(), *stack[0]);

        *stack[0] = EValue(result_);
    }
),
"""
        )

        self.assertEqual(expected_str, result)

    def test_codegen_unboxed_specialized_not_matching(self) -> None:
        specialized_kernel_key = ETKernelKey.gen_from_yaml(
            {"self": ("T0", "D0"), "other": ("T0", "D0"), "out": ("T0", "D0")},
            {"T0": ["Double"]},
            {"D0": [0, 1, 2, 3]},
        )
        selector = SelectiveBuilder.from_yaml_dict(
            {
                "include_all_operators": True,
                "et_kernel_metadata": {
                    "custom_1::op_1": ["v1/8;0,1,2,3|7;0,1,2,3|7;0,1,2,3"]
                },
            }
        )
        use_aten_lib = False
        entry = (
            self.native_function_no_kern,
            (specialized_kernel_key, self.default_backend_metadata),
        )

        self.assertRaises(
            Exception, ComputeCodegenUnboxedKernels(selector, use_aten_lib), entry
        )

    def test_codegen_unboxed_specialized_missing_root_op(self) -> None:
        specialized_kernel_key = ETKernelKey.gen_from_yaml(
            {"self": ("T0", "D0"), "other": ("T0", "D0"), "out": ("T0", "D0")},
            {"T0": ["Double"]},
            {"D0": [0, 1, 2, 3]},
        )
        selector = SelectiveBuilder.from_yaml_dict(
            {
                "et_kernel_metadata": {
                    "custom_1::op_1": ["v1/7;0,1,2,3|7;0,1,2,3|7;0,1,2,3"]
                }
            }
        )
        use_aten_lib = False
        entry = (
            self.native_function_no_kern,
            (specialized_kernel_key, self.default_backend_metadata),
        )

        result = ComputeCodegenUnboxedKernels(selector, use_aten_lib)(entry)
        # Concat used to prevent whitespace stripping
        expected_str = """"""

        self.assertEqual(expected_str, result)

    def test_codegen_unboxed_default(self) -> None:
        """
        This test checks that if there is no specialized kernel, the default kernel is used.
        """
        selector = SelectiveBuilder.from_yaml_dict(
            {
                "include_all_operators": True,
                "et_kernel_metadata": {
                    "custom_1::op_1": ["v1/7;0,1,2,3|7;0,1,2,3|7;0,1,2,3"]
                },
            }
        )
        use_aten_lib = False
        entry = (self.native_function_no_kern, self.default_kernel_entry)

        result = ComputeCodegenUnboxedKernels(selector, use_aten_lib)(entry)
        # Concat used to prevent whitespace stripping
        expected_str = (
            """
Kernel(
    "custom_1::op_1",
    [](torch::executor::KernelRuntimeContext & context, EValue** stack) {
        """
            + """

        internal::EventTracerProfileScope event_tracer_scope(context.internal_event_tracer(), "native_call_op_1");
        EXECUTORCH_SCOPE_PROF("native_call_op_1");
        bool result_ = at::native::default_kernel(context, );
        internal::event_tracer_log_evalue(context.internal_event_tracer(), *stack[0]);

        *stack[0] = EValue(result_);
    }
),
"""
        )

        self.assertEqual(expected_str, result)

    def test_codegen_unboxed_default_kernel_key_selected(self) -> None:
        """
        This test checks that if there is no specialized kernel, the default kernel is used, when the selector only has default key.
        """
        selector = SelectiveBuilder.from_yaml_dict(
            {
                "include_all_operators": True,
                "et_kernel_metadata": {"custom_1::op_1": ["default"]},
            }
        )
        use_aten_lib = False
        entry = (self.native_function_no_kern, self.default_kernel_entry)

        result = ComputeCodegenUnboxedKernels(selector, use_aten_lib)(entry)
        # Concat used to prevent whitespace stripping
        expected_str = (
            """
Kernel(
    "custom_1::op_1",
    [](torch::executor::KernelRuntimeContext & context, EValue** stack) {
        """
            + """

        internal::EventTracerProfileScope event_tracer_scope(context.internal_event_tracer(), "native_call_op_1");
        EXECUTORCH_SCOPE_PROF("native_call_op_1");
        bool result_ = at::native::default_kernel(context, );
        internal::event_tracer_log_evalue(context.internal_event_tracer(), *stack[0]);

        *stack[0] = EValue(result_);
    }
),
"""
        )

        self.assertEqual(expected_str, result)
