import os
import tempfile
import unittest
from typing import Dict

import yaml
from torchgen.gen import LineLoader

from torchgen.gen_executorch import gen_functions_declarations, translate_native_yaml
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
        with open(out_yaml_path, "r") as out_file:
            es = yaml.load(out_file, Loader=LineLoader)
        self.assertTrue(all("func" in e for e in es))
        self.assertTrue(all(e.get("variants") == "function" for e in es))

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

        backend_indices: Dict[DispatchKey, Dict[OperatorName, BackendMetadata]] = {
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

    def test_operators_with_different_namespaces_are_grouped_correctly(self) -> None:
        declarations = gen_functions_declarations(
            native_functions=[
                self.custom_1_native_function,
                self.custom_2_native_function,
            ],
            static_dispatch_idx=self.static_dispatch_idx,
            selector=SelectiveBuilder.get_nop_selector(),
            use_aten_lib=False,
        )
        self.assertTrue(
            """
namespace custom_1 {

// custom_1::op_1() -> bool
TORCH_API inline bool op_1(torch::executor::RuntimeContext & context) {
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
TORCH_API inline bool op_2(torch::executor::RuntimeContext & context) {
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
            static_dispatch_idx=self.static_dispatch_idx,
            selector=SelectiveBuilder.get_nop_selector(),
            use_aten_lib=True,
        )
        print(declarations)
        self.assertTrue(
            """
namespace custom_1 {

// custom_1::op_1() -> bool
TORCH_API inline bool op_1(torch::executor::RuntimeContext & context) {
    return at::op_1();
}

} // namespace custom_1
        """
            in declarations
        )
