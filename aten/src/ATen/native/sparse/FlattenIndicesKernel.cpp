#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/sparse/SparseStubs.h>
#include <ATen/native/sparse/FlattenIndicesCommon.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/AccumulateType.h>

namespace at::native {

namespace {

template <typename func_t>
struct CPUKernelLauncher {
  static void launch(TensorIteratorBase& iter, const func_t& f) {
    cpu_kernel(iter, f);
  }
};

Tensor flatten_indices_cpu_kernel(const Tensor& indices, IntArrayRef size) {
  return _flatten_indices<CPUKernelLauncher>(indices, size);
}

}

REGISTER_ARCH_DISPATCH(flatten_indices_stub, DEFAULT, &flatten_indices_cpu_kernel)
REGISTER_AVX512_DISPATCH(flatten_indices_stub, &flatten_indices_cpu_kernel)
REGISTER_AVX2_DISPATCH(flatten_indices_stub, &flatten_indices_cpu_kernel)
REGISTER_VSX_DISPATCH(flatten_indices_stub, &flatten_indices_cpu_kernel)
REGISTER_ZVECTOR_DISPATCH(flatten_indices_stub, &flatten_indices_cpu_kernel)
REGISTER_SVE256_DISPATCH(flatten_indices_stub, &flatten_indices_cpu_kernel)

} // namespace at::native
