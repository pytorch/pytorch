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

REGISTER_ALL_CPU_DISPATCH(flatten_indices_stub, &flatten_indices_cpu_kernel)

} // namespace at::native
