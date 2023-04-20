#include <ATen/native/sparse/ValidateCompressedIndicesCommon.h>
#include <ATen/native/cpu/Loops.h>

namespace at::native {

namespace {

template <typename func_t>
struct CPUKernel {
  static void launch(TensorIteratorBase& iter, const func_t& f) {
    cpu_kernel(iter, f);
  }
};

template <typename func_t>
struct EmptyKernel {
  static void launch(TensorIteratorBase& iter, const func_t& f) {
  }
};

template <typename func_t, typename vec_func_t>
struct CPUVecKernel {
  static void launch(TensorIteratorBase& iter, const func_t& f, const vec_func_t& vec_f) {
    cpu_kernel_vec(iter, f, vec_f);
  }
};

}

void _validate_compressed_sparse_indices_cpu(
    const bool is_crow,
    const Tensor& cidx,
    const Tensor& idx,
    const int64_t cdim,
    const int64_t dim,
    const int64_t nnz) {
  // Call into
  // compressed_index_invariance_checks_kernel<EmptyKernel, CPUVecKernel, Vectorized>
  // to enable vectorized checks once all the conditions for that are met,
  // see ATen/native/sparse/CompressedIndexChecksCommon.h for more details.
  validate_compressed_sparse_indices_kernel<CPUKernel>(
      is_crow, cidx, idx, cdim, dim, nnz);
}

} //namespace at::native
