#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/sparse/SparseStubs.h>
#include <ATen/native/sparse/SparseBinaryOpIntersectionCommon.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {

namespace {

template <typename func_t>
struct CPUKernelLauncher {
  static void launch(TensorIteratorBase& iter, const func_t& f) {
    cpu_kernel(iter, f);
  }
};

struct MulOp {
  template <typename scalar_t>
  static scalar_t apply(scalar_t a, scalar_t b) {
    return a * b;
  }
};

template <>
bool MulOp::apply(bool a, bool b) {
  return a && b;
}

template <typename binary_op_t>
struct CPUValueSelectionIntersectionKernel {
  static Tensor apply(
      const Tensor& lhs_values,
      const Tensor& lhs_select_idx,
      const Tensor& rhs_values,
      const Tensor& rhs_select_idx) {
    auto iter = make_value_selection_intersection_iter(
        lhs_values,
        lhs_select_idx,
        rhs_values,
        rhs_select_idx);
    auto res_values = iter.tensor(0);

    auto lhs_nnz_stride = lhs_values.stride(0);
    auto rhs_nnz_stride = rhs_values.stride(0);

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        ScalarType::Bool, ScalarType::Half, ScalarType::BFloat16, res_values.scalar_type(),
        "binary_op_intersection_cpu", [&] {
          AT_DISPATCH_INDEX_TYPES(lhs_select_idx.scalar_type(),
              "binary_op_intersection_cpu", [&] {
                auto loop = [&](char** data, const int64_t* strides, int64_t n) {
                  auto* ptr_res_values_bytes = data[0];
                  const auto* ptr_lhs_values_bytes = data[1];
                  const auto* ptr_lhs_select_idx_bytes = data[2];
                  const auto* ptr_rhs_values_bytes = data[3];
                  const auto* ptr_rhs_select_idx_bytes = data[4];

                  for (int64_t i = 0; i < n; ++i) {
                    // Exctract data
                    auto* RESTRICT ptr_res_values = reinterpret_cast<scalar_t*>(ptr_res_values_bytes);
                    const auto* ptr_lhs_values = reinterpret_cast<const scalar_t*>(ptr_lhs_values_bytes);
                    const auto lhs_nnz_idx = *reinterpret_cast<const index_t*>(ptr_lhs_select_idx_bytes);
                    const auto* ptr_rhs_values = reinterpret_cast<const scalar_t*>(ptr_rhs_values_bytes);
                    const auto rhs_nnz_idx = *reinterpret_cast<const index_t*>(ptr_rhs_select_idx_bytes);

                    // Apply op
                    *ptr_res_values = binary_op_t::apply(
                        *(ptr_lhs_values + lhs_nnz_idx * lhs_nnz_stride),
                        *(ptr_rhs_values + rhs_nnz_idx * rhs_nnz_stride));

                    // Advance
                    ptr_res_values_bytes += strides[0];
                    ptr_lhs_values_bytes += strides[1];
                    ptr_lhs_select_idx_bytes += strides[2];
                    ptr_rhs_values_bytes += strides[3];
                    ptr_rhs_select_idx_bytes += strides[4];
                  }
                };
                iter.for_each(loop, at::internal::GRAIN_SIZE);
              });
        });

    return res_values;
  }
};

void mul_sparse_sparse_out_cpu_kernel(
    Tensor& result,
    const Tensor& x,
    const Tensor& y) {
  using CPUValueSelectionMulKernel = CPUValueSelectionIntersectionKernel<MulOp>;
  _sparse_binary_op_intersection_kernel_out<CPUKernelLauncher, CPUValueSelectionMulKernel>(
      result, x, y
  );
}

}

REGISTER_ARCH_DISPATCH(mul_sparse_sparse_out_stub, DEFAULT, &mul_sparse_sparse_out_cpu_kernel);
REGISTER_AVX512_DISPATCH(mul_sparse_sparse_out_stub, &mul_sparse_sparse_out_cpu_kernel);
REGISTER_AVX2_DISPATCH(mul_sparse_sparse_out_stub, &mul_sparse_sparse_out_cpu_kernel);
REGISTER_VSX_DISPATCH(mul_sparse_sparse_out_stub, &mul_sparse_sparse_out_cpu_kernel);
REGISTER_ZVECTOR_DISPATCH(mul_sparse_sparse_out_stub, &mul_sparse_sparse_out_cpu_kernel);

}}
