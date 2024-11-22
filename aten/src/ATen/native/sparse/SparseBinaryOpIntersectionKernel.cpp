#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/sparse/SparseStubs.h>
#include <ATen/native/sparse/SparseBinaryOpIntersectionCommon.h>
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

struct RhsProjOp {
  template <typename scalar_t>
  static scalar_t apply(scalar_t a, scalar_t b) {
    return b;
  }
};

struct LhsProjOp {
  template <typename scalar_t>
  static scalar_t apply(scalar_t a, scalar_t b) {
    return a;
  }
};

template <typename binary_op_t>
struct CPUValueSelectionIntersectionKernel {
  static Tensor apply(
      const Tensor& lhs_values,
      const Tensor& lhs_select_idx,
      const Tensor& rhs_values,
      const Tensor& rhs_select_idx,
      const Tensor& intersection_counts,
      const Tensor& argsort,
      const bool accumulate_matches) {
    auto iter = make_value_selection_intersection_iter(
        lhs_values,
        lhs_select_idx,
        rhs_values,
        rhs_select_idx,
        intersection_counts);
    auto res_values = iter.tensor(0);

    auto lhs_nnz_stride = lhs_values.stride(0);
    auto rhs_nnz_stride = rhs_values.stride(0);

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
        ScalarType::Bool, ScalarType::Half, ScalarType::BFloat16, at::ScalarType::ComplexHalf,
        res_values.scalar_type(),
        "binary_op_intersection_cpu", [&] {
            // COO indices are only 64-bit for now.
            using index_t = int64_t;
            auto loop = [&](char** data, const int64_t* strides, int64_t n) {
              auto* ptr_res_values_bytes = data[0];
              const auto* ptr_lhs_values_bytes = data[1];
              const auto* ptr_lhs_select_idx_bytes = data[2];
              const auto* ptr_rhs_values_bytes = data[3];
              const auto* ptr_rhs_select_idx_bytes = data[4];
              const auto* ptr_intersection_counts_bytes = data[5];
              const auto* ptr_argsort = argsort.const_data_ptr<index_t>();

              for (int64_t i = 0; i < n; ++i) {
                // Exctract data
                auto* ptr_res_values = reinterpret_cast<scalar_t*>(ptr_res_values_bytes);
                const auto* ptr_lhs_values = reinterpret_cast<const scalar_t*>(ptr_lhs_values_bytes);
                const auto lhs_nnz_idx = *reinterpret_cast<const index_t*>(ptr_lhs_select_idx_bytes);
                const auto* ptr_rhs_values = reinterpret_cast<const scalar_t*>(ptr_rhs_values_bytes);
                const auto rhs_nnz_idx = *reinterpret_cast<const index_t*>(ptr_rhs_select_idx_bytes);
                const auto count = *reinterpret_cast<const int64_t*>(ptr_intersection_counts_bytes);

                const auto* ptr_lhs_begin = ptr_lhs_values + lhs_nnz_idx * lhs_nnz_stride;
                const auto* ptr_rhs_sorted_nnz_idx = ptr_argsort + rhs_nnz_idx;

                using accscalar_t = at::acc_type<scalar_t, /*is_gpu=*/false>;
                accscalar_t res_values = 0;
                accscalar_t lhs_values = static_cast<accscalar_t>(*ptr_lhs_begin);
                accscalar_t rhs_values;
                index_t rhs_sorted_nnz_idx;
                const auto match_count = accumulate_matches ? count : std::min<int64_t>(count, 1);
                for (int64_t c = 0; c < match_count; ++c) {
                  rhs_sorted_nnz_idx = *ptr_rhs_sorted_nnz_idx++;
                  rhs_values = static_cast<accscalar_t>(*(ptr_rhs_values + rhs_sorted_nnz_idx * rhs_nnz_stride));
                  res_values += binary_op_t::apply(lhs_values, rhs_values);
                }
                *ptr_res_values = static_cast<scalar_t>(res_values);

                // Advance
                ptr_res_values_bytes += strides[0];
                ptr_lhs_values_bytes += strides[1];
                ptr_lhs_select_idx_bytes += strides[2];
                ptr_rhs_values_bytes += strides[3];
                ptr_rhs_select_idx_bytes += strides[4];
                ptr_intersection_counts_bytes += strides[5];
              }
            };
            iter.for_each(loop, at::internal::GRAIN_SIZE);
        });

    return res_values;
  }
};

using OptTensor = std::optional<Tensor>;

void mul_sparse_sparse_out_cpu_kernel(
    Tensor& result,
    const Tensor& x,
    const Tensor& y) {
  using CPUValueSelectionMulKernel = CPUValueSelectionIntersectionKernel<MulOp>;
  _sparse_binary_op_intersection_kernel_out<CPUKernelLauncher, CPUValueSelectionMulKernel>(
      result, x, y
  );
}

void sparse_mask_intersection_out_cpu_kernel(
    Tensor& result,
    const Tensor& x,
    const Tensor& y,
    const OptTensor& x_hash_opt = std::nullopt) {
  using CPUValueRhsProjKernel = CPUValueSelectionIntersectionKernel<RhsProjOp>;
  _sparse_binary_op_intersection_kernel_out<CPUKernelLauncher, CPUValueRhsProjKernel>(
      result, x, y, x_hash_opt
  );
}

void sparse_mask_projection_out_cpu_kernel(
    Tensor& result,
    const Tensor& x,
    const Tensor& y,
    const OptTensor& x_hash_opt,
    bool accumulate_matches) {
  using CPUValueLhsProjKernel = CPUValueSelectionIntersectionKernel<LhsProjOp>;
  _sparse_binary_op_intersection_kernel_out<CPUKernelLauncher, CPUValueLhsProjKernel>(
      result, x, y, x_hash_opt, std::nullopt, accumulate_matches
  );
}

}

REGISTER_ARCH_DISPATCH(mul_sparse_sparse_out_stub, DEFAULT, &mul_sparse_sparse_out_cpu_kernel)
REGISTER_AVX512_DISPATCH(mul_sparse_sparse_out_stub, &mul_sparse_sparse_out_cpu_kernel)
REGISTER_AVX2_DISPATCH(mul_sparse_sparse_out_stub, &mul_sparse_sparse_out_cpu_kernel)
REGISTER_VSX_DISPATCH(mul_sparse_sparse_out_stub, &mul_sparse_sparse_out_cpu_kernel)
REGISTER_ZVECTOR_DISPATCH(mul_sparse_sparse_out_stub, &mul_sparse_sparse_out_cpu_kernel)
REGISTER_SVE256_DISPATCH(mul_sparse_sparse_out_stub, &mul_sparse_sparse_out_cpu_kernel)

REGISTER_ARCH_DISPATCH(sparse_mask_intersection_out_stub, DEFAULT, &sparse_mask_intersection_out_cpu_kernel)
REGISTER_AVX512_DISPATCH(sparse_mask_intersection_out_stub, &sparse_mask_intersection_out_cpu_kernel)
REGISTER_AVX2_DISPATCH(sparse_mask_intersection_out_stub, &sparse_mask_intersection_out_cpu_kernel)
REGISTER_VSX_DISPATCH(sparse_mask_intersection_out_stub, &sparse_mask_intersection_out_cpu_kernel)
REGISTER_ZVECTOR_DISPATCH(sparse_mask_intersection_out_stub, &sparse_mask_intersection_out_cpu_kernel)
REGISTER_SVE256_DISPATCH(sparse_mask_intersection_out_stub, &sparse_mask_intersection_out_cpu_kernel)

REGISTER_ARCH_DISPATCH(sparse_mask_projection_out_stub, DEFAULT, &sparse_mask_projection_out_cpu_kernel)
REGISTER_AVX512_DISPATCH(sparse_mask_projection_out_stub, &sparse_mask_projection_out_cpu_kernel)
REGISTER_AVX2_DISPATCH(sparse_mask_projection_out_stub, &sparse_mask_projection_out_cpu_kernel)
REGISTER_VSX_DISPATCH(sparse_mask_projection_out_stub, &sparse_mask_projection_out_cpu_kernel)
REGISTER_ZVECTOR_DISPATCH(sparse_mask_projection_out_stub, &sparse_mask_projection_out_cpu_kernel)
REGISTER_SVE256_DISPATCH(sparse_mask_projection_out_stub, &sparse_mask_projection_out_cpu_kernel)
}
