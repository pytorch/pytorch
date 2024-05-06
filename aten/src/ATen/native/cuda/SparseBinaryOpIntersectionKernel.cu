#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/sparse/SparseStubs.h>
#include <ATen/native/sparse/SparseBinaryOpIntersectionCommon.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/AccumulateType.h>

namespace at::native {

namespace {

template <typename func_t>
struct CUDAKernelLauncher {
  static void launch(TensorIteratorBase& iter, const func_t& f) {
    gpu_kernel(iter, f);
  }
};

struct MulOp {
  template <typename scalar_t>
  static FUNCAPI INLINE scalar_t apply(scalar_t a, scalar_t b) {
    return a * b;
  }
};

template <>
FUNCAPI INLINE bool MulOp::apply(bool a, bool b) {
  return a && b;
}

struct RhsProjOp {
  template <typename scalar_t>
  static FUNCAPI scalar_t apply(scalar_t a, scalar_t b) {
    return b;
  }
};

struct LhsProjOp {
  template <typename scalar_t>
  static FUNCAPI scalar_t apply(scalar_t a, scalar_t b) {
    return a;
  }
};

template <int nt, int vt, typename loop_t>
C10_LAUNCH_BOUNDS_2(nt, vt)
__global__ void apply_kernel(int n, loop_t loop) {
  constexpr int nv = nt * vt;
  int idx = nv * blockIdx.x + threadIdx.x;

  #pragma unroll
  for (int i = 0; i < vt; ++i) {
    if (idx < n) {
      loop(idx);
      idx += nt;
    }
  }
}

template <int nt, int vt, typename loop_t>
void launch_kernel(int64_t n, const loop_t& loop) {
  TORCH_INTERNAL_ASSERT(0 <= n && n <= std::numeric_limits<int32_t>::max());
  if (!n) {
    return;
  }

  const dim3 block(nt);
  const dim3 grid((n + block.x * vt - 1) / (block.x * vt));
  const auto stream = at::cuda::getCurrentCUDAStream();
  apply_kernel<nt, vt, loop_t><<<grid, block, 0, stream>>>(n, loop);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename binary_op_t, typename scalar_t, typename index_t>
void binary_op_intersection_kernel(
    TensorIterator& iter,
    int64_t lhs_nnz_stride,
    int64_t rhs_nnz_stride,
    const Tensor& argsort,
    const bool accumulate_matches) {
  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      binary_op_intersection_kernel<binary_op_t, scalar_t, index_t>(
          sub_iter, lhs_nnz_stride, rhs_nnz_stride, argsort, accumulate_matches);
    }
    return;
  }

  auto* RESTRICT ptr_res_values_bytes = reinterpret_cast<char*>(iter.data_ptr(0));
  const auto* RESTRICT ptr_lhs_values_bytes = reinterpret_cast<char*>(iter.data_ptr(1));
  const auto* RESTRICT ptr_lhs_select_idx_bytes = reinterpret_cast<char*>(iter.data_ptr(2));
  const auto* RESTRICT ptr_rhs_values_bytes = reinterpret_cast<char*>(iter.data_ptr(3));
  const auto* RESTRICT ptr_rhs_select_idx_bytes = reinterpret_cast<char*>(iter.data_ptr(4));
  const auto* RESTRICT ptr_intersction_counts_bytes = reinterpret_cast<char*>(iter.data_ptr(5));
  const auto* RESTRICT ptr_argsort = argsort.const_data_ptr<index_t>();

  auto offset_calc = make_offset_calculator<6>(iter);
  auto loop = [=] FUNCAPI (int i) {
    auto offsets = offset_calc.get(i);

    auto* RESTRICT ptr_res_values = reinterpret_cast<scalar_t*>(ptr_res_values_bytes + offsets[0]);
    const auto* RESTRICT ptr_lhs_values = reinterpret_cast<const scalar_t*>(ptr_lhs_values_bytes + offsets[1]);
    const auto lhs_nnz_idx = *reinterpret_cast<const index_t*>(ptr_lhs_select_idx_bytes + offsets[2]);
    const auto* RESTRICT ptr_rhs_values = reinterpret_cast<const scalar_t*>(ptr_rhs_values_bytes + offsets[3]);
    const auto rhs_nnz_idx = *reinterpret_cast<const index_t*>(ptr_rhs_select_idx_bytes + offsets[4]);
    const auto count = *reinterpret_cast<const int64_t*>(ptr_intersction_counts_bytes + offsets[5]);

    const auto* RESTRICT ptr_lhs_begin = ptr_lhs_values + lhs_nnz_idx * lhs_nnz_stride;
    const auto* RESTRICT ptr_rhs_sorted_nnz_idx = ptr_argsort + rhs_nnz_idx;

    using accscalar_t = at::acc_type<scalar_t, /*is_gpu=*/true>;
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
  };

  launch_kernel<num_threads(), thread_work_size()>(iter.numel(), loop);
}


template <typename binary_op_t>
struct CUDAValueSelectionIntersectionKernel {
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

    // If res_values is empty, we can return it right away.
    // Otherwise floating point issues with OffsetCalculator.
    if (!res_values.numel()) {
      return res_values;
    }

    const auto lhs_nnz_stride = lhs_values.stride(0);
    const auto rhs_nnz_stride = rhs_values.stride(0);

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
        ScalarType::Bool, ScalarType::Half, ScalarType::BFloat16, at::ScalarType::ComplexHalf, res_values.scalar_type(),
        "binary_op_intersection_cuda", [&] {
          // COO indices are only 64-bit for now.
          using index_t = int64_t;
          binary_op_intersection_kernel<binary_op_t, scalar_t, index_t>(
              iter, lhs_nnz_stride, rhs_nnz_stride, argsort, accumulate_matches);
        });

    return res_values;
  }
};

using OptTensor = c10::optional<Tensor>;

void mul_sparse_sparse_out_cuda_kernel(
    Tensor& result,
    const Tensor& x,
    const Tensor& y) {
  using CUDAValueSelectionMulKernel = CUDAValueSelectionIntersectionKernel<MulOp>;
  _sparse_binary_op_intersection_kernel_out<CUDAKernelLauncher, CUDAValueSelectionMulKernel>(
      result, x, y
  );
}

void sparse_mask_intersection_out_cuda_kernel(
    Tensor& result,
    const Tensor& x,
    const Tensor& y,
    const OptTensor& x_hash_opt = c10::nullopt) {
  using CUDAValueRhsProjKernel = CUDAValueSelectionIntersectionKernel<RhsProjOp>;
  _sparse_binary_op_intersection_kernel_out<CUDAKernelLauncher, CUDAValueRhsProjKernel>(
      result, x, y, x_hash_opt
  );
}

void sparse_mask_projection_out_cuda_kernel(
    Tensor& result,
    const Tensor& x,
    const Tensor& y,
    const OptTensor& x_hash_opt,
    bool accumulate_matches) {
  using CUDAValueLhsProjKernel = CUDAValueSelectionIntersectionKernel<LhsProjOp>;
  _sparse_binary_op_intersection_kernel_out<CUDAKernelLauncher, CUDAValueLhsProjKernel>(
      result, x, y, x_hash_opt, c10::nullopt, accumulate_matches
  );
}

}

REGISTER_CUDA_DISPATCH(mul_sparse_sparse_out_stub, &mul_sparse_sparse_out_cuda_kernel);
REGISTER_CUDA_DISPATCH(sparse_mask_intersection_out_stub, &sparse_mask_intersection_out_cuda_kernel);
REGISTER_CUDA_DISPATCH(sparse_mask_projection_out_stub, &sparse_mask_projection_out_cuda_kernel);

} // namespace at::native
