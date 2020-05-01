#include <ATen/native/UnfoldBackward.h>

#include <ATen/native/cuda/Loops.cuh>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/zmath.cuh>

#include <vector>

// unfold_backward, the algorithm.
//
// Consider out = in.unfold(dim, size, step), then
// out.shape[dim] == (in.shape[dim] - size) / step + 1,
// out.shape[-1] == size.
// out.dims() == in.dims() + 1
//
// unfold_backward receives grad_in and returns grad_out such that
// grad_in.shape == out.shape,
// grad_out.shape = in.shape.
//
// unfold_backward considers the following two cases:
// case1. step >= size.
// case2. step < size.
//
// case1. step >= size.
// In this case the iteration takes over grad_in and performs the following copy:
// grad_out[..., i_out_dim,...] = grad_in[..., i_in_dim,..., i_in_last_dim],
// where i_out_dim = i_in_dim * step + i_in_last_dim.
//
// case2. step < size.
// In this case the iteration takes over grad_out,
// where grad_out[...,i_out_dim,...] accumulates all values
// grad_in[...,i_in_dim,...,i_in_last_dim], where
// i_in_dim is in [left_idx_fold, right_idx_fold],
// i_in_last_dim = i_out_dim - i_in_dim * step,
// left_idx_fold = (i_out_dim - size) / step 
//  if i_out_dim in [left_idx_fold * step, left_idx_fold * step + size)
//  else (i_out_dim - size) / step + 1,
// right_idx_fold = i_out_dim / step.
//
// Simply put, given i_out_dim, we find which folds of grad_in
// intersect with i_out_dim, these are precisely [left_idx_fold, right_idx_fold],
// and then the corresponding value of grad_in[...,i_in_dim,...,i_in_last_dim]
// gets added up to grad_out[...,i_out_dim,...].

namespace at { namespace native {

namespace {

template <int n_threads, int n_elems_per_thread, typename func_t>
C10_LAUNCH_BOUNDS_2(n_threads, n_elems_per_thread)
__global__ void _unfold_backward_elementwise_kernel(int total_n_elems, func_t f) {
  constexpr int total_work_block = n_threads * n_elems_per_thread;
  int idx = total_work_block * blockIdx.x + threadIdx.x;

  #pragma unroll
  for (int i = 0; i < n_elems_per_thread; ++i) {
    if (idx < total_n_elems) {
      f(idx);
      idx += n_threads;
    }
  }
}

template <int n_threads, int n_elems_per_thread, typename func_t>
static void _launch_unfold_backward_kernel(int total_n_elems, func_t f) {
  TORCH_INTERNAL_ASSERT(
    total_n_elems >= 0 && total_n_elems <= std::numeric_limits<int32_t>::max()
  );

  if (total_n_elems == 0) {
    return;
  }

  dim3 block(n_threads);
  constexpr int total_work_block = n_threads * n_elems_per_thread;
  dim3 grid((total_n_elems + total_work_block - 1) / total_work_block);
  
  auto stream = at::cuda::getCurrentCUDAStream();
  _unfold_backward_elementwise_kernel<n_threads, n_elems_per_thread, func_t>
    <<<grid, block, 0, stream>>>(total_n_elems, f);
  AT_CUDA_CHECK(cudaGetLastError());
}

template <typename scalar_t>
void _unfold_backward_internal_kernel(
  TensorIterator& iter,
  int64_t size,
  int64_t step,
  int64_t grad_in_dim_stride,
  int64_t grad_in_last_dim_stride,
  int64_t grad_in_dim_size,
  int64_t grad_out_dim_stride,
  bool is_step_ge_size
) {
  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      _unfold_backward_internal_kernel<scalar_t>(
        sub_iter,
        size,
        step,
        grad_in_dim_stride,
        grad_in_last_dim_stride,
        grad_in_dim_size,
        grad_out_dim_stride,
        is_step_ge_size
      );
    }
    return;
  }

  char* __restrict__ grad_out_ptr = reinterpret_cast<char*>(iter.data_ptr(0));
  char* __restrict__ grad_in_ptr = reinterpret_cast<char*>(iter.data_ptr(1));
  char* __restrict__ idx_dim_ptr = reinterpret_cast<char*>(iter.data_ptr(2));

  if (is_step_ge_size) {
    char* __restrict__ idx_last_dim_ptr = reinterpret_cast<char*>(iter.data_ptr(3));

    auto offset_calc = make_offset_calculator<4>(iter);

    // this loop simply copies the data
    // from proper places in grad_out to grad_in
    auto loop = [=]C10_DEVICE(int i) {
      auto offsets = offset_calc.get(i);

      auto* __restrict__ grad_out_data = reinterpret_cast<scalar_t*>(grad_out_ptr + offsets[0]);
      auto* __restrict__ grad_in_data = reinterpret_cast<scalar_t*>(grad_in_ptr + offsets[1]);

      auto idx_dim = *reinterpret_cast<int64_t*>(idx_dim_ptr + offsets[2]);
      auto idx_last_dim = *reinterpret_cast<int64_t*>(idx_last_dim_ptr + offsets[3]);

      auto grad_out_idx_dim = idx_dim * step + idx_last_dim;
      grad_out_data[grad_out_idx_dim * grad_out_dim_stride] = *grad_in_data;
    };

    _launch_unfold_backward_kernel<num_threads, thread_work_size>(iter.numel(), loop);
  }
  else {
    auto offset_calc = make_offset_calculator<3>(iter);

    // The algorithm is: for each index in grad_out find
    // the elements contributing to it and sum them up.
    // Note: the algorithm does not require any synchronization.
    auto loop = [=]C10_DEVICE(int i) {
      auto offsets = offset_calc.get(i);

      auto* __restrict__ grad_out_data = reinterpret_cast<scalar_t*>(grad_out_ptr + offsets[0]);
      auto* __restrict__ grad_in_data = reinterpret_cast<scalar_t*>(grad_in_ptr + offsets[1]);

      auto idx_dim = *reinterpret_cast<int64_t*>(idx_dim_ptr + offsets[2]);

      // left_fold potentially intersecting with idx_dim
      // is either (idx_dim - size) / step or the next integer.
      int64_t left_fold_idx = (idx_dim > size) ? (idx_dim - size) / step : 0;
      if (!(left_fold_idx * step <= idx_dim && idx_dim < left_fold_idx * step + size)) {
        ++left_fold_idx;
      }

      auto right_fold_idx = idx_dim / step;
      right_fold_idx = (right_fold_idx >= grad_in_dim_size) ?
        (grad_in_dim_size - 1) : right_fold_idx;

      for (auto fold_idx = left_fold_idx; fold_idx <= right_fold_idx; ++fold_idx) {
        auto idx_last_dim = idx_dim - fold_idx * step;
        *grad_out_data += grad_in_data[fold_idx * grad_in_dim_stride
                                    + idx_last_dim * grad_in_last_dim_stride];
      }

    };

    _launch_unfold_backward_kernel<num_threads, thread_work_size>(iter.numel(), loop);
  }
}

void unfold_backward_cuda_kernel(
  Tensor& grad_out,
  const Tensor& grad_in,
  int64_t dim,
  int64_t size,
  int64_t step
) {
  dim = maybe_wrap_dim(dim, grad_out.dim());
  // last dim stores the folds
  auto last_dim = maybe_wrap_dim(-1, grad_in.dim());

  auto grad_in_dim_stride = ensure_nonempty_stride(grad_in, dim);
  auto grad_in_last_dim_stride = ensure_nonempty_stride(grad_in, last_dim);
  auto grad_in_dim_size = ensure_nonempty_size(grad_in, dim);

  auto grad_out_dim_stride = ensure_nonempty_stride(grad_out, dim);

  auto is_step_ge_size = (step >= size);

  TensorIterator iter;
  if (is_step_ge_size) {
    iter = _make_unfold_backward_iter_over_grad_in(
      grad_out, grad_in, dim, size, step
    );
  }
  else {
    iter = _make_unfold_backward_iter_over_grad_out(
      grad_out, grad_in, dim, size, step
    );
  }

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
    at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16,
    iter.dtype(),
    "unfold_backward_cuda", [&] {
      using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
      _unfold_backward_internal_kernel<thrust_t>(
        iter,
        size,
        step,
        grad_in_dim_stride,
        grad_in_last_dim_stride,
        grad_in_dim_size,
        grad_out_dim_stride,
        is_step_ge_size
      );
    }
  );
}

}

REGISTER_DISPATCH(unfold_backward_stub, &unfold_backward_cuda_kernel);

}} // namespace at::native
