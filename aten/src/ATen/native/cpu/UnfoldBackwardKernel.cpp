#include <ATen/Parallel.h>
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/native/UnfoldBackward.h>
#include <ATen/native/cpu/Loops.h>

#if (defined(_WIN32) || defined(_WIN64))
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif

namespace at {
namespace native {

namespace {

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

  auto loop = [&](char** data, const int64_t* strides, int64_t nelems) {
    auto* RESTRICT grad_out_ptr = data[0];
    auto* RESTRICT grad_in_ptr = data[1];
    auto* RESTRICT idx_dim_ptr = data[2];

    if (is_step_ge_size) {
      auto* RESTRICT idx_last_dim_ptr = data[3];

      for (int64_t elem = 0; elem < nelems; ++elem) {
        auto* RESTRICT grad_out_data = reinterpret_cast<scalar_t*>(grad_out_ptr);
        auto* RESTRICT grad_in_data = reinterpret_cast<scalar_t*>(grad_in_ptr);

        auto idx_dim = *reinterpret_cast<int64_t*>(idx_dim_ptr);
        auto idx_last_dim = *reinterpret_cast<int64_t*>(idx_last_dim_ptr);

        auto grad_out_idx_dim = idx_dim * step + idx_last_dim;
        grad_out_data[grad_out_idx_dim * grad_out_dim_stride] = *grad_in_data;

        grad_out_ptr += strides[0];
        grad_in_ptr += strides[1];
        idx_dim_ptr += strides[2];
        idx_last_dim_ptr += strides[3];
      }
    }
    else {
      for (int64_t elem = 0; elem < nelems; ++elem) {
        auto* RESTRICT grad_out_data = reinterpret_cast<scalar_t*>(grad_out_ptr);
        auto* RESTRICT grad_in_data = reinterpret_cast<scalar_t*>(grad_in_ptr);

        auto idx_dim = *reinterpret_cast<int64_t*>(idx_dim_ptr);

        // left_fold potentially intersecting with idx_dim
        // is either (idx_dim - size) / step or the next integer.
        int64_t left_fold_idx = (idx_dim > size) ? (idx_dim - size) / step : 0;
        if (!(left_fold_idx * step <= idx_dim && idx_dim < left_fold_idx * step + size)) {
          ++left_fold_idx;
        }

        auto right_fold_idx = idx_dim / step;
        right_fold_idx = (right_fold_idx >= grad_in_dim_size)
          ? (grad_in_dim_size - 1) : right_fold_idx;

        for (auto fold_idx = left_fold_idx; fold_idx <= right_fold_idx; ++fold_idx) {
          auto idx_last_dim = idx_dim - fold_idx * step;
          *grad_out_data += grad_in_data[fold_idx * grad_in_dim_stride
                                      + idx_last_dim * grad_in_last_dim_stride];
        }

        grad_out_ptr += strides[0];
        grad_in_ptr += strides[1];
        idx_dim_ptr += strides[2];
      }
    }
  };

  iter.for_each(loop);
}

void unfold_backward_cpu_kernel(
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
    "unfold_backward_cpu", [&] {
      _unfold_backward_internal_kernel<scalar_t>(
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

REGISTER_DISPATCH(unfold_backward_stub, &unfold_backward_cpu_kernel);

}} // namespace at::native
