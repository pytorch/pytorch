#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/NonEmptyUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/arange.h>
#endif

namespace at::native {

using unfold_backward_fn = void (*)(
  Tensor& grad_in,
  const Tensor& grad,
  int64_t dim,
  int64_t size,
  int64_t step
);

DECLARE_DISPATCH(unfold_backward_fn, unfold_backward_stub);

namespace {

// Note on naming: it is unconventional.
// grad_in does not mean that it is a gradient wrt to input,
// grad_in/grad_out is just an input/output of unfold_backward kernel.

static C10_UNUSED TensorIterator _make_unfold_backward_iter_over_grad_out(
  Tensor& grad_out,
  const Tensor& grad_in,
  int64_t dim,
  int64_t size,
  int64_t step
) {
  dim = maybe_wrap_dim(dim, grad_out.dim());
  // last dim stores the folds

  auto grad_out_dim_size = ensure_nonempty_size(grad_out, dim);
  auto grad_in_dim_size = ensure_nonempty_size(grad_in, dim);
  // dictates the number of elements to iterate over
  // in dimension `dim`
  auto iter_dim_size = std::min(
    grad_out_dim_size,
    (grad_in_dim_size - 1) * step + size
  );

  /* prepare grad_out for TensorIterator { */
  auto grad_out_strides = ensure_nonempty_vec(grad_out.strides().vec());
  auto grad_out_sizes = ensure_nonempty_vec(grad_out.sizes().vec());
  grad_out_sizes[dim] = iter_dim_size;
  auto grad_out_restrided = grad_out.as_strided(
    grad_out_sizes, grad_out_strides
  );
  /* } */

  /* prepare grad_in for TensorIterator { */
  auto grad_in_strides = ensure_nonempty_vec(grad_in.strides().vec());
  auto grad_in_sizes = ensure_nonempty_vec(grad_in.sizes().vec());

  // set strides for dim to 0
  // and size to 1 because
  // this dimension is indexed inside the kernel
  grad_in_strides[dim] = 0;
  grad_in_sizes[dim] = 1;

  grad_in_strides.pop_back();
  grad_in_sizes.pop_back();

  auto grad_in_restrided = grad_in.squeeze(-1).as_strided(
    grad_in_sizes, grad_in_strides
  );
  /* } */

  // During the TensorIterator iteration we have to know
  // i_dim in grad_out[i_1,...,i_dim,...i_n],
  // idx_dim stores this information
  /* prepare idx_dim for TensorIterator { */
  auto idx_dim = at::arange(
    0, iter_dim_size, grad_in.options().dtype(at::kLong)
  );

  auto grad_out_dim = ensure_nonempty_dim(grad_out.dim());

  auto idx_dim_strides = std::vector<int64_t>(grad_out_dim, 0);
  auto idx_dim_sizes = std::vector<int64_t>(grad_out_dim, 1);

  idx_dim_strides[dim] = 1;
  idx_dim_sizes[dim] = iter_dim_size;

  // idx_dim size will broadcast over determined by grad_out sizes in TensorIterator
  auto idx_dim_restrided = idx_dim.as_strided(idx_dim_sizes, idx_dim_strides);
  /* } */

  auto iter = TensorIteratorConfig()
    .set_check_mem_overlap(false)
    .check_all_same_dtype(false)
    .resize_outputs(false)
    .add_owned_output(grad_out_restrided)
    .add_owned_input(grad_in_restrided)
    .add_owned_input(idx_dim_restrided)
    .build();

  return iter;
}

}

} // namespace at::native
