#pragma once
#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>

namespace at::native {

void alloc_with_matching_layout(
    const Tensor& q,
    Tensor& output,
    const std::vector<int64_t>& shape) {
  TORCH_INTERNAL_ASSERT(
      shape.size() == q.sizes().size(),
      "SDPA alloc_with_matching_layout got requested shape ndim != q ndim");

  if (std::equal(q.sizes().begin(), q.sizes().end(), shape.begin())) {
    output = at::empty_like(q);
    return;
  }

  // get the "fill order," which is just an argsort on the strides
  std::vector<int> fill_order(shape.size());
  std::iota(fill_order.begin(), fill_order.end(), 0);
  const auto q_strides = q.strides();
  std::stable_sort(
      fill_order.begin(), fill_order.end(), [&q_strides](int idx1, int idx2) {
        return q_strides[idx1] < q_strides[idx2];
      });
  std::vector<int64_t> ordered_strides(shape.size());
  int64_t current_stride = 1;
  for (const int dim_idx : fill_order) {
    ordered_strides[dim_idx] = current_stride;
    current_stride *= shape[dim_idx];
  }
  output = at::empty(at::IntArrayRef(shape), q.options())
               .as_strided(
                   at::IntArrayRef(shape), at::IntArrayRef(ordered_strides), 0);
}

void permute_to_matching_layout(const Tensor& output, Tensor& grad_output) {
  const int dims = output.sizes().size();
  std::vector<int64_t> outer_to_inner(dims);
  std::iota(outer_to_inner.begin(), outer_to_inner.end(), 0);
  const auto o_strides = output.strides();
  std::stable_sort(
      outer_to_inner.begin(),
      outer_to_inner.end(),
      [&o_strides](int idx1, int idx2) {
        return o_strides[idx1] > o_strides[idx2];
      });
  std::vector<int64_t> inverse(dims);
  for (int d = 0; d < dims; d++) {
    inverse[d] = std::find(outer_to_inner.begin(), outer_to_inner.end(), d) -
        outer_to_inner.begin();
  }
  grad_output = grad_output.permute(at::IntArrayRef(outer_to_inner))
                    .contiguous()
                    .permute(at::IntArrayRef(inverse));
}

bool same_strides(const Tensor& t1, const Tensor& t2) {
  std::vector<int> t1_strides_no_ones;
  std::vector<int> t2_strides_no_ones;
  const auto t1strides = t1.strides();
  const auto t2strides = t2.strides();
  const int dim = t1strides.size();
  if (dim != (int)t2strides.size()) {
    return false;
  }
  const auto t1sizes = t1.sizes();
  const auto t2sizes = t2.sizes();

  // we are going through strides backward here, but if both are backward it's
  // comparable
  for (int i = 0; i < dim; i++) {
    if (t1sizes[i] > 1) {
      t1_strides_no_ones.push_back(t1strides[i]);
    }
    if (t2sizes[i] > 1) {
      t2_strides_no_ones.push_back(t2strides[i]);
    }
  }
  return std::equal(
      t1_strides_no_ones.begin(),
      t1_strides_no_ones.end(),
      t2_strides_no_ones.begin(),
      t2_strides_no_ones.end());
}
} // namespace at::native
