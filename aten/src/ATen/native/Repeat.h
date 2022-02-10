#pragma once

#include <ATen/ATen.h>

namespace at {
namespace native {

template <
    typename index_t,
    void compute(index_t*, int64_t*, index_t*, int64_t, int64_t)>
static inline Tensor repeat_interleave_common(
    const Tensor& repeats,
    c10::optional<int64_t> output_size) {
  TORCH_CHECK(
      repeats.dim() == 1, "repeat_interleave only accept 1D vector as repeat");
  TORCH_CHECK(
      repeats.scalar_type() == at::kLong || repeats.scalar_type() == at::kInt,
      "repeats has to be Long or Int tensor");
  if (repeats.size(0) == 0) {
    return at::empty_like(repeats, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  Tensor repeats_ = repeats.contiguous();
  Tensor cumsum = repeats.cumsum(0);
  int64_t total;
  if (output_size.has_value()) {
    total = output_size.value();
  } else {
    total = cumsum[-1].item<int64_t>();
    TORCH_CHECK(
        (repeats >= 0).all().item<uint8_t>(), "repeats can not be negative");
  }

  Tensor result = at::empty({total}, repeats.options());
  index_t* repeat_ptr = repeats_.data_ptr<index_t>();
  int64_t* cumsum_ptr = cumsum.data_ptr<int64_t>();
  index_t* result_ptr = result.data_ptr<index_t>();
  compute(repeat_ptr, cumsum_ptr, result_ptr, repeats.size(0), total);
  return result;
}

} // namespace native
} // namespace at
