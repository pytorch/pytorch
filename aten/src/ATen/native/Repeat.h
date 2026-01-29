#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/TensorOperators.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#endif

#include <limits>

namespace at::native {

template <
    typename index_t,
    void compute(const index_t*, const int64_t*, index_t*, int64_t, int64_t)>
static inline Tensor repeat_interleave_common(
    const Tensor& repeats,
    std::optional<int64_t> output_size) {
  TORCH_CHECK(
      repeats.dim() == 1, "repeat_interleave only accept 1D vector as repeat");
  TORCH_CHECK(
      repeats.scalar_type() == at::kLong || repeats.scalar_type() == at::kInt,
      "repeats has to be Long or Int tensor");
  if (repeats.size(0) == 0) {
    return at::empty_like(repeats, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }

  // pre check for negative values and potential overflow before computing cumsum
  auto min_max = repeats.aminmax();
  int64_t min_repeat = std::get<0>(min_max).item<int64_t>();
  int64_t max_repeat = std::get<1>(min_max).item<int64_t>();

  TORCH_CHECK(min_repeat >= 0, "repeats can not be negative");

  // if max_repeat * count could overflow the dtype, reject before overflowing in cumsum
  int64_t count = repeats.size(0);
  int64_t dtype_max = (repeats.scalar_type() == at::kInt)
      ? static_cast<int64_t>(std::numeric_limits<int32_t>::max())
      : std::numeric_limits<int64_t>::max();

  TORCH_CHECK(
      max_repeat <= dtype_max / count,
      "repeats values are too large. The sum of repeats would overflow. "
      "Max repeat value: ", max_repeat, ", count: ", count);

  Tensor repeats_ = repeats.contiguous();
  Tensor cumsum = repeats.cumsum(0);
  int64_t total = 0;
  if (output_size.has_value()) {
    total = output_size.value();
  } else {
    total = cumsum[-1].item<int64_t>();
  }

  Tensor result = at::empty({total}, repeats.options());
  const index_t* repeat_ptr = repeats_.const_data_ptr<index_t>();
  const int64_t* cumsum_ptr = cumsum.const_data_ptr<int64_t>();
  index_t* result_ptr = result.data_ptr<index_t>();
  compute(repeat_ptr, cumsum_ptr, result_ptr, repeats.size(0), total);
  return result;
}

} // namespace at::native
