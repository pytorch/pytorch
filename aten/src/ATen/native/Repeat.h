#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/TensorOperators.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <c10/util/safe_numerics.h>
#endif

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
  Tensor repeats_ = repeats.contiguous();
  // custom cumsum with overflow detection
  Tensor cumsum = at::empty({repeats.size(0)}, repeats.options().dtype(at::kLong));
  const index_t* repeat_ptr = repeats_.const_data_ptr<index_t>();
  int64_t* cumsum_ptr = cumsum.data_ptr<int64_t>();

  uint64_t cumsum_val = 0;
  for (int64_t i = 0; i < repeats.size(0); i++) {
    TORCH_CHECK(repeat_ptr[i] >= 0, "repeats can not be negative");
    uint64_t result;
    TORCH_CHECK(
        !c10::add_overflows(cumsum_val, static_cast<uint64_t>(repeat_ptr[i]), &result) &&
        result <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
        "cumulative sum values overflowed. check the repeats tensor values");
    cumsum_val = result;
    cumsum_ptr[i] = static_cast<int64_t>(cumsum_val);
  }

  int64_t total = output_size.has_value() ? output_size.value() : static_cast<int64_t>(cumsum_val);

  Tensor result = at::empty({total}, repeats.options());
  index_t* result_ptr = result.data_ptr<index_t>();
  compute(repeat_ptr, cumsum_ptr, result_ptr, repeats.size(0), total);
  return result;
}

} // namespace at::native
