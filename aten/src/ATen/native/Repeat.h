#pragma once

#include <ATen/ATen.h>

namespace at { namespace native {

template <void compute(int64_t *, int64_t *, int64_t *, int64_t)>
static inline Tensor repeat_interleave_common(const Tensor &repeats) {
    TORCH_CHECK(repeats.dim() == 1, "repeat_interleave only accept 1D vector as repeat");
    TORCH_CHECK(repeats.scalar_type() == at::kLong, "repeats has to be Long tensor");
    TORCH_CHECK((repeats >= 0).all().item<uint8_t>(), "repeats can not be negative");
    if (repeats.size(0) == 0) {
        return at::empty_like(repeats, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    }
    Tensor repeats_ = repeats.contiguous();
    Tensor cumsum = repeats.cumsum(0);
    int64_t total = cumsum[-1].item<int64_t>();
    Tensor result = at::empty({total}, repeats.options());
    int64_t *repeat_ptr = repeats_.data_ptr<int64_t>();
    int64_t *cumsum_ptr = cumsum.data_ptr<int64_t>();
    int64_t *result_ptr = result.data_ptr<int64_t>();
    compute(repeat_ptr, cumsum_ptr, result_ptr, repeats.size(0));
    return result;
}

}}
