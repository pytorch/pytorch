#pragma once
#include <ATen/core/TensorBase.h>
#include <optional>

namespace c10d::cuda::detail {

at::Tensor async_input_mm_out(
    at::Tensor a,
    at::Tensor b,
    at::Tensor a_chunk_signals,
    int64_t begin_chunk,
    at::Tensor out);

at::Tensor async_input_mm(
    at::Tensor a,
    at::Tensor b,
    at::Tensor a_chunk_signals,
    int64_t begin_chunk);

} // namespace c10d::cuda::detail
