#pragma once
#include <ATen/core/TensorBase.h>
#include <optional>

namespace c10d::symmetric_memory::cuda {

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

} // namespace c10d::symmetric_memory::cuda
