#pragma once

#include <ATen/core/Tensor.h>

namespace at::native::mobile {

Tensor allocate_padded_contiguous_if_needed(
    const Tensor& input,
    c10::MemoryFormat memory_format);

// TODO: Remove this function when at::native::empty() is modified to accept a
// custom memory allocator.

at::Tensor empty_with_tail_padding(
    IntArrayRef size,
    const caffe2::TypeMeta dtype,
    c10::MemoryFormat memory_format,
    std::optional<DimnameList> maybe_names);

} // namespace at
