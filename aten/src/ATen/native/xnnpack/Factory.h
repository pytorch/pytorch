#pragma once

#include <ATen/native/xnnpack/Common.h>

#ifdef USE_XNNPACK

namespace at {
namespace native {
namespace xnnpack {
namespace internal {

bool can_avoid_reallocation(
    const Tensor& input,
    c10::MemoryFormat memory_format);

Tensor allocate_padded_contiguous_if_needed(
    const Tensor& input,
    c10::MemoryFormat memory_format);

// TODO: Remove this function when at::native::empty() is modified to accept a
// custom memory allocator.

at::Tensor empty_with_tail_padding(
    IntArrayRef size,
    const caffe2::TypeMeta dtype,
    c10::MemoryFormat memory_format,
    DimnameList maybe_names);

} // namespace internal
} // namespace xnnpack
} // namespace native
} // namespace at

#endif /* USE_XNNPACK */
