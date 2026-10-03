//  Copyright © 2022 Apple Inc.

#pragma once
#include <ATen/core/Tensor.h>

namespace at::native::mps {

at::Tensor& mps_copy_(
    at::Tensor& dst,
    const at::Tensor& src,
    bool non_blocking);
void copy_blit_mps(void* dst, const void* src, size_t size);

// General MPS->MPS copy: handles arbitrary strides, a dtype cast and conj/neg
// bit flips on either operand in a single dispatch. `dst` is written in place,
// so it must already have the sizes and bits the caller wants.
void copy_cast_kernel_mps(at::Tensor& dst, const at::Tensor& src);

// Vectorized same-dtype scatter of a CONTIGUOUS `input` into regularly-strided
// blocks of `output`; see StridedBlockParams (kernels/Copy.h) for the byte
// geometry. Backs the contiguous cat fast path.
void inner_contiguous_scatter_mps(
    const at::Tensor& input,
    const at::Tensor& output,
    uint64_t slice_bytes,
    uint64_t out_stride_bytes,
    uint64_t off_bytes);

} // namespace at::native::mps
