// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#import <stdlib.h>
#import <functional>

#import "arm_neon_support.h"
#import <dispatch/dispatch.h>

template <typename T1, typename T2, void process_data(T1*, const T2*, size_t), int P>
void parallelize(T1* dst, const T2* src, size_t items, dispatch_queue_t queue) {
  process_data(dst, src, items);
  dispatch_apply(P, queue, ^(size_t it) {
    const size_t chunk_size = it == P - 1 ? items - (P - 1) * (items / P) : items / P;
    const size_t offset     = it * chunk_size;

    process_data(dst + offset, src + offset, chunk_size);
  });
}

template <typename filter_type>
filter_type* reformatKernelImage(
    const float*                        input_data,
    int                                 kernels,
    int                                 input_kernels,
    int                                 kernel_offset,
    int                                 kernel_stride,
    int                                 channels,
    int                                 width,
    int                                 height,
    bool                                transposed,
    std::function<filter_type*(size_t)> allocator);

template <typename filter_type>
bool reformatKernelImage(
    const float* input_data,
    filter_type* output_data,
    int          kernels,
    int          input_kernels,
    int          kernel_offset,
    int          kernel_stride,
    int          channels,
    int          width,
    int          height,
    bool         transposed);

template <typename filter_type>
bool reformatKernelImage(
    const float*                     input_data,
    int                              kernels,
    int                              kernel_channels,
    int                              kernel_width,
    int                              kernel_height,
    bool                             transposed,
    std::function<filter_type*(int)> allocator);

template <typename out_buffer_type>
out_buffer_type* reformatInputImage(
    const float*                            data,
    int                                     channels,
    int                                     width,
    int                                     height,
    std::function<out_buffer_type*(size_t)> allocator);

template <typename T1, typename T2>
void memcpycvt(T1* dst, const T2* src, size_t n);
