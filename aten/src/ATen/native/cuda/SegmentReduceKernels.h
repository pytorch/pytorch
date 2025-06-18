#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/native/SegmentReduce.h>
#include <c10/util/Optional.h>

namespace at::native {

// Forward kernels and functions
template <typename scalar_t, typename index_t>
__global__ void post_sum_div_kernel(
    scalar_t* output_data,
    const index_t* lengths_data,
    const int64_t segment_count,
    bool is_initial_set,
    scalar_t initial);

template <typename scalar_t, typename index_t>
__global__ void segment_reduce_forward_kernel(
    ReductionType reduction,
    scalar_t* output_data,
    const scalar_t* values_data,
    const index_t* lengths_data,
    const index_t* lengths_cumsum_data,
    const int64_t segment_count,
    const int64_t lengths_stride_axis,
    bool is_initial_set,
    scalar_t initial_value,
    const int64_t outer_offset,
    const int64_t inner_offset,
    const int64_t data_stride_axis,
    const int64_t data_size_axis,
    const int64_t output_stride_axis,
    const int64_t output_size_axis,
    const int64_t lengths_cumsum_stride_axis);

Tensor _segment_reduce_lengths_offsets_cuda_kernel(
    ReductionType reduction,
    const Tensor& data,
    const Tensor& lengths_or_offsets,
    int64_t axis,
    const std::optional<Scalar>& initial,
    bool is_offsets_like);

Tensor _segment_reduce_lengths_cuda_kernel(
    ReductionType reduction,
    const Tensor& data,
    const Tensor& lengths,
    int64_t axis,
    const std::optional<Scalar>& initial);

Tensor _segment_reduce_offsets_cuda_kernel(
    ReductionType reduction,
    const Tensor& data,
    const Tensor& offsets,
    int64_t axis,
    const std::optional<Scalar>& initial);

// Backward kernels and functions
template <typename scalar_t, typename index_t>
__global__ void segment_reduce_backward_kernel(
    ReductionType reduction,
    scalar_t* grad_input_data,
    const scalar_t* grad_data,
    const scalar_t* output_data,
    const scalar_t* values_data,
    const index_t* lengths_data,
    const index_t* lengths_cumsum_data,
    const int64_t segment_count,
    const int64_t lengths_stride_axis,
    scalar_t initial_prod_value,
    const int64_t outer_offset,
    const int64_t inner_offset,
    const int64_t data_stride_axis,
    const int64_t data_size_axis,
    const int64_t output_stride_axis,
    const int64_t output_size_axis,
    const int64_t lengths_cumsum_stride_axis);

Tensor _segment_reduce_lengths_offsets_backward_cuda_kernel(
    const Tensor& grad_contig,
    const Tensor& output_contig,
    const Tensor& data_contig,
    ReductionType reduction,
    const Tensor& lengths_or_offsets_contig,
    int64_t axis,
    const std::optional<Scalar>& initial,
    bool is_offsets_like);

Tensor _segment_reduce_lengths_backward_cuda_kernel(
    const Tensor& grad_contig,
    const Tensor& output_contig,
    const Tensor& data_contig,
    ReductionType reduction,
    const Tensor& lengths_contig,
    int64_t axis,
    const std::optional<Scalar>& initial);

Tensor _segment_reduce_offsets_backward_cuda_kernel(
    const Tensor& grad_contig,
    const Tensor& output_contig,
    const Tensor& data_contig,
    ReductionType reduction,
    const Tensor& offsets_contig,
    int64_t axis,
    const std::optional<Scalar>& initial);

} // namespace at::native 