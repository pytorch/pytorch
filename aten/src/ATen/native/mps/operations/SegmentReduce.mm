//  Copyright © 2024 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/SegmentReduce.h>

#include <ATen/Dispatch.h>
#include <ATen/TensorOperators.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/mps/OperationUtils.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/cumsum.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#endif
#include <fmt/format.h>

namespace at::native {
namespace mps {

// NOTE: This is a CPU fallback implementation for segment_reduce on MPS.
// Data is copied from MPS to CPU, computed on CPU, and copied back to MPS.
// This ensures correctness but has performance overhead due to memory transfers.
// A native Metal shader implementation would provide better performance.
// TODO: Implement native Metal kernels for segment_reduce operations.

// Helper function to validate offsets for bounds safety
template <typename index_t>
static void validate_offsets(
    const index_t* offsets_data,
    int64_t offsets_size,
    int64_t data_size_axis,
    int64_t outer_idx,
    int64_t stride) {
  // Validate that offsets are monotonically non-decreasing and within bounds
  index_t prev_offset = offsets_data[outer_idx * stride];
  TORCH_CHECK(
      prev_offset >= 0 && prev_offset <= static_cast<index_t>(data_size_axis),
      "segment_reduce: offset[0]=", prev_offset,
      " is out of bounds [0, ", data_size_axis, "]");

  for (int64_t i = 1; i < offsets_size; ++i) {
    index_t curr_offset = offsets_data[outer_idx * stride + i];
    TORCH_CHECK(
        curr_offset >= prev_offset,
        "segment_reduce: offsets must be monotonically non-decreasing, but offset[", i,
        "]=", curr_offset, " < offset[", i-1, "]=", prev_offset);
    TORCH_CHECK(
        curr_offset <= static_cast<index_t>(data_size_axis),
        "segment_reduce: offset[", i, "]=", curr_offset,
        " is out of bounds [0, ", data_size_axis, "]");
    prev_offset = curr_offset;
  }
}

// Helper function to validate lengths for bounds safety
template <typename index_t>
static void validate_lengths(
    const index_t* lengths_data,
    int64_t lengths_size,
    int64_t data_size_axis,
    int64_t outer_idx,
    int64_t stride) {
  index_t total_length = 0;
  for (int64_t i = 0; i < lengths_size; ++i) {
    index_t length = lengths_data[outer_idx * stride + i];
    TORCH_CHECK(
        length >= 0,
        "segment_reduce: lengths must be non-negative, but length[", i, "]=", length);
    total_length += length;
  }
  TORCH_CHECK(
      total_length <= static_cast<index_t>(data_size_axis),
      "segment_reduce: sum of lengths (", total_length,
      ") exceeds data size along axis (", data_size_axis, ")");
}

// CPU fallback implementation for segment_reduce
// Performance note: This implementation copies data between MPS and CPU,
// which adds overhead. For performance-critical applications, consider
// using CPU tensors directly or waiting for a native Metal implementation.
template <typename scalar_t, typename index_t, bool is_offsets_like = false>
static void segment_reduce_mps_kernel_impl(
    ReductionType reduction,
    const Tensor& data,
    const index_t* lengths_or_offsets_data,
    int64_t axis,
    const std::optional<Scalar>& initial,
    Tensor& output,
    int64_t segment_count,
    int64_t lengths_stride_axis) {

  // Get dimensions
  int64_t outer_offset = 1, inner_offset = 1;
  for (int64_t d = 0; d < axis; d++)
    outer_offset *= output.size(d);
  for (int64_t d = axis + 1; d < output.dim(); d++)
    inner_offset *= output.size(d);

  int64_t lengths_size_axis = is_offsets_like ? segment_count + 1 : segment_count;
  auto data_stride_axis = data.stride(axis);
  auto data_size_axis = data.size(axis);
  auto output_stride_axis = output.stride(axis);
  auto output_size_axis = output.size(axis);

  // Move tensors to CPU for computation (CPU fallback)
  auto data_cpu = data.to(kCPU);
  auto output_cpu = at::empty(output.sizes(), output.options().device(kCPU));

  auto* output_data = output_cpu.data_ptr<scalar_t>();
  const auto* values_data = data_cpu.const_data_ptr<scalar_t>();

  for (const auto outer_idx : c10::irange(outer_offset)) {
    // Validate bounds before processing
    if constexpr (is_offsets_like) {
      validate_offsets(lengths_or_offsets_data, lengths_size_axis, data_size_axis,
                       outer_idx, lengths_stride_axis);
    } else {
      validate_lengths(lengths_or_offsets_data, lengths_size_axis, data_size_axis,
                       outer_idx, lengths_stride_axis);
    }

    int64_t segment_start, segment_length;
    int64_t segment_end = is_offsets_like
        ? lengths_or_offsets_data[outer_idx * lengths_stride_axis * lengths_size_axis]
        : 0;

    for (const auto dim_idx : c10::irange(segment_count)) {
      segment_start = segment_end;
      auto lengths_idx = outer_idx * lengths_stride_axis * lengths_size_axis + dim_idx;

      if (is_offsets_like) {
        segment_end = lengths_or_offsets_data[lengths_idx + 1];
        segment_length = segment_end - segment_start;
      } else {
        segment_length = lengths_or_offsets_data[lengths_idx];
        segment_end += segment_length;
      }

      // Clamp segment bounds for safety
      segment_start = std::max<int64_t>(0, std::min<int64_t>(segment_start, data_size_axis));
      segment_end = std::max<int64_t>(0, std::min<int64_t>(segment_end, data_size_axis));
      segment_length = segment_end - segment_start;

      for (const auto inner_idx : c10::irange(inner_offset)) {
        // Initialize starting value with safe default
        scalar_t initial_value = static_cast<scalar_t>(0);

        if (initial.has_value()) {
          initial_value = initial.value().to<scalar_t>();
        } else {
          switch (reduction) {
            case ReductionType::MAX:
              initial_value = -std::numeric_limits<scalar_t>::infinity();
              break;
            case ReductionType::MEAN:
            case ReductionType::SUM:
              initial_value = static_cast<scalar_t>(0);
              break;
            case ReductionType::MIN:
              initial_value = std::numeric_limits<scalar_t>::infinity();
              break;
            case ReductionType::PROD:
              initial_value = static_cast<scalar_t>(1);
              break;
            default:
              TORCH_CHECK(false, "segment_reduce: unsupported reduction type");
          }
        }

        // Apply reduction
        for (int64_t j = segment_start; j < segment_end; ++j) {
          int64_t data_index = outer_idx * data_stride_axis * data_size_axis
                               + j * data_stride_axis + inner_idx;
          const auto val = values_data[data_index];

          switch (reduction) {
            case ReductionType::MAX:
              initial_value = at::_isnan(val) ? val : std::max<scalar_t>(initial_value, val);
              break;
            case ReductionType::MEAN:
            case ReductionType::SUM:
              initial_value = initial_value + val;
              break;
            case ReductionType::MIN:
              initial_value = at::_isnan(val) ? val : std::min<scalar_t>(initial_value, val);
              break;
            case ReductionType::PROD:
              initial_value = initial_value * val;
              break;
            default:
              break;
          }
        }

        // Finalize reduction
        TORCH_CHECK(segment_length >= 0);

        if (segment_length == 0 && !initial.has_value() && reduction == ReductionType::MEAN) {
          initial_value = static_cast<scalar_t>(NAN);
        } else if (reduction == ReductionType::MEAN && segment_length > 0 && !at::_isnan(initial_value)) {
          initial_value = initial_value / segment_length;
        }

        int64_t output_index = outer_idx * output_stride_axis * output_size_axis
                               + dim_idx * output_stride_axis + inner_idx;
        output_data[output_index] = initial_value;
      }
    }
  }

  // Copy result back to MPS
  output.copy_(output_cpu);
}

Tensor _segment_reduce_lengths_mps_kernel(
    ReductionType reduction,
    const Tensor& data,
    const Tensor& lengths,
    int64_t axis,
    const std::optional<Scalar>& initial) {
  TORCH_CHECK(data.is_contiguous(), "Expected data to be contiguous.");
  TORCH_CHECK(lengths.is_contiguous(), "Expected lengths to be contiguous.");

  axis = lengths.dim() - 1;
  int64_t segment_count = lengths.size(axis);
  int64_t lengths_stride_axis = lengths.stride(axis);

  auto output_shape = data.sizes().vec();
  output_shape[axis] = segment_count;
  auto output = at::empty(output_shape, data.options());

  // Move lengths to CPU for indexing
  auto lengths_cpu = lengths.to(kCPU);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, data.scalar_type(), "_segment_reduce_mps", [&]() {
        AT_DISPATCH_INDEX_TYPES(lengths.scalar_type(), "_segment_reduce_mps_index", [&]() {
          const auto* lengths_data = lengths_cpu.const_data_ptr<index_t>();
          segment_reduce_mps_kernel_impl<scalar_t, index_t, false>(
              reduction, data, lengths_data, axis, initial, output, segment_count, lengths_stride_axis);
        });
      });

  return output;
}

Tensor _segment_reduce_offsets_mps_kernel(
    ReductionType reduction,
    const Tensor& data,
    const Tensor& offsets,
    int64_t axis,
    const std::optional<Scalar>& initial) {
  TORCH_CHECK(data.is_contiguous(), "Expected data to be contiguous.");
  TORCH_CHECK(offsets.is_contiguous(), "Expected offsets to be contiguous.");

  axis = offsets.dim() - 1;
  int64_t segment_count = offsets.size(axis) - 1;
  int64_t offsets_stride_axis = offsets.stride(axis);

  auto output_shape = data.sizes().vec();
  output_shape[axis] = segment_count;
  auto output = at::empty(output_shape, data.options());

  // Move offsets to CPU for indexing
  auto offsets_cpu = offsets.to(kCPU);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, data.scalar_type(), "_segment_reduce_mps", [&]() {
        AT_DISPATCH_INDEX_TYPES(offsets.scalar_type(), "_segment_reduce_mps_index", [&]() {
          const auto* offsets_data = offsets_cpu.const_data_ptr<index_t>();
          segment_reduce_mps_kernel_impl<scalar_t, index_t, true>(
              reduction, data, offsets_data, axis, initial, output, segment_count, offsets_stride_axis);
        });
      });

  return output;
}

// CPU fallback backward implementation for segment_reduce
// Performance note: Same overhead as forward pass due to CPU fallback.
template <typename scalar_t, typename index_t, bool is_offsets_like = false>
static void segment_reduce_backward_mps_kernel_impl(
    const Tensor& grad_contig,
    const Tensor& output_contig,
    const Tensor& data_contig,
    ReductionType reduction,
    const index_t* lengths_or_offsets_data,
    int64_t axis,
    const std::optional<Scalar>& initial,
    Tensor& grad_input,
    int64_t segment_count,
    int64_t lengths_stride_axis) {

  int64_t outer_offset = 1, inner_offset = 1;
  for (int64_t d = 0; d < axis; d++)
    outer_offset *= output_contig.size(d);
  for (int64_t d = axis + 1; d < output_contig.dim(); d++)
    inner_offset *= output_contig.size(d);

  int64_t lengths_size_axis = is_offsets_like ? segment_count + 1 : segment_count;
  auto data_stride_axis = data_contig.stride(axis);
  auto data_size_axis = data_contig.size(axis);
  auto output_stride_axis = output_contig.stride(axis);
  auto output_size_axis = output_contig.size(axis);

  // Move tensors to CPU (CPU fallback)
  auto grad_cpu = grad_contig.to(kCPU);
  auto output_cpu = output_contig.to(kCPU);
  auto data_cpu = data_contig.to(kCPU);
  auto grad_input_cpu = at::zeros(grad_input.sizes(), grad_input.options().device(kCPU));

  auto* grad_input_data = grad_input_cpu.data_ptr<scalar_t>();
  const auto* grad_data = grad_cpu.const_data_ptr<scalar_t>();
  const auto* output_data = output_cpu.const_data_ptr<scalar_t>();
  const auto* values_data = data_cpu.const_data_ptr<scalar_t>();

  for (const auto outer_idx : c10::irange(outer_offset)) {
    // Validate bounds before processing
    if constexpr (is_offsets_like) {
      validate_offsets(lengths_or_offsets_data, lengths_size_axis, data_size_axis,
                       outer_idx, lengths_stride_axis);
    } else {
      validate_lengths(lengths_or_offsets_data, lengths_size_axis, data_size_axis,
                       outer_idx, lengths_stride_axis);
    }

    int64_t segment_start, segment_length;
    int64_t segment_end = is_offsets_like
        ? lengths_or_offsets_data[outer_idx * lengths_stride_axis * lengths_size_axis]
        : 0;

    for (const auto dim_idx : c10::irange(segment_count)) {
      segment_start = segment_end;
      auto lengths_idx = outer_idx * lengths_stride_axis * lengths_size_axis + dim_idx;

      if (is_offsets_like) {
        segment_end = lengths_or_offsets_data[lengths_idx + 1];
        segment_length = segment_end - segment_start;
      } else {
        segment_length = lengths_or_offsets_data[lengths_idx];
        segment_end += segment_length;
      }

      // Clamp segment bounds for safety
      segment_start = std::max<int64_t>(0, std::min<int64_t>(segment_start, data_size_axis));
      segment_end = std::max<int64_t>(0, std::min<int64_t>(segment_end, data_size_axis));
      segment_length = segment_end - segment_start;

      for (const auto inner_idx : c10::irange(inner_offset)) {
        int64_t output_index = outer_idx * output_stride_axis * output_size_axis
                               + dim_idx * output_stride_axis + inner_idx;

        scalar_t grad_val = grad_data[output_index];
        scalar_t output_val = output_data[output_index];

        if (reduction == ReductionType::MAX || reduction == ReductionType::MIN) {
          // Count how many elements equal the max/min value for proper gradient averaging
          int64_t count = 0;
          for (int64_t j = segment_start; j < segment_end; ++j) {
            int64_t data_index = outer_idx * data_stride_axis * data_size_axis
                                 + j * data_stride_axis + inner_idx;
            if (values_data[data_index] == output_val) {
              count++;
            }
          }

          // Distribute gradient evenly among all tied elements
          scalar_t distributed_grad = count > 0 ? grad_val / static_cast<scalar_t>(count) : static_cast<scalar_t>(0);

          for (int64_t j = segment_start; j < segment_end; ++j) {
            int64_t data_index = outer_idx * data_stride_axis * data_size_axis
                                 + j * data_stride_axis + inner_idx;
            scalar_t data_val = values_data[data_index];
            grad_input_data[data_index] = (data_val == output_val) ? distributed_grad : static_cast<scalar_t>(0);
          }
        } else {
          for (int64_t j = segment_start; j < segment_end; ++j) {
            int64_t data_index = outer_idx * data_stride_axis * data_size_axis
                                 + j * data_stride_axis + inner_idx;
            scalar_t data_val = values_data[data_index];

            if (reduction == ReductionType::SUM) {
              grad_input_data[data_index] = grad_val;
            } else if (reduction == ReductionType::MEAN) {
              grad_input_data[data_index] = segment_length > 0 ? grad_val / static_cast<scalar_t>(segment_length) : static_cast<scalar_t>(0);
            } else if (reduction == ReductionType::PROD) {
              if (data_val != static_cast<scalar_t>(0)) {
                grad_input_data[data_index] = grad_val * output_val / data_val;
              } else {
                // Compute product of all other elements
                scalar_t prod = static_cast<scalar_t>(1);
                for (int64_t k = segment_start; k < segment_end; ++k) {
                  if (k != j) {
                    int64_t other_idx = outer_idx * data_stride_axis * data_size_axis
                                        + k * data_stride_axis + inner_idx;
                    prod *= values_data[other_idx];
                  }
                }
                grad_input_data[data_index] = grad_val * prod;
              }
            }
          }
        }
      }
    }
  }

  grad_input.copy_(grad_input_cpu);
}

Tensor _segment_reduce_lengths_backward_mps_kernel(
    const Tensor& grad,
    const Tensor& output,
    const Tensor& data,
    ReductionType reduction,
    const Tensor& lengths,
    int64_t axis,
    const std::optional<Scalar>& initial) {

  auto grad_contig = grad.contiguous();
  auto output_contig = output.contiguous();
  auto data_contig = data.contiguous();

  axis = lengths.dim() - 1;
  int64_t segment_count = lengths.size(axis);
  int64_t lengths_stride_axis = lengths.stride(axis);

  auto grad_input = at::zeros(data.sizes(), data.options());
  auto lengths_cpu = lengths.to(kCPU);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, data.scalar_type(), "_segment_reduce_backward_mps", [&]() {
        AT_DISPATCH_INDEX_TYPES(lengths.scalar_type(), "_segment_reduce_backward_mps_index", [&]() {
          const auto* lengths_data = lengths_cpu.const_data_ptr<index_t>();
          segment_reduce_backward_mps_kernel_impl<scalar_t, index_t, false>(
              grad_contig, output_contig, data_contig, reduction, lengths_data,
              axis, initial, grad_input, segment_count, lengths_stride_axis);
        });
      });

  return grad_input;
}

Tensor _segment_reduce_offsets_backward_mps_kernel(
    const Tensor& grad,
    const Tensor& output,
    const Tensor& data,
    ReductionType reduction,
    const Tensor& offsets,
    int64_t axis,
    const std::optional<Scalar>& initial) {

  auto grad_contig = grad.contiguous();
  auto output_contig = output.contiguous();
  auto data_contig = data.contiguous();

  axis = offsets.dim() - 1;
  int64_t segment_count = offsets.size(axis) - 1;
  int64_t offsets_stride_axis = offsets.stride(axis);

  auto grad_input = at::zeros(data.sizes(), data.options());
  auto offsets_cpu = offsets.to(kCPU);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, data.scalar_type(), "_segment_reduce_backward_mps", [&]() {
        AT_DISPATCH_INDEX_TYPES(offsets.scalar_type(), "_segment_reduce_backward_mps_index", [&]() {
          const auto* offsets_data = offsets_cpu.const_data_ptr<index_t>();
          segment_reduce_backward_mps_kernel_impl<scalar_t, index_t, true>(
              grad_contig, output_contig, data_contig, reduction, offsets_data,
              axis, initial, grad_input, segment_count, offsets_stride_axis);
        });
      });

  return grad_input;
}

} // namespace mps

REGISTER_DISPATCH(_segment_reduce_lengths_stub, &mps::_segment_reduce_lengths_mps_kernel)
REGISTER_DISPATCH(_segment_reduce_offsets_stub, &mps::_segment_reduce_offsets_mps_kernel)
REGISTER_DISPATCH(_segment_reduce_lengths_backward_stub, &mps::_segment_reduce_lengths_backward_mps_kernel)
REGISTER_DISPATCH(_segment_reduce_offsets_backward_stub, &mps::_segment_reduce_offsets_backward_mps_kernel)

} // namespace at::native
