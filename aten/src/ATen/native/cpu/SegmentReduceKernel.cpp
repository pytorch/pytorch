#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/ExpandUtils.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/SegmentReduce.h>
#include <ATen/native/cpu/ReduceUtils.h>
#include <ATen/native/cpu/utils.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#endif

namespace at { namespace native {

namespace {

// NB: segment reduce kernel
//
// since `segment_reduce` allows batching, we normalize the input and output shapes as:
//
// input: {B, M, K}; output: {B, N, K};
// offsets: {B, N + 1}; lengths: {B, N}
//
// where: B: batch size or the outer dimension size
//   M: input.size(axis)
//   N: number of segments
//   K: the inner dimension size
//
template <typename scalar_t, typename index_t, ReductionType reduce>
void segment_reduce_kernel_impl(
    const Tensor& output,
    const Tensor& input,
    const Tensor& offsets,
    const c10::optional<Scalar>& initial) {

  int64_t axis = offsets.dim() - 1;

  scalar_t* output_data = output.data_ptr<scalar_t>();
  scalar_t* input_data = input.data_ptr<scalar_t>();
  index_t* offsets_data = offsets.data_ptr<index_t>();

  auto input_sizes = input.sizes().vec();
  auto output_sizes = output.sizes().vec();

  int64_t B = c10::size_to_dim_(axis, input_sizes);
  int64_t M = input_sizes[axis];
  int64_t N = output_sizes[axis];
  int64_t K = c10::size_from_dim_(axis + 1, input_sizes);

  // parallel on {B, N}
  at::parallel_for(0, B * N, 1, [&](int64_t begin, int64_t end) {
    int64_t b{0}, n{0};
    data_index_init(begin, b, B, n, N);

    for (const auto i : c10::irange(begin, end)) {
      scalar_t* output_ptr = output_data + i * K;
      scalar_t* input_ptr = input_data + b * (M * K);
      index_t* offsets_ptr = offsets_data + b * (N + 1);

      // step 1: init the output lane
      init<scalar_t, reduce>(output_ptr, K, initial);

      // step 2: reduce
      int64_t row_start = offsets_ptr[n];
      int64_t row_end = offsets_ptr[n + 1];
      for (const auto m : c10::irange(row_start, row_end)) {
        update<scalar_t, reduce>(output_ptr, input_ptr + m * K, K);
      }

      // step 3: finalize
      write<scalar_t, reduce>(output_ptr, row_end - row_start, K);

      // move to the next {b, n}
      data_index_step(b, B, n, N);
    }
  });
}

Tensor _segment_lengths_to_offsets(const Tensor& lengths, int64_t M) {
  int64_t axis = lengths.dim() - 1;
  auto sizes = lengths.sizes().vec();
  int64_t B = c10::size_to_dim_(axis, sizes);
  int64_t N = sizes[axis];
  sizes[axis] = N + 1;

  // lengths: {B, N}; offsets: {B, N + 1}
  auto offsets = at::empty(sizes, lengths.options());
  AT_DISPATCH_INDEX_TYPES(lengths.scalar_type(), "_segment_lengths_to_offsets", [&]() {
    index_t* lengths_data = lengths.data_ptr<index_t>();
    index_t* offsets_data = offsets.data_ptr<index_t>();

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
      for (const auto b : c10::irange(begin, end)) {
        index_t* lengths_ptr = lengths_data + b * N;
        index_t* offsets_ptr = offsets_data + b * (N + 1);
        index_t sum = 0;
        for (const auto n: c10::irange(N)) {
          index_t segment_length = lengths_ptr[n];
          offsets_ptr[n] = sum;
          sum += lengths_ptr[n];
        }
        offsets_ptr[N] = sum;
      }
    });
  });
  return offsets;
}

Tensor _segment_reduce_lengths_cpu_kernel(
    ReductionType reduction,
    const Tensor& data,
    const Tensor& lengths,
    int64_t axis,
    const c10::optional<Scalar>& initial) {
  auto output_shape = data.sizes().vec();
  output_shape[axis] = lengths.size(axis);
  auto output = at::empty(output_shape, data.options());

  auto offsets = _segment_lengths_to_offsets(lengths, data.size(axis));
   AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, data.scalar_type(), "_segment_reduce_lengths_cpu_kernel", [&]() {
    AT_DISPATCH_INDEX_TYPES(offsets.scalar_type(), "_segment_reduce_lengths_cpu_indices", [&]() {
      AT_DISPATCH_REDUCTION_TYPES(reduction, [&]() {
        segment_reduce_kernel_impl<scalar_t, index_t, reduce>(output, data, offsets, initial);
      });
    });
  });
  return output;
}

Tensor _segment_reduce_offsets_cpu_kernel(
    ReductionType reduction,
    const Tensor& data,
    const Tensor& offsets,
    int64_t axis,
    const c10::optional<Scalar>& initial) {
  auto output_shape = data.sizes().vec();
  output_shape[axis] = offsets.size(axis) - 1;
  auto output = at::empty(output_shape, data.options());

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, data.scalar_type(), "_segment_reduce_offsets_cpu_kernel", [&]() {
    AT_DISPATCH_INDEX_TYPES(offsets.scalar_type(), "_segment_reduce_offsets_cpu_indices", [&]() {
      AT_DISPATCH_REDUCTION_TYPES(reduction, [&]() {
        segment_reduce_kernel_impl<scalar_t, index_t, reduce>(output, data, offsets, initial);
      });
    });
  });
  return output;
}

template <typename T, bool is_offsets_like = false>
void _segment_reduce_cpu_lengths_backward_kernel1(
    const Tensor& grad_contig,
    const Tensor& output_contig,
    const Tensor& data_contig,
    ReductionType reduction,
    const T* lengths_data,
    int64_t axis,
    const c10::optional<Scalar>& initial,
    Tensor& grad_input,
    int64_t segment_count,
    int64_t lengths_stride_axis) {
  // outer_offset is the size of the outer dimensions of output (before axis)
  // inner_offset is the size of the inner dimensions of output (after axis)
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
  // TODO: Switch to TensorIterator for better maintainablility and
  // readability
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16,
      kHalf,
      data_contig.scalar_type(),
      "_segment_reduce_cpu",
      [&]() {
        auto* output_data = output_contig.data_ptr<scalar_t>();
        auto* grad_data = grad_contig.data_ptr<scalar_t>();
        auto* grad_input_data = grad_input.data_ptr<scalar_t>();
        const auto* values_data = data_contig.data_ptr<scalar_t>();
        // Used to calculate exclusive prod
        scalar_t initial_prod_value;
        if (reduction == ReductionType::PROD) {
          if (initial.has_value()) {
            initial_prod_value = initial.value().to<scalar_t>();
          } else {
            initial_prod_value = 1;
          }
        }

        for (const auto outer_idx : c10::irange(outer_offset)) {
          // int64_t lengths_cum_sum = 0;
          int64_t segment_start, segment_length;
          int64_t segment_end = is_offsets_like ?
                                lengths_data[outer_idx * lengths_stride_axis * lengths_size_axis] :
                                0;
          for (const auto dim_idx : c10::irange(segment_count)) {
            // int64_t segment_length = lengths_data[outer_idx * lengths_stride_axis * segment_count + dim_idx];
            segment_start = segment_end;
            auto lengths_idx = outer_idx * lengths_stride_axis * lengths_size_axis + dim_idx;
            if (is_offsets_like) {
              segment_end = lengths_data[lengths_idx + 1];
              segment_length = segment_end - segment_start;
            } else {
              segment_length = lengths_data[lengths_idx];
              segment_end += segment_length;
            }
            if (segment_length == 0) {
              continue;
            }
            for (const auto inner_idx : c10::irange(inner_offset)) {
              int64_t output_index = outer_idx * output_stride_axis * output_size_axis
                                     + dim_idx * output_stride_axis + inner_idx;
              if (reduction == ReductionType::MAX ||
                  reduction == ReductionType::MIN) {
                int64_t counter = 0;
                for (const auto j : c10::irange(segment_start, segment_end)) {
                  int64_t data_index = outer_idx * data_stride_axis * data_size_axis
                                       + j * data_stride_axis + inner_idx;
                  if (at::_isnan(values_data[data_index]) ||
                      values_data[data_index] == output_data[output_index]) {
                    grad_input_data[data_index] = grad_data[output_index];
                    counter++;
                  }
                }
                // Average gradient based on number of maximum elements in
                // the segment
                if (counter < 2) {
                  continue;
                }
                for (const auto j : c10::irange(segment_start, segment_end)) {
                  int64_t data_index = outer_idx * data_stride_axis * data_size_axis
                                       + j * data_stride_axis + inner_idx;
                  if (grad_input_data[data_index] > 0) {
                    grad_input_data[data_index] =
                        grad_input_data[data_index] / counter;
                  }
                }
              } else if (reduction == ReductionType::MEAN) {
                auto grad_val = grad_data[output_index] / segment_length;
                for (const auto j : c10::irange(segment_start, segment_end)) {
                  int64_t data_index = outer_idx * data_stride_axis * data_size_axis
                                       + j * data_stride_axis + inner_idx;
                  grad_input_data[data_index] = grad_val;
                }
              } else if (reduction == ReductionType::SUM) {
                const auto& grad_val = grad_data[output_index];
                for (const auto j : c10::irange(segment_start, segment_end)) {
                  int64_t data_index = outer_idx * data_stride_axis * data_size_axis
                                       + j * data_stride_axis + inner_idx;
                  grad_input_data[data_index] = grad_val;
                }
              } else if (reduction == ReductionType::PROD) {
                const auto& grad_val = grad_data[output_index] * output_data[output_index];
                for (const auto j : c10::irange(segment_start, segment_end)) {
                  int64_t data_index = outer_idx * data_stride_axis * data_size_axis
                                       + j * data_stride_axis + inner_idx;
                  if (at::_isnan(values_data[data_index]) ||
                      values_data[data_index] == 0) {
                    // explicitly compute exclusive prod
                    scalar_t exclusive_prod = initial_prod_value;
                    int64_t idx;
                    for (const auto k : c10::irange(segment_start, segment_end)) {
                      if (k != j) {
                        idx = outer_idx * data_stride_axis * data_size_axis
                              + k * data_stride_axis + inner_idx;
                        exclusive_prod *= values_data[idx];
                      }
                    }
                    grad_input_data[data_index] = grad_data[output_index] * exclusive_prod;
                  } else {
                    grad_input_data[data_index] = grad_val / values_data[data_index];
                  }
                }
              }
            }
          }
        }
      });
}

Tensor _segment_reduce_cpu_lengths_backward_kernel(
    const Tensor& grad_contig,
    const Tensor& output_contig,
    const Tensor& data_contig,
    ReductionType reduction,
    const Tensor& lengths_contig,
    int64_t axis,
    const c10::optional<Scalar>& initial) {
  axis = lengths_contig.dim() - 1;
  int64_t segment_count = lengths_contig.size(axis);
  int64_t lengths_stride_axis = lengths_contig.stride(axis);
  auto grad_input = at::zeros({data_contig.sizes()}, grad_contig.options());

  AT_DISPATCH_INDEX_TYPES(
      lengths_contig.scalar_type(), "_segment_reduce_cpu_lengths_backward_kernel1", [&] {
        const auto* lengths_data = lengths_contig.data_ptr<index_t>();
        _segment_reduce_cpu_lengths_backward_kernel1(
            grad_contig,
            output_contig,
            data_contig,
            reduction,
            lengths_data,
            axis,
            initial,
            grad_input,
            segment_count,
            lengths_stride_axis);
      });

  return grad_input;
}

Tensor _segment_reduce_cpu_offsets_backward_kernel(
    const Tensor& grad_contig,
    const Tensor& output_contig,
    const Tensor& data_contig,
    ReductionType reduction,
    const Tensor& offsets_contig,
    int64_t axis,
    const c10::optional<Scalar>& initial) {
  axis = offsets_contig.dim() - 1;
  int64_t segment_count = offsets_contig.size(axis) - 1;
  int64_t offsets_stride_axis = offsets_contig.stride(axis);
  auto grad_input = at::zeros({data_contig.sizes()}, grad_contig.options());

  AT_DISPATCH_INDEX_TYPES(
      offsets_contig.scalar_type(), "_segment_reduce_cpu_offsets_backward_kernel1", [&] {
        const auto* offsets_data = offsets_contig.data_ptr<index_t>();
        _segment_reduce_cpu_lengths_backward_kernel1<index_t, /*is_offsets_like=*/true>(
            grad_contig,
            output_contig,
            data_contig,
            reduction,
            offsets_data,
            axis,
            initial,
            grad_input,
            segment_count,
            offsets_stride_axis);
      });

  return grad_input;
}

} // anonymous namespace

REGISTER_DISPATCH(_segment_reduce_lengths_stub, &_segment_reduce_lengths_cpu_kernel);
REGISTER_DISPATCH(_segment_reduce_offsets_stub, &_segment_reduce_offsets_cpu_kernel);
REGISTER_DISPATCH(_segment_reduce_lengths_backward_stub, &_segment_reduce_cpu_lengths_backward_kernel);
REGISTER_DISPATCH(_segment_reduce_offsets_backward_stub, &_segment_reduce_cpu_offsets_backward_kernel);

}} // at::native
