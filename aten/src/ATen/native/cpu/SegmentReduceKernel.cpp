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

template <typename scalar_t, typename index_t>
void segment_reduce_backward_kernel_impl(
    const Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& output,
    const Tensor& input,
    ReductionType reduce,
    const Tensor& offsets,
    const c10::optional<Scalar>& initial) {

  int64_t axis = offsets.dim() - 1;

  scalar_t* grad_input_data = grad_input.data_ptr<scalar_t>();
  scalar_t* grad_output_data = grad_output.data_ptr<scalar_t>();
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
  using acc_t = vec_scalar_t<scalar_t>;
  using Vec = Vectorized<acc_t>;
  at::parallel_for(0, B * N, 1, [&](int64_t begin, int64_t end) {
    int64_t b{0}, n{0};
    data_index_init(begin, b, B, n, N);

    for (const auto i : c10::irange(begin, end)) {
      scalar_t* grad_input_ptr = grad_input_data + b * (M * K);
      scalar_t* grad_output_ptr = grad_output_data + i * K;
      scalar_t* output_ptr = output_data + i * K;
      scalar_t* input_ptr = input_data + b * (M * K);
      index_t* offsets_ptr = offsets_data + b * (N + 1);

      int64_t row_start = offsets_ptr[n];
      int64_t row_end = offsets_ptr[n + 1];

      // skip the empty segment
      if (row_end == row_start) { continue; }

      if (reduce == ReductionType::SUM) {
        for (const auto m : c10::irange(row_start, row_end)) {
          vec::map<scalar_t>(
              [](Vec x) { return x; },
              grad_input_ptr + m * K,
              grad_output_ptr,
              K);
        }
      } else if (reduce == ReductionType::MEAN) {
        int64_t count = row_end - row_start;
        for (const auto m : c10::irange(row_start, row_end)) {
          vec::map<scalar_t>(
              [count](Vec x) { return x / Vec(count); },
              grad_input_ptr + m * K,
              grad_output_ptr,
              K);
        }
      } else if (reduce == ReductionType::MAX || reduce == ReductionType::MIN) {
        for (const auto k : c10::irange(K)) {
          int64_t counter = 0;
          for (const auto m : c10::irange(row_start, row_end)) {
            scalar_t value = input_ptr[m * K + k];
            if (at::_isnan(value) || value == output_ptr[k]) {
              grad_input_ptr[m * K + k] = grad_output_ptr[k];
              counter++;
            }

            if (counter > 1) {
              for (const auto m : c10::irange(row_start, row_end)) {
                if (grad_input_ptr[m * K + k] != 0) {
                  grad_input_ptr[m * K + k] /=  counter;
                }
              }
            }
          }
        }
      } else {
        for (const auto k : c10::irange(K)) {
          for (const auto m : c10::irange(row_start, row_end)) {
            scalar_t value = input_ptr[m * K + k];
            if (at::_isnan(value) || value == 0) {
              // explicitly compute exclusive prod
              acc_t exclusive_prod = init_value<scalar_t, ReductionType::PROD>(initial);;
              for (const auto m1 : c10::irange(row_start, row_end)) {
                if (m1 != m) { exclusive_prod *= input_ptr[m1 * K + k]; }
              }
              grad_input_ptr[m * K + k] = grad_output_ptr[k] * exclusive_prod;
            } else {
              acc_t grad_val = grad_output_ptr[k] * output_ptr[k];
              grad_input_ptr[m * K + k] = grad_val / value;
            }
          }
        }
      }

      // move to the next {b, n}
      data_index_step(b, B, n, N);
    }
  });
}

Tensor _segment_reduce_cpu_lengths_backward_kernel(
    const Tensor& grad_output,
    const Tensor& output,
    const Tensor& data,
    ReductionType reduction,
    const Tensor& lengths,
    int64_t axis,
    const c10::optional<Scalar>& initial) {

  auto grad_input = at::zeros({data.sizes()}, grad_output.options());
  auto offsets = _segment_lengths_to_offsets(lengths, data.size(axis));
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, data.scalar_type(), "_segment_reduce_cpu_lengths_backward_kernel", [&]() {
    AT_DISPATCH_INDEX_TYPES(offsets.scalar_type(), "_segment_reduce_cpu_lengths_backward_indices", [&]() {
      segment_reduce_backward_kernel_impl<scalar_t, index_t>(grad_input, grad_output, output, data, reduction, offsets, initial);
    });
  });
  return grad_input;
}

Tensor _segment_reduce_cpu_offsets_backward_kernel(
    const Tensor& grad_output,
    const Tensor& output,
    const Tensor& data,
    ReductionType reduction,
    const Tensor& offsets,
    int64_t axis,
    const c10::optional<Scalar>& initial) {

  auto grad_input = at::zeros({data.sizes()}, grad_output.options());
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, data.scalar_type(), "_segment_reduce_cpu_offsets_backward_kernel", [&]() {
    AT_DISPATCH_INDEX_TYPES(offsets.scalar_type(), "_segment_reduce_cpu_offsets_backward_indices", [&]() {
      segment_reduce_backward_kernel_impl<scalar_t, index_t>(grad_input, grad_output, output, data, reduction, offsets, initial);
    });
  });
  return grad_input;
}

} // anonymous namespace

REGISTER_DISPATCH(_segment_reduce_lengths_stub, &_segment_reduce_lengths_cpu_kernel);
REGISTER_DISPATCH(_segment_reduce_offsets_stub, &_segment_reduce_offsets_cpu_kernel);
REGISTER_DISPATCH(_segment_reduce_lengths_backward_stub, &_segment_reduce_cpu_lengths_backward_kernel);
REGISTER_DISPATCH(_segment_reduce_offsets_backward_stub, &_segment_reduce_cpu_offsets_backward_kernel);

}} // at::native
