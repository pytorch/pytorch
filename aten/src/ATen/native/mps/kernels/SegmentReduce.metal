#include <ATen/native/mps/kernels/SegmentReduce.h>
#include <metal_stdlib>

using namespace metal;

// The host normalizes the `lengths` form into offsets, so both variants of the
// op reach the same kernel.
struct SegmentBounds {
  long start;
  long end;
  long length;
  long output_index;
  long data_base;
};

inline SegmentBounds segment_bounds(
    constant long* offsets,
    constant SegmentReduceParams& params,
    long tid) {
  const long inner_idx = tid % params.inner_offset;
  const long dim_idx = (tid / params.inner_offset) % params.segment_count;
  const long outer_idx = tid / (params.inner_offset * params.segment_count);

  const long offsets_idx =
      outer_idx * params.offsets_stride_axis * params.offsets_size_axis +
      dim_idx;

  SegmentBounds b;
  b.start = offsets[offsets_idx];
  b.end = offsets[offsets_idx + 1];
  b.length = b.end - b.start;
  b.output_index =
      outer_idx * params.output_stride_axis * params.output_size_axis +
      dim_idx * params.output_stride_axis + inner_idx;
  b.data_base =
      outer_idx * params.data_stride_axis * params.data_size_axis + inner_idx;
  return b;
}

template <typename T>
kernel void segment_reduce(
    constant T* data [[buffer(0)]],
    constant long* offsets [[buffer(1)]],
    device T* output [[buffer(2)]],
    constant SegmentReduceParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  const auto b = segment_bounds(offsets, params, static_cast<long>(tid));
  const auto reduction = static_cast<SegmentReductionType>(params.reduction);

  float acc = 0.0;
  if (params.has_initial) {
    acc = params.initial;
  } else if (reduction == SegmentReductionType::MAX) {
    acc = -INFINITY;
  } else if (reduction == SegmentReductionType::MIN) {
    acc = INFINITY;
  } else if (reduction == SegmentReductionType::PROD) {
    acc = 1.0;
  }

  for (long j = b.start; j < b.end; ++j) {
    const float val =
        static_cast<float>(data[b.data_base + j * params.data_stride_axis]);
    switch (reduction) {
      case SegmentReductionType::MAX:
        // A NaN anywhere in the segment wins and sticks, as on CPU.
        acc = ::metal::isnan(val) ? val : (acc < val ? val : acc);
        break;
      case SegmentReductionType::MIN:
        acc = ::metal::isnan(val) ? val : (val < acc ? val : acc);
        break;
      case SegmentReductionType::MEAN:
      case SegmentReductionType::SUM:
        acc = acc + val;
        break;
      case SegmentReductionType::PROD:
        acc = acc * val;
        break;
    }
  }

  if (reduction == SegmentReductionType::MEAN) {
    if (b.length == 0 && !params.has_initial) {
      acc = NAN;
    } else if (b.length > 0 && !::metal::isnan(acc)) {
      acc = acc / static_cast<float>(b.length);
    }
  }

  output[b.output_index] = static_cast<T>(acc);
}

// `grad_input` is zeroed by the host. Threads own disjoint segments of the
// reduction axis, so no atomics are needed.
template <typename T>
kernel void segment_reduce_backward(
    device T* grad_input [[buffer(0)]],
    constant T* grad [[buffer(1)]],
    constant T* output [[buffer(2)]],
    constant T* data [[buffer(3)]],
    constant long* offsets [[buffer(4)]],
    constant SegmentReduceParams& params [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  const auto b = segment_bounds(offsets, params, static_cast<long>(tid));
  if (b.length == 0) {
    return;
  }
  const auto reduction = static_cast<SegmentReductionType>(params.reduction);
  const float grad_val = static_cast<float>(grad[b.output_index]);

  if (reduction == SegmentReductionType::MAX ||
      reduction == SegmentReductionType::MIN) {
    const float out_val = static_cast<float>(output[b.output_index]);
    long counter = 0;
    for (long j = b.start; j < b.end; ++j) {
      const long idx = b.data_base + j * params.data_stride_axis;
      const float val = static_cast<float>(data[idx]);
      if (::metal::isnan(val) || val == out_val) {
        grad_input[idx] = static_cast<T>(grad_val);
        counter++;
      }
    }
    if (counter < 2) {
      return;
    }
    // Spread the gradient over the tied extrema. The positivity test mirrors
    // the CPU kernel exactly.
    for (long j = b.start; j < b.end; ++j) {
      const long idx = b.data_base + j * params.data_stride_axis;
      if (static_cast<float>(grad_input[idx]) > 0.0) {
        grad_input[idx] =
            static_cast<T>(static_cast<float>(grad_input[idx]) / counter);
      }
    }
    return;
  }

  if (reduction == SegmentReductionType::MEAN ||
      reduction == SegmentReductionType::SUM) {
    const float share = (reduction == SegmentReductionType::MEAN)
        ? grad_val / static_cast<float>(b.length)
        : grad_val;
    for (long j = b.start; j < b.end; ++j) {
      grad_input[b.data_base + j * params.data_stride_axis] =
          static_cast<T>(share);
    }
    return;
  }

  // PROD: divide the segment product out where that is safe, and recompute the
  // exclusive product where the element is zero or NaN.
  const float initial_prod = params.has_initial ? params.initial : 1.0;
  const float scaled = grad_val * static_cast<float>(output[b.output_index]);
  for (long j = b.start; j < b.end; ++j) {
    const long idx = b.data_base + j * params.data_stride_axis;
    const float val = static_cast<float>(data[idx]);
    if (::metal::isnan(val) || val == 0.0) {
      float exclusive = initial_prod;
      for (long k = b.start; k < b.end; ++k) {
        if (k != j) {
          exclusive *= static_cast<float>(
              data[b.data_base + k * params.data_stride_axis]);
        }
      }
      grad_input[idx] = static_cast<T>(grad_val * exclusive);
    } else {
      grad_input[idx] = static_cast<T>(scaled / val);
    }
  }
}

#define REGISTER_SEGMENT_REDUCE(T)                                            \
  template [[host_name("segment_reduce_" #T)]] kernel void segment_reduce<T>( \
      constant T * data [[buffer(0)]],                                        \
      constant long* offsets [[buffer(1)]],                                   \
      device T* output [[buffer(2)]],                                         \
      constant SegmentReduceParams& params [[buffer(3)]],                     \
      uint tid [[thread_position_in_grid]]);                                  \
  template [[host_name("segment_reduce_backward_" #T)]] kernel void           \
  segment_reduce_backward<T>(                                                 \
      device T * grad_input [[buffer(0)]],                                    \
      constant T * grad [[buffer(1)]],                                        \
      constant T * output [[buffer(2)]],                                      \
      constant T * data [[buffer(3)]],                                        \
      constant long* offsets [[buffer(4)]],                                   \
      constant SegmentReduceParams& params [[buffer(5)]],                     \
      uint tid [[thread_position_in_grid]]);

REGISTER_SEGMENT_REDUCE(float);
REGISTER_SEGMENT_REDUCE(half);
REGISTER_SEGMENT_REDUCE(bfloat);
