#include <SegmentReduce.h>
#include <c10/metal/error.h>
#include <c10/metal/reduction_utils.h>
#include <c10/metal/utils.h>
#include <metal_stdlib>

using namespace metal;
using namespace c10::metal;

template <typename I>
kernel void segment_validate(
    constant I* offsets,
    device uint* valid,
    constant SegmentReduceParams& p,
    device ErrorMessages* errors,
    constant ulong& base,
    uint tid [[thread_position_in_grid]]) {
  const ulong row = base + tid;
  const ulong start = row * (p.segments + 1);
  long previous = offsets[start];
  bool ok = previous >= 0 && ulong(previous) <= p.axis_size;
  for (ulong s = 1; s <= p.segments; ++s) {
    const long next = offsets[start + s];
    ok = ok && next >= previous && next >= 0 && ulong(next) <= p.axis_size;
    previous = next;
  }
  valid[row] = ok;
  if (!ok) {
    TORCH_REPORT_ERROR(
        errors,
        "segment_reduce(): offsets must be nondecreasing and within the data axis");
  }
}

template <SegmentReduction R>
inline float segment_identity() {
  if (R == SegmentReduction::Max) {
    return -INFINITY;
  } else if (R == SegmentReduction::Min) {
    return INFINITY;
  } else if (R == SegmentReduction::Prod) {
    return 1.0f;
  }
  return 0.0f;
}

template <SegmentReduction R>
inline float segment_combine(float a, float b) {
  if (R == SegmentReduction::Max) {
    return c10::metal::max(a, b);
  } else if (R == SegmentReduction::Min) {
    return c10::metal::min(a, b);
  } else if (R == SegmentReduction::Prod) {
    return a * b;
  }
  return a + b;
}

template <SegmentReduction R>
inline float segment_finalize(float value, ulong length, bool has_initial) {
  if (R == SegmentReduction::Mean) {
    if (length != 0) {
      return value / float(length);
    }
    if (!has_initial) {
      return NAN;
    }
  }
  return value;
}

template <typename T, typename I, SegmentReduction R>
kernel void segment_reduce_serial(
    constant T* data,
    device T* output,
    constant I* offsets,
    constant uint* valid,
    constant SegmentReduceParams& p,
    constant ulong& base,
    uint tid [[thread_position_in_grid]]) {
  const ulong index = base + tid;
  const ulong row = index / p.inner;
  const ulong outer = row / p.segments;
  if (!valid[outer]) {
    return;
  }
  const ulong o = outer * (p.segments + 1) + row % p.segments;
  const ulong start = offsets[o];
  const ulong end = offsets[o + 1];
  const ulong data_base = outer * p.axis_size * p.inner + index % p.inner;
  opmath_t<T> value = T(p.has_initial ? p.initial : segment_identity<R>());
  for (ulong j = start; j < end; ++j) {
    value = segment_combine<R>(value, float(data[data_base + j * p.inner]));
  }
  output[index] =
      T(segment_finalize<R>(float(value), end - start, p.has_initial));
}

// Like CUDA's 1-D segmented reduction, distribute long contiguous segments
// across a threadgroup instead of serializing them on one thread.
template <typename T, typename I, SegmentReduction R>
kernel void segment_reduce_parallel(
    constant T* data,
    device T* output,
    constant I* offsets,
    constant uint* valid,
    constant SegmentReduceParams& p,
    constant ulong& base,
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]]) {
  const ulong index = base + group;
  const ulong outer = index / p.segments;
  if (!valid[outer]) {
    return;
  }
  const ulong o = outer * (p.segments + 1) + index % p.segments;
  const ulong start = offsets[o];
  const ulong end = offsets[o + 1];
  const ulong data_base = outer * p.axis_size;
  float value = segment_identity<R>();
  for (ulong j = start + tid; j < end; j += width) {
    value = segment_combine<R>(value, float(data[data_base + j]));
  }
  threadgroup float partial[8];
  if (R == SegmentReduction::Max) {
    value = threadgroup_max(partial, value, tid, width);
  } else if (R == SegmentReduction::Min) {
    value = threadgroup_min(partial, value, tid, width);
  } else if (R == SegmentReduction::Prod) {
    value = threadgroup_prod(partial, value, tid, width);
  } else {
    value = threadgroup_sum(partial, value, tid, width);
  }
  if (tid == 0) {
    if (p.has_initial) {
      value = start == end ? float(T(p.initial))
                           : segment_combine<R>(float(T(p.initial)), value);
    }
    output[index] = T(segment_finalize<R>(value, end - start, p.has_initial));
  }
}

template <typename T, typename I, SegmentReduction R, bool Parallel = false>
kernel void segment_reduce_backward(
    constant T* grad,
    constant T* output,
    constant T* data,
    device T* grad_input,
    constant I* offsets,
    constant uint* valid,
    constant SegmentReduceParams& p,
    constant ulong& base,
    device float* prod_prefix,
    uint tid [[thread_position_in_grid]],
    uint group [[threadgroup_position_in_grid]],
    uint width [[threads_per_threadgroup]]) {
  const ulong index = base + (Parallel ? group : tid);
  const ulong row = index / p.inner;
  const ulong outer = row / p.segments;
  if (!valid[outer]) {
    return;
  }
  const ulong o = outer * (p.segments + 1) + row % p.segments;
  const ulong start = offsets[o];
  const ulong end = offsets[o + 1];
  const ulong data_base = outer * p.axis_size * p.inner + index % p.inner;
  const float g = float(grad[index]);
  float result = float(output[index]);
  bool parallel_writes = Parallel;
  if (Parallel && R == SegmentReduction::Prod && sizeof(T) < sizeof(float) &&
      result != 0 && !isnan(result)) {
    // Cooperative loads preserve the ordered FP32 product without serial
    // device reads.
    threadgroup float tile[1024];
    threadgroup float product;
    float prefix = T(p.has_initial ? p.initial : 1.0f);
    const uint lane = tid % width;
    for (ulong first = start; first < end; first += 1024) {
      const uint count = uint(min(ulong(1024), end - first));
      for (uint j = lane; j < count; j += width) {
        tile[j] = float(data[data_base + (first + j) * p.inner]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (lane == 0) {
        for (uint j = 0; j < count; ++j) {
          prefix *= tile[j];
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
      product = prefix;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    result = product;
  }
  if (R == SegmentReduction::Prod &&
      ((!Parallel && sizeof(T) < sizeof(float)) || result == 0 || isnan(result))) {
    if (Parallel && tid % width != 0) {
      return;
    }
    parallel_writes = false;
    // Exclusive prefix/suffix products keep zero/NaN handling linear.
    opmath_t<T> prefix = T(p.has_initial ? p.initial : 1.0f);
    for (ulong j = start; j < end; ++j) {
      const ulong input_index = data_base + j * p.inner;
      const float x = float(data[input_index]);
      if (x == 0 || isnan(x)) {
        prod_prefix[input_index] = prefix;
      }
      prefix *= x;
    }
    if (sizeof(T) < sizeof(float)) {
      // The stored low-precision output can overflow even when gradients do
      // not.
      result = prefix;
    }
    if (result == 0 || isnan(result)) {
      opmath_t<T> suffix = 1;
      for (ulong j = end; j > start;) {
        --j;
        const ulong input_index = data_base + j * p.inner;
        const float x = float(data[input_index]);
        const float value = x == 0 || isnan(x)
            ? g * (prod_prefix[input_index] * suffix)
            : g * result / x;
        grad_input[input_index] = T(value);
        suffix *= x;
      }
      return;
    }
  }
  const ulong first = start + (parallel_writes ? tid % width : 0);
  const ulong step = parallel_writes ? width : 1;
  long ties = 0;
  if (R == SegmentReduction::Min || R == SegmentReduction::Max) {
    for (ulong j = first; j < end; j += step) {
      const float x = float(data[data_base + j * p.inner]);
      ties += isnan(x) || x == result;
    }
    if (Parallel) {
      threadgroup long partial[8];
      ties = threadgroup_sum(partial, ties, tid % width, width);
    }
  }
  for (ulong j = first; j < end; j += step) {
    const ulong input_index = data_base + j * p.inner;
    const float x = float(data[input_index]);
    float value = g;
    if (R == SegmentReduction::Min || R == SegmentReduction::Max) {
      value = isnan(x) || x == result ? g : 0.0f;
      // Preserve the CPU/CUDA behavior for nonpositive upstream gradients.
      if (ties > 1 && value > 0) {
        value /= float(ties);
      }
    } else if (R == SegmentReduction::Mean) {
      value /= float(end - start);
    } else if (R == SegmentReduction::Prod) {
      value = g * result / x;
    }
    grad_input[input_index] = T(value);
  }
}

#define REGISTER_SEGMENT(T, I, R)                                 \
  template [[host_name("segment_serial_" #T "_" #I "_" #R)]]      \
  kernel void segment_reduce_serial<T, I, SegmentReduction::R>(   \
      constant T*,                                                \
      device T*,                                                  \
      constant I*,                                                \
      constant uint*,                                             \
      constant SegmentReduceParams&,                              \
      constant ulong&,                                            \
      uint);                                                      \
  template [[host_name("segment_parallel_" #T "_" #I "_" #R)]]    \
  kernel void segment_reduce_parallel<T, I, SegmentReduction::R>( \
      constant T*,                                                \
      device T*,                                                  \
      constant I*,                                                \
      constant uint*,                                             \
      constant SegmentReduceParams&,                              \
      constant ulong&,                                            \
      uint,                                                       \
      uint,                                                       \
      uint);                                                      \
  template [[host_name("segment_backward_" #T "_" #I "_" #R)]]    \
  kernel void segment_reduce_backward<T, I, SegmentReduction::R>( \
      constant T*,                                                \
      constant T*,                                                \
      constant T*,                                                \
      device T*,                                                  \
      constant I*,                                                \
      constant uint*,                                             \
      constant SegmentReduceParams&,                              \
      constant ulong&,                                            \
      device float*,                                              \
      uint,                                                       \
      uint,                                                       \
      uint)

#define REGISTER_SEGMENT_BACKWARD_PARALLEL(T, I, R)                      \
  template [[host_name("segment_backward_parallel_" #T "_" #I "_" #R)]] \
  kernel void segment_reduce_backward<T, I, SegmentReduction::R, true>( \
      constant T*,                                                     \
      constant T*,                                                     \
      constant T*,                                                     \
      device T*,                                                       \
      constant I*,                                                     \
      constant uint*,                                                  \
      constant SegmentReduceParams&,                                   \
      constant ulong&,                                                 \
      device float*,                                                   \
      uint,                                                            \
      uint,                                                            \
      uint)

#define REGISTER_SEGMENT_REDUCTIONS(T, I)        \
  REGISTER_SEGMENT(T, I, Max);                   \
  REGISTER_SEGMENT(T, I, Mean);                  \
  REGISTER_SEGMENT(T, I, Min);                   \
  REGISTER_SEGMENT(T, I, Sum);                   \
  REGISTER_SEGMENT(T, I, Prod);                  \
  REGISTER_SEGMENT_BACKWARD_PARALLEL(T, I, Max);  \
  REGISTER_SEGMENT_BACKWARD_PARALLEL(T, I, Mean); \
  REGISTER_SEGMENT_BACKWARD_PARALLEL(T, I, Min);  \
  REGISTER_SEGMENT_BACKWARD_PARALLEL(T, I, Sum)

#define REGISTER_SEGMENT_INDEX(I)                    \
  template [[host_name("segment_validate_" #I)]]     \
  kernel void segment_validate<I>(                   \
      constant I*,                                   \
      device uint*,                                  \
      constant SegmentReduceParams&,                 \
      device ErrorMessages*,                         \
      constant ulong&,                               \
      uint);                                         \
  REGISTER_SEGMENT_REDUCTIONS(float, I);             \
  REGISTER_SEGMENT_REDUCTIONS(half, I);              \
  REGISTER_SEGMENT_REDUCTIONS(bfloat, I);            \
  REGISTER_SEGMENT_BACKWARD_PARALLEL(half, I, Prod);  \
  REGISTER_SEGMENT_BACKWARD_PARALLEL(bfloat, I, Prod); \
  REGISTER_SEGMENT_BACKWARD_PARALLEL(float, I, Prod)

REGISTER_SEGMENT_INDEX(int);
REGISTER_SEGMENT_INDEX(long);
