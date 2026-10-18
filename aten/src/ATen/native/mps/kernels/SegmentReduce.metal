#include <ATen/native/mps/kernels/SegmentReduce.h>
#include <c10/metal/reduction_utils.h>
#include <metal_stdlib>

using namespace metal;
using namespace c10::metal;

template <SegmentReduction R>
using SegmentOp = conditional_t<
    R == SegmentReduction::Max,
    MaxOp<float>,
    conditional_t<
        R == SegmentReduction::Min,
        MinOp<float>,
        conditional_t<
            R == SegmentReduction::Prod,
            ProdOp<float>,
            SumOp<float>>>>;

template <typename T, typename I, SegmentReduction R, bool Parallel>
kernel void segment_reduce_forward(
    constant T* data,
    device T* output,
    constant I* offsets,
    constant SegmentReduceParams& p,
    constant ulong& base,
    uint tid [[thread_position_in_grid]],
    uint group [[threadgroup_position_in_grid]],
    uint width [[threads_per_threadgroup]]) {
  using Op = SegmentOp<R>;
  const ulong index = base + (Parallel ? group : tid);
  const ulong row = index / p.inner;
  const ulong outer = row / p.segments;
  const ulong o = outer * (p.segments + 1) + row % p.segments;
  const ulong start = offsets[o];
  const ulong end = offsets[o + 1];
  const ulong data_base = outer * p.axis_size * p.inner + index % p.inner;
  const uint lane = Parallel ? tid % width : 0;
  const uint step = Parallel ? width : 1;
  float value = Parallel ? Op::identity()
                         : float(T(p.has_initial ? p.initial : Op::identity()));
  for (ulong j = start + lane; j < end; j += step) {
    value = Op::combine(value, float(data[data_base + j * p.inner]));
  }
  if (Parallel) {
    threadgroup float partial[8];
    value = Op::threadgroup_reduce(partial, value, lane, width);
    if (p.has_initial) {
      value = start == end ? float(T(p.initial))
                           : Op::combine(float(T(p.initial)), value);
    }
  }
  if (lane == 0) {
    if (R == SegmentReduction::Mean) {
      value = end != start ? value / float(end - start)
                           : (p.has_initial ? value : NAN);
    }
    output[index] = T(value);
  }
}

#define REGISTER_SEGMENT_FORWARD(T, I, R, NAME, PARALLEL)                  \
  template [[host_name("segment_" #NAME "_" #T "_" #I "_" #R)]]            \
  kernel void segment_reduce_forward<T, I, SegmentReduction::R, PARALLEL>( \
      constant T*,                                                         \
      device T*,                                                           \
      constant I*,                                                         \
      constant SegmentReduceParams&,                                       \
      constant ulong&,                                                     \
      uint,                                                                \
      uint,                                                                \
      uint)

#define REGISTER_SEGMENT(T, I, R)                   \
  REGISTER_SEGMENT_FORWARD(T, I, R, serial, false); \
  REGISTER_SEGMENT_FORWARD(T, I, R, parallel, true)

#define REGISTER_SEGMENT_REDUCTIONS(T, I) \
  REGISTER_SEGMENT(T, I, Max);            \
  REGISTER_SEGMENT(T, I, Mean);           \
  REGISTER_SEGMENT(T, I, Min);            \
  REGISTER_SEGMENT(T, I, Sum);            \
  REGISTER_SEGMENT(T, I, Prod)

#define REGISTER_SEGMENT_INDEX(I)        \
  REGISTER_SEGMENT_REDUCTIONS(float, I); \
  REGISTER_SEGMENT_REDUCTIONS(half, I);  \
  REGISTER_SEGMENT_REDUCTIONS(bfloat, I)

REGISTER_SEGMENT_INDEX(int);
REGISTER_SEGMENT_INDEX(long);
