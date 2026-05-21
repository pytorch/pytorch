#include <metal_stdlib>
using namespace metal;

// Atomically increments the count for each input element's bin.
// `mtl_dispatch1DJob` dispatches exactly numel threads so no bounds check
// is needed inside the kernel. The accumulator is uint32; the host caps
// numel at UINT32_MAX so the counter cannot overflow.
template <typename IDX_T>
kernel void bincount_unweighted(
    constant IDX_T* indices [[buffer(0)]],
    device atomic_uint* counts [[buffer(1)]],
    constant long& indices_stride [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  long bin = long(indices[tid * indices_stride]);
  atomic_fetch_add_explicit(&counts[bin], 1u, memory_order_relaxed);
}

// Widens uint32 counts to int64 in-place (one thread per bin). Run after
// bincount_unweighted on the same encoder; back-to-back compute dispatches
// on a single encoder are serialised by Metal, so no explicit barrier is
// needed. Doing the widen as a fused dispatch is measurably faster than
// `Tensor::to(kLong)` because it avoids a separate encoder commit / stream
// round-trip.
kernel void bincount_widen_uint_to_long(
    constant uint* counts [[buffer(0)]],
    device long* output [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
  output[tid] = long(counts[tid]);
}

// Per-element float-weighted bincount. atomic_fetch_add_explicit on
// atomic<float> is supported on Apple Silicon (Metal 3 / macOS 13+).
template <typename IDX_T>
kernel void bincount_weighted_float(
    constant IDX_T* indices [[buffer(0)]],
    constant float* weights [[buffer(1)]],
    device atomic_float* output [[buffer(2)]],
    constant long& indices_stride [[buffer(3)]],
    constant long& weights_stride [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
  long bin = long(indices[tid * indices_stride]);
  atomic_fetch_add_explicit(
      &output[bin], weights[tid * weights_stride], memory_order_relaxed);
}

// Per-element int32-weighted bincount.
template <typename IDX_T>
kernel void bincount_weighted_int(
    constant IDX_T* indices [[buffer(0)]],
    constant int* weights [[buffer(1)]],
    device atomic_int* output [[buffer(2)]],
    constant long& indices_stride [[buffer(3)]],
    constant long& weights_stride [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
  long bin = long(indices[tid * indices_stride]);
  atomic_fetch_add_explicit(
      &output[bin], weights[tid * weights_stride], memory_order_relaxed);
}

#define REGISTER_BINCOUNT_FOR_IDX(IDX_T, IDX_NAME)                         \
  template [[host_name("bincount_unweighted_" #IDX_NAME)]] kernel void     \
  bincount_unweighted<IDX_T>(                                              \
      constant IDX_T * indices [[buffer(0)]],                              \
      device atomic_uint * counts [[buffer(1)]],                           \
      constant long& indices_stride [[buffer(2)]],                         \
      uint tid [[thread_position_in_grid]]);                               \
  template [[host_name("bincount_weighted_float_" #IDX_NAME)]] kernel void \
  bincount_weighted_float<IDX_T>(                                          \
      constant IDX_T * indices [[buffer(0)]],                              \
      constant float* weights [[buffer(1)]],                               \
      device atomic_float* output [[buffer(2)]],                           \
      constant long& indices_stride [[buffer(3)]],                         \
      constant long& weights_stride [[buffer(4)]],                         \
      uint tid [[thread_position_in_grid]]);                               \
  template [[host_name("bincount_weighted_int_" #IDX_NAME)]] kernel void   \
  bincount_weighted_int<IDX_T>(                                            \
      constant IDX_T * indices [[buffer(0)]],                              \
      constant int* weights [[buffer(1)]],                                 \
      device atomic_int* output [[buffer(2)]],                             \
      constant long& indices_stride [[buffer(3)]],                         \
      constant long& weights_stride [[buffer(4)]],                         \
      uint tid [[thread_position_in_grid]]);

REGISTER_BINCOUNT_FOR_IDX(char, char)
REGISTER_BINCOUNT_FOR_IDX(short, short)
REGISTER_BINCOUNT_FOR_IDX(int, int)
REGISTER_BINCOUNT_FOR_IDX(long, long)
REGISTER_BINCOUNT_FOR_IDX(uchar, uchar)
