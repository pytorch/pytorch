#include <c10/metal/atomic.h>
#include <metal_stdlib>
using namespace metal;
using c10::metal::AtomicType;
using c10::metal::AtomicType_t;

// Atomically increments the count for each input element's bin.
// `mtl_dispatch1DJob` dispatches exactly numel threads so no bounds check
// is needed inside the kernel. Uses a uint32 accumulator with a fused
// widen-to-int64 pass (below) rather than `AtomicType<long>`, which would
// double the atomic ops per thread (low + high lanes) and regress on the
// highly-skewed workload that motivates this op. The host caps numel at
// UINT32_MAX so the per-bin uint32 counter cannot overflow.
template <typename IDX_T>
kernel void bincount_unweighted(
    constant IDX_T* indices [[buffer(0)]],
    device atomic_uint* counts [[buffer(1)]],
    constant long& indices_stride [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  long bin = long(indices[tid * indices_stride]);
  atomic_fetch_add_explicit(&counts[bin], 1u, memory_order_relaxed);
}

// Widens uint32 counts to int64. Run as a fused dispatch on the same encoder
// as bincount_unweighted. The counts buffer is hazard-tracked tensor storage,
// so Metal auto-inserts the write->read barrier between the two same-encoder
// dispatches; no explicit barrier is needed. (Serial-encoder ordering alone
// does not prevent dispatch overlap.) In-encoder widening is measurably faster
// than `Tensor::to(kLong)` because it avoids a separate encoder commit /
// stream round-trip.
kernel void bincount_widen_uint_to_long(
    constant uint* counts [[buffer(0)]],
    device long* output [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
  output[tid] = long(counts[tid]);
}

// Per-element weighted bincount. All weighted bincount accumulates in and
// returns float32 (see the host wrapper for the dtype rationale), so the host
// casts any non-float32 weights to float first. atomic<float> is a native
// (non-CAS) Metal atomic, unlike AtomicType<half>/<bfloat> which are CAS-based
// and ~75x slower under contention.
template <typename IDX_T>
kernel void bincount_weighted(
    constant IDX_T* indices [[buffer(0)]],
    constant float* weights [[buffer(1)]],
    device AtomicType_t<float>* output [[buffer(2)]],
    constant long& indices_stride [[buffer(3)]],
    constant long& weights_stride [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
  long bin = long(indices[tid * indices_stride]);
  AtomicType<float>::atomic_add(output, bin, weights[tid * weights_stride]);
}

#define REGISTER_BINCOUNT_UNWEIGHTED(IDX_T, IDX_NAME)                  \
  template [[host_name("bincount_unweighted_" #IDX_NAME)]] kernel void \
  bincount_unweighted<IDX_T>(                                          \
      constant IDX_T * indices [[buffer(0)]],                          \
      device atomic_uint * counts [[buffer(1)]],                       \
      constant long& indices_stride [[buffer(2)]],                     \
      uint tid [[thread_position_in_grid]]);

#define REGISTER_BINCOUNT_WEIGHTED(IDX_T, IDX_NAME)                  \
  template [[host_name("bincount_weighted_" #IDX_NAME)]] kernel void \
  bincount_weighted<IDX_T>(                                          \
      constant IDX_T * indices [[buffer(0)]],                        \
      constant float* weights [[buffer(1)]],                         \
      device AtomicType_t<float>* output [[buffer(2)]],              \
      constant long& indices_stride [[buffer(3)]],                   \
      constant long& weights_stride [[buffer(4)]],                   \
      uint tid [[thread_position_in_grid]]);

#define REGISTER_BINCOUNT_FOR_IDX(IDX_T, IDX_NAME) \
  REGISTER_BINCOUNT_UNWEIGHTED(IDX_T, IDX_NAME)    \
  REGISTER_BINCOUNT_WEIGHTED(IDX_T, IDX_NAME)

REGISTER_BINCOUNT_FOR_IDX(char, char)
REGISTER_BINCOUNT_FOR_IDX(short, short)
REGISTER_BINCOUNT_FOR_IDX(int, int)
REGISTER_BINCOUNT_FOR_IDX(long, long)
REGISTER_BINCOUNT_FOR_IDX(uchar, uchar)
