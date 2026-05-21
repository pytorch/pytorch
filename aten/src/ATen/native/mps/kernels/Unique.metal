#include <metal_stdlib>
using namespace metal;

// mask[0] = 1; mask[i] = (sorted[i] != sorted[i-1]) for i >= 1.
// The output is int32 so we can run cumsum on it directly via at::cumsum.
template <typename T>
kernel void unique_mark_boundaries(
    constant T* sorted [[buffer(0)]],
    device int* mask [[buffer(1)]],
    constant ulong& numel [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  if (ulong(tid) >= numel) {
    return;
  }
  if (tid == 0) {
    mask[0] = 1;
  } else {
    mask[tid] = (sorted[tid] != sorted[tid - 1]) ? 1 : 0;
  }
}

// At each boundary position i, emit:
//   - the boundary's value into unique_values[scan[i]-1]
//   - the position i into bound_pos[scan[i]-1]
// scan[] is the inclusive scan of the mask, so scan[i]-1 is the 0-indexed
// unique-group ID. Each group ID is written by exactly one thread (the
// boundary thread), so there is no contention.
template <typename T, typename SCAN_T>
kernel void unique_emit(
    constant T* sorted [[buffer(0)]],
    constant int* mask [[buffer(1)]],
    constant SCAN_T* scan [[buffer(2)]],
    device T* unique_values [[buffer(3)]],
    device long* bound_pos [[buffer(4)]],
    constant ulong& numel [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  if (ulong(tid) >= numel) {
    return;
  }
  if (mask[tid]) {
    long k = long(scan[tid]) - 1;
    unique_values[k] = sorted[tid];
    bound_pos[k] = long(tid);
  }
}

// counts[k] = bound_pos[k+1] - bound_pos[k] (with bound_pos[num_unique] = N).
kernel void unique_counts(
    constant long* bound_pos [[buffer(0)]],
    device long* counts [[buffer(1)]],
    constant ulong& num_unique [[buffer(2)]],
    constant ulong& numel [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  if (ulong(tid) >= num_unique) {
    return;
  }
  long next = (ulong(tid + 1) == num_unique) ? long(numel) : bound_pos[tid + 1];
  counts[tid] = next - bound_pos[tid];
}

// inverse[sort_idx[k]] = scan[k] - 1.
// sort_idx is a permutation of [0, N) so writes are conflict-free.
template <typename SCAN_T>
kernel void unique_inverse(
    constant long* sort_idx [[buffer(0)]],
    constant SCAN_T* scan [[buffer(1)]],
    device long* inverse [[buffer(2)]],
    constant ulong& numel [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  if (ulong(tid) >= numel) {
    return;
  }
  inverse[sort_idx[tid]] = long(scan[tid]) - 1;
}

#define REGISTER_UNIQUE_FOR_T(T, NAME)                                \
  template [[host_name("unique_mark_boundaries_" #NAME)]] kernel void \
  unique_mark_boundaries<T>(                                          \
      constant T * sorted [[buffer(0)]],                              \
      device int* mask [[buffer(1)]],                                 \
      constant ulong& numel [[buffer(2)]],                            \
      uint tid [[thread_position_in_grid]]);                          \
  template [[host_name("unique_emit_" #NAME "_32")]] kernel void      \
  unique_emit<T, int>(                                                \
      constant T * sorted [[buffer(0)]],                              \
      constant int* mask [[buffer(1)]],                               \
      constant int* scan [[buffer(2)]],                               \
      device T* unique_values [[buffer(3)]],                          \
      device long* bound_pos [[buffer(4)]],                           \
      constant ulong& numel [[buffer(5)]],                            \
      uint tid [[thread_position_in_grid]]);                          \
  template [[host_name("unique_emit_" #NAME "_64")]] kernel void      \
  unique_emit<T, long>(                                               \
      constant T * sorted [[buffer(0)]],                              \
      constant int* mask [[buffer(1)]],                               \
      constant long* scan [[buffer(2)]],                              \
      device T* unique_values [[buffer(3)]],                          \
      device long* bound_pos [[buffer(4)]],                           \
      constant ulong& numel [[buffer(5)]],                            \
      uint tid [[thread_position_in_grid]]);

REGISTER_UNIQUE_FOR_T(float, float)
REGISTER_UNIQUE_FOR_T(half, half)
REGISTER_UNIQUE_FOR_T(bfloat, bfloat)
REGISTER_UNIQUE_FOR_T(long, long)
REGISTER_UNIQUE_FOR_T(int, int)
REGISTER_UNIQUE_FOR_T(short, short)
REGISTER_UNIQUE_FOR_T(char, char)
REGISTER_UNIQUE_FOR_T(uchar, uchar)
REGISTER_UNIQUE_FOR_T(bool, bool)

template [[host_name("unique_inverse_32")]] kernel void unique_inverse<int>(
    constant long* sort_idx [[buffer(0)]],
    constant int* scan [[buffer(1)]],
    device long* inverse [[buffer(2)]],
    constant ulong& numel [[buffer(3)]],
    uint tid [[thread_position_in_grid]]);

template [[host_name("unique_inverse_64")]] kernel void unique_inverse<long>(
    constant long* sort_idx [[buffer(0)]],
    constant long* scan [[buffer(1)]],
    device long* inverse [[buffer(2)]],
    constant ulong& numel [[buffer(3)]],
    uint tid [[thread_position_in_grid]]);
