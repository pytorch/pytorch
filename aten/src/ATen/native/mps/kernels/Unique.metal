#include <metal_stdlib>
using namespace metal;

// mask[0] = 1; mask[i] = (sorted[i] != sorted[i-1]) for i >= 1.
// The output is int32 so we can run cumsum on it directly via at::cumsum.
// With grid.x > 1 threads per row, only differing rows store into the mask,
// which the host zeroes.
template <typename T>
kernel void unique_mark_boundaries(
    device const T* sorted [[buffer(0)]],
    device int* mask [[buffer(1)]],
    constant long2& shape [[buffer(2)]],
    uint2 tid [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]]) {
  for (long r = tid.y; r < shape.x; r += grid.y) {
    bool differs = r == 0;
    for (long c = tid.x; !differs && c < shape.y; c += grid.x) {
      differs = sorted[r * shape.y + c] != sorted[(r - 1) * shape.y + c];
    }
    if (grid.x == 1) {
      mask[r] = differs;
    } else if (differs) {
      atomic_store_explicit(
          (device atomic_int*)(mask + r), 1, memory_order_relaxed);
    }
  }
}

// At each boundary position i, emit:
//   - the boundary's value into unique_values[scan[i]-1]
//   - the position i into bound_pos[scan[i]-1]
// scan[] is the inclusive scan of the mask, so scan[i]-1 is the 0-indexed
// unique-group ID. Each group ID is written by exactly one thread (the
// boundary thread), so there is no contention.
// Every row r also writes its group ID to inverse[sort_idx ? sort_idx[r] : r].
template <typename T, typename SCAN_T>
kernel void unique_emit(
    device const T* sorted [[buffer(0)]],
    device const int* mask [[buffer(1)]],
    device const SCAN_T* scan [[buffer(2)]],
    device T* unique_values [[buffer(3)]],
    device long* bound_pos [[buffer(4)]],
    device const long* sort_idx [[buffer(5)]],
    device long* inverse [[buffer(6)]],
    constant long2& shape [[buffer(7)]],
    uint2 tid [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]]) {
  for (long r = tid.y; r < shape.x; r += grid.y) {
    if (tid.x == 0 && inverse != nullptr) {
      inverse[sort_idx != nullptr ? sort_idx[r] : r] = long(scan[r]) - 1;
    }
    if (!mask[r]) {
      continue;
    }
    const long k = long(scan[r]) - 1;
    if (tid.x == 0 && bound_pos != nullptr) {
      bound_pos[k] = r;
    }
    for (long c = tid.x; c < shape.y; c += grid.x) {
      unique_values[k * shape.y + c] = sorted[r * shape.y + c];
    }
  }
}

// counts[k] = bound_pos[k+1] - bound_pos[k] (with bound_pos[num_unique] = N).
kernel void unique_counts(
    device const long* bound_pos [[buffer(0)]],
    device long* counts [[buffer(1)]],
    constant ulong& num_unique [[buffer(2)]],
    constant ulong& numel [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint grid [[threads_per_grid]]) {
  for (ulong k = tid; k < num_unique; k += grid) {
    long next = (k + 1 == num_unique) ? long(numel) : bound_pos[k + 1];
    counts[k] = next - bound_pos[k];
  }
}

#define REGISTER_UNIQUE_FOR_T(T, NAME)                                \
  template [[host_name("unique_mark_boundaries_" #NAME)]] kernel void \
  unique_mark_boundaries<T>(                                          \
      device const T* sorted [[buffer(0)]],                           \
      device int* mask [[buffer(1)]],                                 \
      constant long2& shape [[buffer(2)]],                            \
      uint2 tid [[thread_position_in_grid]],                          \
      uint2 grid [[threads_per_grid]]);                               \
  template [[host_name("unique_emit_" #NAME "_32")]] kernel void      \
  unique_emit<T, int>(                                                \
      device const T* sorted [[buffer(0)]],                           \
      device const int* mask [[buffer(1)]],                           \
      device const int* scan [[buffer(2)]],                           \
      device T* unique_values [[buffer(3)]],                          \
      device long* bound_pos [[buffer(4)]],                           \
      device const long* sort_idx [[buffer(5)]],                      \
      device long* inverse [[buffer(6)]],                             \
      constant long2& shape [[buffer(7)]],                            \
      uint2 tid [[thread_position_in_grid]],                          \
      uint2 grid [[threads_per_grid]]);                               \
  template [[host_name("unique_emit_" #NAME "_64")]] kernel void      \
  unique_emit<T, long>(                                               \
      device const T* sorted [[buffer(0)]],                           \
      device const int* mask [[buffer(1)]],                           \
      device const long* scan [[buffer(2)]],                          \
      device T* unique_values [[buffer(3)]],                          \
      device long* bound_pos [[buffer(4)]],                           \
      device const long* sort_idx [[buffer(5)]],                      \
      device long* inverse [[buffer(6)]],                             \
      constant long2& shape [[buffer(7)]],                            \
      uint2 tid [[thread_position_in_grid]],                          \
      uint2 grid [[threads_per_grid]]);

REGISTER_UNIQUE_FOR_T(float, float)
REGISTER_UNIQUE_FOR_T(half, half)
REGISTER_UNIQUE_FOR_T(bfloat, bfloat)
REGISTER_UNIQUE_FOR_T(long, long)
REGISTER_UNIQUE_FOR_T(int, int)
REGISTER_UNIQUE_FOR_T(short, short)
REGISTER_UNIQUE_FOR_T(char, char)
REGISTER_UNIQUE_FOR_T(uchar, uchar)
REGISTER_UNIQUE_FOR_T(bool, bool)
REGISTER_UNIQUE_FOR_T(ulong, ulong)
REGISTER_UNIQUE_FOR_T(uint, uint)
REGISTER_UNIQUE_FOR_T(ushort, ushort)
