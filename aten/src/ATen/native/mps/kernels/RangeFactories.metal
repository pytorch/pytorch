#include <metal_stdlib>
using namespace metal;

template <typename T>
inline T linspace_cast(float v) {
  return static_cast<T>(v);
}
template <>
inline float2 linspace_cast(float v) {
  return float2(v, 0.0);
}

struct RangeVals {
  float start;
  float step;
  float end;
};

template <typename I>
struct RangeIdx {
  I halfway;
  I steps;
  I stride;
};

// Halfway split anchors both endpoints exactly (out[0]==start,
// out[steps-1]==end).
template <typename I>
inline float linspace_value(I i, RangeVals v, I halfway, I steps) {
  return i < halfway ? v.start + v.step * static_cast<float>(i)
                     : v.end - v.step * static_cast<float>(steps - i - 1);
}

template <typename T, typename I>
kernel void linspace(
    device T* out [[buffer(0)]],
    constant RangeVals& v [[buffer(1)]],
    constant RangeIdx<I>& p [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  const I i = static_cast<I>(index);
  const float val = linspace_value(i, v, p.halfway, p.steps);
  out[i * p.stride] = linspace_cast<T>(val);
}

// Multi-dim non-contiguous output: scatter each element to its strided offset.
template <typename T>
kernel void linspace_strided(
    device T* out [[buffer(0)]],
    constant RangeVals& v [[buffer(1)]],
    constant RangeIdx<long>& p [[buffer(2)]],
    constant int& ndim [[buffer(3)]],
    constant long* sizes [[buffer(4)]],
    constant long* strides [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
  const long i = static_cast<long>(index);
  const float val = linspace_value(i, v, p.halfway, p.steps);
  long off = 0;
  long rem = i;
  for (int d = ndim - 1; d >= 0; --d) {
    off += (rem % sizes[d]) * strides[d];
    rem /= sizes[d];
  }
  out[off] = linspace_cast<T>(val);
}

#define REGISTER_LINSPACE_OP(DTYPE)                              \
  template [[host_name("linspace_" #DTYPE "_i32")]] kernel void  \
  linspace<DTYPE, int>(                                          \
      device DTYPE * out [[buffer(0)]],                          \
      constant RangeVals & v [[buffer(1)]],                      \
      constant RangeIdx<int> & p [[buffer(2)]],                  \
      uint index [[thread_position_in_grid]]);                   \
  template [[host_name("linspace_" #DTYPE "_i64")]] kernel void  \
  linspace<DTYPE, long>(                                         \
      device DTYPE * out [[buffer(0)]],                          \
      constant RangeVals & v [[buffer(1)]],                      \
      constant RangeIdx<long> & p [[buffer(2)]],                 \
      uint index [[thread_position_in_grid]]);                   \
  template [[host_name("linspace_strided_" #DTYPE)]] kernel void \
  linspace_strided<DTYPE>(                                       \
      device DTYPE * out [[buffer(0)]],                          \
      constant RangeVals & v [[buffer(1)]],                      \
      constant RangeIdx<long> & p [[buffer(2)]],                 \
      constant int& ndim [[buffer(3)]],                          \
      constant long* sizes [[buffer(4)]],                        \
      constant long* strides [[buffer(5)]],                      \
      uint index [[thread_position_in_grid]]);

REGISTER_LINSPACE_OP(float);
REGISTER_LINSPACE_OP(half);
REGISTER_LINSPACE_OP(bfloat);
REGISTER_LINSPACE_OP(float2);
REGISTER_LINSPACE_OP(long);
REGISTER_LINSPACE_OP(int);
REGISTER_LINSPACE_OP(short);
REGISTER_LINSPACE_OP(char);
REGISTER_LINSPACE_OP(uchar);
REGISTER_LINSPACE_OP(bool);
