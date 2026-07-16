#include <c10/metal/indexing.h>
#include <metal_stdlib>
using namespace metal;

// Halfway split anchors both endpoints exactly (out[0]==start,
// out[steps-1]==end). v = {start, step, end}.
template <typename I>
inline float linspace_value(I i, constant array<float, 3>& v, I steps) {
  const I halfway = steps / 2;
  return i < halfway ? v[0] + v[1] * static_cast<float>(i)
                     : v[2] - v[1] * static_cast<float>(steps - i - 1);
}

// p = {steps, stride}.
template <typename T, typename I>
kernel void linspace(
    device T* out [[buffer(0)]],
    constant array<float, 3>& v [[buffer(1)]],
    constant array<I, 2>& p [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  const I i = static_cast<I>(index);
  out[i * p[1]] = c10::metal::cast_to<T>(linspace_value(i, v, p[0]));
}

template <typename T>
kernel void linspace_strided(
    device T* out [[buffer(0)]],
    constant array<float, 3>& v [[buffer(1)]],
    constant uint& steps [[buffer(2)]],
    constant int& ndim [[buffer(3)]],
    constant long* sizes [[buffer(4)]],
    constant long* strides [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
  const float val = linspace_value(index, v, steps);
  const long off =
      c10::metal::offset_from_thread_index(index, sizes, strides, ndim);
  out[off] = c10::metal::cast_to<T>(val);
}

template <typename T, typename C, typename I>
kernel void arange(
    device T* out [[buffer(0)]],
    constant array<C, 2>& se [[buffer(1)]],
    constant I& stride [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  const C val = se[0] + se[1] * static_cast<C>(index);
  out[static_cast<I>(index) * stride] = c10::metal::cast_to<T>(val);
}

template <typename T, typename C>
kernel void arange_strided(
    device T* out [[buffer(0)]],
    constant array<C, 2>& se [[buffer(1)]],
    constant int& ndim [[buffer(2)]],
    constant long* sizes [[buffer(3)]],
    constant long* strides [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  const C val = se[0] + se[1] * static_cast<C>(index);
  const long off =
      c10::metal::offset_from_thread_index(index, sizes, strides, ndim);
  out[off] = c10::metal::cast_to<T>(val);
}

#define REGISTER_LINSPACE_OP(DTYPE)                              \
  template [[host_name("linspace_" #DTYPE "_i32")]] kernel void  \
  linspace<DTYPE, int>(                                          \
      device DTYPE * out [[buffer(0)]],                          \
      constant array<float, 3> & v [[buffer(1)]],                \
      constant array<int, 2> & p [[buffer(2)]],                  \
      uint index [[thread_position_in_grid]]);                   \
  template [[host_name("linspace_" #DTYPE "_i64")]] kernel void  \
  linspace<DTYPE, long>(                                         \
      device DTYPE * out [[buffer(0)]],                          \
      constant array<float, 3> & v [[buffer(1)]],                \
      constant array<long, 2> & p [[buffer(2)]],                 \
      uint index [[thread_position_in_grid]]);                   \
  template [[host_name("linspace_strided_" #DTYPE)]] kernel void \
  linspace_strided<DTYPE>(                                       \
      device DTYPE * out [[buffer(0)]],                          \
      constant array<float, 3> & v [[buffer(1)]],                \
      constant uint & steps [[buffer(2)]],                       \
      constant int& ndim [[buffer(3)]],                          \
      constant long* sizes [[buffer(4)]],                        \
      constant long* strides [[buffer(5)]],                      \
      uint index [[thread_position_in_grid]]);

#define REGISTER_ARANGE_OP(DTYPE, CTYPE)                       \
  template [[host_name("arange_" #DTYPE "_i32")]] kernel void  \
  arange<DTYPE, CTYPE, int>(                                   \
      device DTYPE * out [[buffer(0)]],                        \
      constant array<CTYPE, 2> & se [[buffer(1)]],             \
      constant int& stride [[buffer(2)]],                      \
      uint index [[thread_position_in_grid]]);                 \
  template [[host_name("arange_" #DTYPE "_i64")]] kernel void  \
  arange<DTYPE, CTYPE, long>(                                  \
      device DTYPE * out [[buffer(0)]],                        \
      constant array<CTYPE, 2> & se [[buffer(1)]],             \
      constant long& stride [[buffer(2)]],                     \
      uint index [[thread_position_in_grid]]);                 \
  template [[host_name("arange_strided_" #DTYPE)]] kernel void \
  arange_strided<DTYPE, CTYPE>(                                \
      device DTYPE * out [[buffer(0)]],                        \
      constant array<CTYPE, 2> & se [[buffer(1)]],             \
      constant int& ndim [[buffer(2)]],                        \
      constant long* sizes [[buffer(3)]],                      \
      constant long* strides [[buffer(4)]],                    \
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

REGISTER_ARANGE_OP(float, float);
REGISTER_ARANGE_OP(half, float);
REGISTER_ARANGE_OP(bfloat, float);
REGISTER_ARANGE_OP(long, long);
REGISTER_ARANGE_OP(int, long);
REGISTER_ARANGE_OP(short, long);
REGISTER_ARANGE_OP(char, long);
REGISTER_ARANGE_OP(uchar, long);
