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

template <typename T, typename I>
inline T integral_linspace_value(
    I i,
    constant array<ulong, 4>& params,
    I steps) {
  // params = {start bits, end bits, distance quotient, distance remainder}.
  const long start = as_type<long>(params[0]);
  const long end = as_type<long>(params[1]);
  const bool increasing = end >= start;
  const ulong denominator = static_cast<ulong>(steps - 1);
  const bool from_start = i < steps / 2;
  const long base = from_start ? start : end;
  const ulong position = static_cast<ulong>(from_start ? i : steps - i - 1);
  ulong offset = params[2] * position;
  bool has_fraction = false;
  if (params[3] != 0) {
    const ulong fractional_product = params[3] * position;
    offset += fractional_product / denominator;
    has_fraction = fractional_product % denominator != 0;
  }
  const bool add = from_start ? increasing : !increasing;
  long value = as_type<long>(
      add ? as_type<ulong>(base) + offset : as_type<ulong>(base) - offset);

  if (has_fraction) {
    if (add && value < 0) {
      ++value;
    } else if (!add && value > 0) {
      --value;
    }
  }
  return static_cast<T>(value);
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

template <typename T, typename I>
kernel void linspace_integral(
    device T* out [[buffer(0)]],
    constant array<ulong, 4>& params [[buffer(1)]],
    constant array<I, 2>& p [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  const I i = static_cast<I>(index);
  out[i * p[1]] = integral_linspace_value<T>(i, params, p[0]);
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

template <typename T>
kernel void linspace_integral_strided(
    device T* out [[buffer(0)]],
    constant array<ulong, 4>& params [[buffer(1)]],
    constant uint& steps [[buffer(2)]],
    constant int& ndim [[buffer(3)]],
    constant long* sizes [[buffer(4)]],
    constant long* strides [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
  const long off =
      c10::metal::offset_from_thread_index(index, sizes, strides, ndim);
  out[off] = integral_linspace_value<T>(index, params, steps);
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

#define REGISTER_INTEGRAL_LINSPACE_OP(DTYPE)                              \
  template [[host_name("linspace_integral_" #DTYPE "_i32")]] kernel void  \
  linspace_integral<DTYPE, int>(                                          \
      device DTYPE * out [[buffer(0)]],                                   \
      constant array<ulong, 4> & params [[buffer(1)]],                    \
      constant array<int, 2> & p [[buffer(2)]],                           \
      uint index [[thread_position_in_grid]]);                            \
  template [[host_name("linspace_integral_" #DTYPE "_i64")]] kernel void  \
  linspace_integral<DTYPE, long>(                                         \
      device DTYPE * out [[buffer(0)]],                                   \
      constant array<ulong, 4> & params [[buffer(1)]],                    \
      constant array<long, 2> & p [[buffer(2)]],                          \
      uint index [[thread_position_in_grid]]);                            \
  template [[host_name("linspace_integral_strided_" #DTYPE)]] kernel void \
  linspace_integral_strided<DTYPE>(                                       \
      device DTYPE * out [[buffer(0)]],                                   \
      constant array<ulong, 4> & params [[buffer(1)]],                    \
      constant uint & steps [[buffer(2)]],                                \
      constant int& ndim [[buffer(3)]],                                   \
      constant long* sizes [[buffer(4)]],                                 \
      constant long* strides [[buffer(5)]],                               \
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
REGISTER_LINSPACE_OP(bool);
REGISTER_LINSPACE_OP(short);
REGISTER_LINSPACE_OP(char);
REGISTER_LINSPACE_OP(uchar);

REGISTER_INTEGRAL_LINSPACE_OP(long);
REGISTER_INTEGRAL_LINSPACE_OP(int);

REGISTER_ARANGE_OP(float, float);
REGISTER_ARANGE_OP(half, float);
REGISTER_ARANGE_OP(bfloat, float);
REGISTER_ARANGE_OP(long, long);
REGISTER_ARANGE_OP(int, long);
REGISTER_ARANGE_OP(short, long);
REGISTER_ARANGE_OP(char, long);
REGISTER_ARANGE_OP(uchar, long);
