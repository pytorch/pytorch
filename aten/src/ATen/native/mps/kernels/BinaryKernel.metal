#include <metal_stdlib>
using namespace metal;

template <typename T>
kernel void fmax(
    constant void* input_ [[buffer(0)]],
    constant void* other_ [[buffer(1)]],
    device void* out_ [[buffer(2)]],
    constant uint3* offsets [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  device T* out = (device T*)((device uint8_t*)out_ + offsets[tid].x);
  constant T* input = (constant T*)((constant uint8_t*)input_ + offsets[tid].y);
  constant T* other = (constant T*)((constant uint8_t*)other_ + offsets[tid].z);

  *out = static_cast<T>(fmax(*input, *other));
}

template <typename T>
kernel void fmin(
    constant void* input_ [[buffer(0)]],
    constant void* other_ [[buffer(1)]],
    device void* out_ [[buffer(2)]],
    constant uint3* offsets [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  device T* out = (device T*)((device uint8_t*)out_ + offsets[tid].x);
  constant T* input = (constant T*)((constant uint8_t*)input_ + offsets[tid].y);
  constant T* other = (constant T*)((constant uint8_t*)other_ + offsets[tid].z);

  *out = static_cast<T>(fmin(*input, *other));
}

template <typename T>
kernel void copysign(
    constant void* input_ [[buffer(0)]],
    constant void* other_ [[buffer(1)]],
    device void* out_ [[buffer(2)]],
    constant uint3* offsets [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  device T* out = (device T*)((device uint8_t*)out_ + offsets[tid].x);
  constant T* input = (constant T*)((constant uint8_t*)input_ + offsets[tid].y);
  constant T* other = (constant T*)((constant uint8_t*)other_ + offsets[tid].z);

  *out = static_cast<T>(copysign(*input, *other));
}

template <typename T>
kernel void copysign_integral(
    constant void* input_ [[buffer(0)]],
    constant void* other_ [[buffer(1)]],
    device void* out_ [[buffer(2)]],
    constant uint3* offsets [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  device float* out = (device float*)((device uint8_t*)out_ + offsets[tid].x);
  constant T* input = (constant T*)((constant uint8_t*)input_ + offsets[tid].y);
  constant T* other = (constant T*)((constant uint8_t*)other_ + offsets[tid].z);

  *out = copysign(static_cast<float>(*input), static_cast<float>(*other));
}

#define REGISTER_FMAX_OP(DTYPE)                                   \
  template [[host_name("fmax_" #DTYPE)]] kernel void fmax<DTYPE>( \
      constant void* input_ [[buffer(0)]],                        \
      constant void* other_ [[buffer(1)]],                        \
      device void* out_ [[buffer(2)]],                            \
      constant uint3* offsets [[buffer(3)]],                      \
      uint tid [[thread_position_in_grid]]);

#define REGISTER_FMIN_OP(DTYPE)                                   \
  template [[host_name("fmin_" #DTYPE)]] kernel void fmin<DTYPE>( \
      constant void* input_ [[buffer(0)]],                        \
      constant void* other_ [[buffer(1)]],                        \
      device void* out_ [[buffer(2)]],                            \
      constant uint3* offsets [[buffer(3)]],                      \
      uint tid [[thread_position_in_grid]]);

#define REGISTER_COPYSIGN_OP(DTYPE)                                       \
  template [[host_name("copysign_" #DTYPE)]] kernel void copysign<DTYPE>( \
      constant void* input_ [[buffer(0)]],                                \
      constant void* other_ [[buffer(1)]],                                \
      device void* out_ [[buffer(2)]],                                    \
      constant uint3* offsets [[buffer(3)]],                              \
      uint tid [[thread_position_in_grid]]);

#define REGISTER_COPYSIGN_INTEGRAL_OP(DTYPE)             \
  template [[host_name("copysign_" #DTYPE)]] kernel void \
  copysign_integral<DTYPE>(                              \
      constant void* input_ [[buffer(0)]],               \
      constant void* other_ [[buffer(1)]],               \
      device void* out_ [[buffer(2)]],                   \
      constant uint3* offsets [[buffer(3)]],             \
      uint tid [[thread_position_in_grid]]);

REGISTER_FMAX_OP(float);
REGISTER_FMAX_OP(half);
REGISTER_FMIN_OP(float);
REGISTER_FMIN_OP(half);
REGISTER_COPYSIGN_OP(float);
REGISTER_COPYSIGN_OP(half);
#if __METAL_VERSION__ >= 310
REGISTER_FMAX_OP(bfloat);
REGISTER_FMIN_OP(bfloat);
REGISTER_COPYSIGN_OP(bfloat);
#endif
REGISTER_COPYSIGN_INTEGRAL_OP(int);
REGISTER_COPYSIGN_INTEGRAL_OP(long);
REGISTER_COPYSIGN_INTEGRAL_OP(short);
REGISTER_COPYSIGN_INTEGRAL_OP(char);
REGISTER_COPYSIGN_INTEGRAL_OP(uchar);
REGISTER_COPYSIGN_INTEGRAL_OP(bool);

template <typename T>
kernel void polar(
    constant void* abs_ [[buffer(0)]],
    constant void* angle_ [[buffer(1)]],
    device void* out_ [[buffer(2)]],
    constant uint3* offsets [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  device T* out = (device T*)((device uint8_t*)out_ + offsets[tid].x);
  constant T* angle = (constant T*)((constant uint8_t*)angle_ + offsets[tid].z);
  constant T* abs = (constant T*)((constant uint8_t*)abs_ + offsets[tid].y);
  out[0] = abs[0] * cos(angle[0]);
  out[1] = abs[0] * sin(angle[0]);
}

#define REGISTER_POLAR_OP(DTYPE)                                    \
  template [[host_name("polar_" #DTYPE)]] kernel void polar<DTYPE>( \
      constant void* abs,                                           \
      constant void* angle,                                         \
      device void* out,                                             \
      constant uint3* offsets,                                      \
      uint tid)

REGISTER_POLAR_OP(float);
REGISTER_POLAR_OP(half);

template <typename T>
kernel void complex_mul(
    constant void* input_ [[buffer(0)]],
    constant void* other_ [[buffer(1)]],
    device void* out_ [[buffer(2)]],
    constant uint3* offsets [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  device T* out = (device T*)((device uint8_t*)out_ + offsets[tid].x);
  constant T* input = (constant T*)((constant uint8_t*)input_ + offsets[tid].y);
  constant T* other = (constant T*)((constant uint8_t*)other_ + offsets[tid].z);
  out[0] = input[0] * other[0] - input[1] * other[1];
  out[1] = input[0] * other[1] + input[1] * other[0];
}

#define REGISTER_COMPLEX_MUL_OP(DTYPE)                      \
  template [[host_name("complex_mul_" #DTYPE)]] kernel void \
  complex_mul<DTYPE>(                                       \
      constant void* input,                                 \
      constant void* other,                                 \
      device void* out,                                     \
      constant uint3* offsets,                              \
      uint tid)

REGISTER_COMPLEX_MUL_OP(float);
REGISTER_COMPLEX_MUL_OP(half);

template <typename T, typename U>
kernel void nextafter_kernel(
    constant void* input_ [[buffer(0)]],
    constant void* other_ [[buffer(1)]],
    device void* out_ [[buffer(2)]],
    constant uint3* offsets [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  auto out = (device T*)((device uint8_t*)out_ + offsets[tid].x);
  auto input = *(constant T*)((constant uint8_t*)input_ + offsets[tid].y);
  auto other = *(constant T*)((constant uint8_t*)other_ + offsets[tid].z);
#if __METAL_VERSION__ >= 310
  *out = static_cast<T>(nextafter(input, other));
#else
  if (input == other) {
    *out = input;
  } else if (isnan(input) || isnan(other)) {
    *out = NAN;
  } else if (input == 0) {
    constexpr auto one = as_type<T>(static_cast<U>(1));
    *out = other > 0 ? one : -one;
  } else {
    U bits = as_type<U>(input);
    (input > 0) ^ (input > other) ? bits++ : bits--;
    *out = as_type<T>(bits);
  }
#endif
}

#define REGISTER_NEXTAFTER_OP(DTYPE, UTYPE)                      \
  template [[host_name("nextafter_kernel_" #DTYPE)]] kernel void \
  nextafter_kernel<DTYPE, UTYPE>(                                \
      constant void* input,                                      \
      constant void* other,                                      \
      device void* out,                                          \
      constant uint3* offsets,                                   \
      uint tid)

REGISTER_NEXTAFTER_OP(float, uint);
REGISTER_NEXTAFTER_OP(half, ushort);
#if __METAL_VERSION__ >= 310
REGISTER_NEXTAFTER_OP(bfloat, ushort);
#endif

template <typename T>
kernel void complex_kernel(
    constant void* real_ [[buffer(0)]],
    constant void* imag_ [[buffer(1)]],
    device void* out_ [[buffer(2)]],
    constant uint3* offsets [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  device T* out = (device T*)((device uint8_t*)out_ + offsets[tid].x);
  constant T* real = (constant T*)((constant uint8_t*)real_ + offsets[tid].y);
  constant T* imag = (constant T*)((constant uint8_t*)imag_ + offsets[tid].z);
  out[0] = real[0];
  out[1] = imag[0];
}

#define REGISTER_COMPLEX_OUT_OP(DTYPE)                         \
  template [[host_name("complex_kernel_" #DTYPE)]] kernel void \
  complex_kernel<DTYPE>(                                       \
      constant void* real,                                     \
      constant void* imag,                                     \
      device void* out,                                        \
      constant uint3* offsets,                                 \
      uint tid)

REGISTER_COMPLEX_OUT_OP(float);
REGISTER_COMPLEX_OUT_OP(half);
