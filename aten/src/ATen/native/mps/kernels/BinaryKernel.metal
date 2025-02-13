#include <c10/metal/special_math.h>
#include <metal_stdlib>
using namespace metal;

struct fmax_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(::metal::fmax(a, b));
  }
};

struct fmin_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(::metal::fmin(a, b));
  }
};

struct copysign_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(::metal::copysign(a, b));
  }
};

struct zeta_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(c10::metal::zeta(a, b));
  }
};

template <typename T, typename F>
kernel void binary_indexing(
    constant void* input_ [[buffer(0)]],
    constant void* other_ [[buffer(1)]],
    device void* out_ [[buffer(2)]],
    constant uint3* offsets [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  device T* out = (device T*)((device uint8_t*)out_ + offsets[tid].x);
  constant T* input = (constant T*)((constant uint8_t*)input_ + offsets[tid].y);
  constant T* other = (constant T*)((constant uint8_t*)other_ + offsets[tid].z);
  F f;
  *out = f(*input, *other);
}

template <typename T, typename F>
kernel void binary_dense(
    constant T* input [[buffer(0)]],
    constant T* other [[buffer(1)]],
    device T* out [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  F f;
  out[tid] = f(input[tid], other[tid]);
}

#define REGISTER_BINARY_INDEXING_OP(NAME, DTYPE)             \
  template [[host_name(#NAME "_" #DTYPE)]] kernel void       \
  binary_indexing<DTYPE, NAME##_functor>(                    \
      constant void* input_,                                 \
      constant void* other_,                                 \
      device void* out_,                                     \
      constant uint3* offsets,                               \
      uint tid);                                             \
  template [[host_name(#NAME "_dense_" #DTYPE)]] kernel void \
  binary_dense<DTYPE, NAME##_functor>(                       \
      constant DTYPE * input_,                               \
      constant DTYPE * other_,                               \
      device DTYPE * out_,                                   \
      uint tid)

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

#define REGISTER_BINARY_OP(NAME, DTYPE)                             \
  template [[host_name(#NAME "_" #DTYPE)]] kernel void NAME<DTYPE>( \
      constant void* input_,                                        \
      constant void* other_,                                        \
      device void* out_,                                            \
      constant uint3* offsets,                                      \
      uint tid)

#define REGISTER_COPYSIGN_INTEGRAL_OP(DTYPE)             \
  template [[host_name("copysign_" #DTYPE)]] kernel void \
  copysign_integral<DTYPE>(                              \
      constant void* input_ [[buffer(0)]],               \
      constant void* other_ [[buffer(1)]],               \
      device void* out_ [[buffer(2)]],                   \
      constant uint3* offsets [[buffer(3)]],             \
      uint tid [[thread_position_in_grid]]);

REGISTER_BINARY_INDEXING_OP(fmax, float);
REGISTER_BINARY_INDEXING_OP(fmax, half);
REGISTER_BINARY_INDEXING_OP(fmin, float);
REGISTER_BINARY_INDEXING_OP(fmin, half);
REGISTER_BINARY_INDEXING_OP(copysign, float);
REGISTER_BINARY_INDEXING_OP(copysign, half);
REGISTER_BINARY_INDEXING_OP(zeta, float);
REGISTER_BINARY_INDEXING_OP(zeta, half);
#if __METAL_VERSION__ >= 310
REGISTER_BINARY_INDEXING_OP(fmax, bfloat);
REGISTER_BINARY_INDEXING_OP(fmin, bfloat);
REGISTER_BINARY_INDEXING_OP(copysign, bfloat);
REGISTER_BINARY_INDEXING_OP(zeta, bfloat);
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

REGISTER_BINARY_OP(polar, float);
REGISTER_BINARY_OP(polar, half);

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

REGISTER_BINARY_OP(complex_mul, float);
REGISTER_BINARY_OP(complex_mul, half);

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

REGISTER_BINARY_OP(complex_kernel, float);
REGISTER_BINARY_OP(complex_kernel, half);
