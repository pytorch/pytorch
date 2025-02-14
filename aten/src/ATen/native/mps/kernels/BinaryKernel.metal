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
  inline enable_if_t<is_floating_point_v<T>, T> operator()(
      const T a,
      const T b) {
    return static_cast<T>(::metal::copysign(a, b));
  }
  template <typename T>
  inline enable_if_t<!is_floating_point_v<T>, float> operator()(
      const T a,
      const T b) {
    return ::metal::copysign(static_cast<float>(a), static_cast<float>(b));
  }
};

struct zeta_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(c10::metal::zeta(a, b));
  }
};

struct nextafter_functor {
#if __METAL_VERSION__ < 310
  template <typename U>
  struct bit_type {};
  template <>
  struct bit_type<float> {
    using type = int;
  };
  template <>
  struct bit_type<half> {
    using type = short;
  };
#endif
  template <typename T>
  inline T operator()(const T a, const T b) {
#if __METAL_VERSION__ >= 310
    return static_cast<T>(::metal::nextafter(a, b));
#else
    using U = typename bit_type<T>::type;
    if (a == b) {
      return a;
    }
    if (::metal::isunordered(a, b)) {
      return NAN;
    }
    if (a == 0) {
      constexpr auto eps = as_type<T>(static_cast<U>(1));
      return b > 0 ? eps : -eps;
    }
    auto bits = as_type<U>(a);
    (a > 0) ^ (a > b) ? bits++ : bits--;
    return as_type<T>(bits);
#endif
  }
};

// Future BinaryTensorIterator
template <typename T, typename F>
using result_of = decltype(::metal::declval<F>()(
    ::metal::declval<T>(),
    ::metal::declval<T>()));

template <typename T, typename F>
kernel void binary_indexing(
    constant void* input_ [[buffer(0)]],
    constant void* other_ [[buffer(1)]],
    device void* out_ [[buffer(2)]],
    constant uint3* offsets [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  auto out = (device result_of<T, F>*)((device uint8_t*)out_ + offsets[tid].x);
  auto input = (constant T*)((constant uint8_t*)input_ + offsets[tid].y);
  auto other = (constant T*)((constant uint8_t*)other_ + offsets[tid].z);
  F f;
  *out = f(*input, *other);
}

template <typename T, typename F>
kernel void binary_dense(
    constant T* input [[buffer(0)]],
    constant T* other [[buffer(1)]],
    device result_of<T, F>* out [[buffer(2)]],
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
      device result_of<DTYPE, NAME##_functor> * out_,        \
      uint tid)

#define REGISTER_BINARY_OP(NAME, DTYPE)                             \
  template [[host_name(#NAME "_" #DTYPE)]] kernel void NAME<DTYPE>( \
      constant void* input_,                                        \
      constant void* other_,                                        \
      device void* out_,                                            \
      constant uint3* offsets,                                      \
      uint tid)

REGISTER_BINARY_INDEXING_OP(copysign, long);
REGISTER_BINARY_INDEXING_OP(copysign, int);
REGISTER_BINARY_INDEXING_OP(copysign, float);
REGISTER_BINARY_INDEXING_OP(copysign, half);
REGISTER_BINARY_INDEXING_OP(copysign, short);
REGISTER_BINARY_INDEXING_OP(copysign, uchar);
REGISTER_BINARY_INDEXING_OP(copysign, char);
REGISTER_BINARY_INDEXING_OP(copysign, bool);
REGISTER_BINARY_INDEXING_OP(fmax, float);
REGISTER_BINARY_INDEXING_OP(fmax, half);
REGISTER_BINARY_INDEXING_OP(fmin, float);
REGISTER_BINARY_INDEXING_OP(fmin, half);
REGISTER_BINARY_INDEXING_OP(nextafter, float);
REGISTER_BINARY_INDEXING_OP(nextafter, half);
REGISTER_BINARY_INDEXING_OP(zeta, float);
REGISTER_BINARY_INDEXING_OP(zeta, half);

#if __METAL_VERSION__ >= 310
REGISTER_BINARY_INDEXING_OP(copysign, bfloat);
REGISTER_BINARY_INDEXING_OP(fmax, bfloat);
REGISTER_BINARY_INDEXING_OP(fmin, bfloat);
REGISTER_BINARY_INDEXING_OP(nextafter, bfloat);
REGISTER_BINARY_INDEXING_OP(zeta, bfloat);
#endif

// Complex binary functions
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
