#include <c10/metal/indexing.h>
#include <c10/metal/special_math.h>
using namespace c10::metal;
using namespace metal;

DEFINE_UNARY_FLOATING_FUNCTOR(bessel_j0_forward);
DEFINE_UNARY_FLOATING_FUNCTOR(bessel_j1_forward);
DEFINE_UNARY_FLOATING_FUNCTOR(modified_bessel_i0_forward);
DEFINE_UNARY_FLOATING_FUNCTOR(modified_bessel_i1_forward);
DEFINE_UNARY_FLOATING_FUNCTOR(modified_bessel_k0_forward);
DEFINE_UNARY_FLOATING_FUNCTOR(modified_bessel_k1_forward);
DEFINE_UNARY_FLOATING_FUNCTOR(scaled_modified_bessel_k0_forward);
DEFINE_UNARY_FLOATING_FUNCTOR(scaled_modified_bessel_k1_forward);
DEFINE_UNARY_FLOATING_FUNCTOR(i0);
DEFINE_UNARY_FLOATING_FUNCTOR(i0e);
DEFINE_UNARY_FLOATING_FUNCTOR(i1);
DEFINE_UNARY_FLOATING_FUNCTOR(i1e);
DEFINE_UNARY_FLOATING_FUNCTOR(spherical_bessel_j0);

// TODO: Replaceme with DEFINE_UNARY_FLOATING_FUNCTOR
// But for some reason instantinating bessel_y[01] and entr on M1/M2 results in
// Failed to created pipeline state object, error: Error Domain=AGXMetalG14X
// Code=3 "Compiler encountered an internal error"
struct bessel_y0_forward_functor {
  template <typename T>
  inline enable_if_t<is_floating_point_v<T>, T> operator()(const T x) {
    return static_cast<T>(bessel_y0_forward(x));
  }
  template <typename T>
  inline enable_if_t<is_integral_v<T>, float> operator()(const T x) {
    return bessel_y0_forward(static_cast<float>(x));
  }
  inline float operator()(const bool x) {
    return x ? 0.08825694769620895 : -INFINITY;
  }
};

struct bessel_y1_forward_functor {
  template <typename T>
  inline enable_if_t<is_floating_point_v<T>, T> operator()(const T x) {
    return static_cast<T>(bessel_y1_forward(x));
  }
  template <typename T>
  inline enable_if_t<is_integral_v<T>, float> operator()(const T x) {
    return bessel_y1_forward(static_cast<float>(x));
  }
  inline float operator()(const bool x) {
    return x ? -0.7812128067016602 : -INFINITY;
  }
};

struct entr_functor {
  template <typename T>
  inline enable_if_t<is_floating_point_v<T>, T> operator()(const T x) {
    return static_cast<T>(entr(x));
  }
  template <typename T>
  inline enable_if_t<is_integral_v<T>, float> operator()(const T x) {
    return entr(static_cast<float>(x));
  }
  inline float operator()(const bool x) {
    return x ? -0.0 : 0.0;
  }
};

// log_ndtr: log of the cumulative distribution function of the standard normal
// log_ndtr(x) = log(0.5 * erfc(-x / sqrt(2)))
// For numerical stability when x < -1, use: log(erfcx(-t) / 2) - t^2
struct log_ndtr_functor {
  template <typename T>
  inline enable_if_t<is_floating_point_v<T>, T> operator()(const T x) {
    constexpr T SQRT1_2 = T(0.7071067811865475244008443621048490392); // 1/sqrt(2)
    T t = x * SQRT1_2;
    if (x < T(-1.0)) {
      return static_cast<T>(log(erfcx(-t) / 2) - t * t);
    } else {
      return static_cast<T>(log1p(-erfc(t) / 2));
    }
  }
  template <typename T>
  inline enable_if_t<is_integral_v<T>, float> operator()(const T x) {
    return (*this)(static_cast<float>(x));
  }
  inline float operator()(const bool x) {
    // log_ndtr(0) = log(0.5) = -0.693..., log_ndtr(1) = log(0.841...) = -0.173...
    return x ? -0.17275377902344985f : -0.6931471805599453f;
  }
};

// ndtri: inverse of the cumulative distribution function of the standard normal
// Implemented using erfinv: ndtri(p) = sqrt(2) * erfinv(2*p - 1)
struct ndtri_functor {
  template <typename T>
  inline enable_if_t<is_floating_point_v<T>, T> operator()(const T p) {
    // Use erfinv: ndtri(p) = sqrt(2) * erfinv(2*p - 1)
    constexpr T SQRT2 = T(1.4142135623730950488016887242096980785);
    return static_cast<T>(SQRT2 * erfinv(2 * p - 1));
  }
  template <typename T>
  inline enable_if_t<is_integral_v<T>, float> operator()(const T x) {
    return (*this)(static_cast<float>(x));
  }
  inline float operator()(const bool x) {
    // ndtri(0) = -inf, ndtri(1) = +inf
    return x ? INFINITY : -INFINITY;
  }
};

#define REGISTER_SPECIAL(DTI, DTO)                                \
  REGISTER_UNARY_OP(bessel_j0_forward, DTI, DTO);                 \
  REGISTER_UNARY_OP(bessel_j1_forward, DTI, DTO);                 \
  REGISTER_UNARY_OP(modified_bessel_i0_forward, DTI, DTO);        \
  REGISTER_UNARY_OP(modified_bessel_i1_forward, DTI, DTO);        \
  REGISTER_UNARY_OP(modified_bessel_k0_forward, DTI, DTO);        \
  REGISTER_UNARY_OP(modified_bessel_k1_forward, DTI, DTO);        \
  REGISTER_UNARY_OP(scaled_modified_bessel_k0_forward, DTI, DTO); \
  REGISTER_UNARY_OP(scaled_modified_bessel_k1_forward, DTI, DTO); \
  REGISTER_UNARY_OP(bessel_y0_forward, DTI, DTO);                 \
  REGISTER_UNARY_OP(bessel_y1_forward, DTI, DTO);                 \
  REGISTER_UNARY_OP(i0, DTI, DTO);                                \
  REGISTER_UNARY_OP(i0e, DTI, DTO);                               \
  REGISTER_UNARY_OP(i1, DTI, DTO);                                \
  REGISTER_UNARY_OP(i1e, DTI, DTO);                               \
  REGISTER_UNARY_OP(spherical_bessel_j0, DTI, DTO);               \
  REGISTER_UNARY_OP(entr, DTI, DTO);                              \
  REGISTER_UNARY_OP(log_ndtr, DTI, DTO);                          \
  REGISTER_UNARY_OP(ndtri, DTI, DTO)

REGISTER_SPECIAL(float, float);
REGISTER_SPECIAL(bool, float);
REGISTER_SPECIAL(uchar, float);
REGISTER_SPECIAL(char, float);
REGISTER_SPECIAL(short, float);
REGISTER_SPECIAL(int, float);
REGISTER_SPECIAL(long, float);
REGISTER_SPECIAL(half, half);
REGISTER_SPECIAL(bfloat, bfloat);
