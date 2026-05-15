#include <ATen/native/mps/kernels/Activation.h>
#include <c10/metal/indexing.h>
#include <c10/metal/special_math.h>
#include <metal_stdlib>
using namespace metal;
using namespace c10::metal;

struct hardshrink_functor {
  template <typename T>
  inline T operator()(const T x, const T lambda) {
    return abs(float(x)) <= float(lambda) ? T(0) : x;
  }
};

struct softshrink_functor {
  template <typename T>
  inline T operator()(const T x, const T lambda) {
    if (x > lambda) {
      return x - lambda;
    } else if (x < -lambda) {
      return x + lambda;
    } else {
      return T(0);
    }
  }
};

struct shrink_backward_functor {
  template <typename T>
  inline T operator()(const T grad_output, const T x, const T lambda) {
    return abs(float(x)) <= float(lambda) ? T(0) : grad_output;
  }
};

REGISTER_UNARY_ALPHA_OP(hardshrink, float, float, float);
REGISTER_UNARY_ALPHA_OP(hardshrink, half, half, half);
REGISTER_UNARY_ALPHA_OP(hardshrink, bfloat, bfloat, bfloat);

REGISTER_UNARY_ALPHA_OP(softshrink, float, float, float);
REGISTER_UNARY_ALPHA_OP(softshrink, half, half, half);
REGISTER_UNARY_ALPHA_OP(softshrink, bfloat, bfloat, bfloat);

REGISTER_BINARY_ALPHA_OP(shrink_backward, float, float, float);
REGISTER_BINARY_ALPHA_OP(shrink_backward, half, half, half);
REGISTER_BINARY_ALPHA_OP(shrink_backward, bfloat, bfloat, bfloat);

struct relu_functor {
  template <typename T>
  inline T operator()(const T x) {
    return x < T(0) ? T(0) : x;
  }
};

REGISTER_UNARY_OP(relu, float, float);
REGISTER_UNARY_OP(relu, half, half);
REGISTER_UNARY_OP(relu, bfloat, bfloat);
REGISTER_UNARY_OP(relu, long, long);
REGISTER_UNARY_OP(relu, int, int);
REGISTER_UNARY_OP(relu, short, short);
REGISTER_UNARY_OP(relu, char, char);
REGISTER_UNARY_OP(relu, uchar, uchar);
REGISTER_UNARY_OP(relu, bool, bool);

struct hardsigmoid_functor {
  template <typename T>
  inline T operator()(const T x) {
    return static_cast<T>(min(max(x + 3.0f, .0f), 6.f) / 6.f);
  }
};

struct hardsigmoid_backward_functor {
  template <typename T>
  inline T operator()(const T grad_output, const T self) {
    constexpr auto one_sixth = 1.0f / 6.0f;
    return static_cast<T>(
        abs(float(self)) < 3.0f ? float(grad_output) * one_sixth : 0.0f);
  }
};

REGISTER_UNARY_OP(hardsigmoid, float, float);
REGISTER_UNARY_OP(hardsigmoid, half, half);
REGISTER_UNARY_OP(hardsigmoid, bfloat, bfloat);

REGISTER_BINARY_OP(hardsigmoid_backward, float, float);
REGISTER_BINARY_OP(hardsigmoid_backward, half, half);
REGISTER_BINARY_OP(hardsigmoid_backward, bfloat, bfloat);

struct hardswish_functor {
  template <typename T>
  inline T operator()(const T x) {
    return static_cast<T>(float(x) * min(max(float(x) + 3.0f, .0f), 6.f) / 6.f);
  }
};

struct hardswish_backward_functor {
  template <typename T>
  inline T operator()(const T grad_output, const T self) {
    constexpr T zero(0);
    constexpr T three(3);
    constexpr T neg_three(-3);

    if (self <= neg_three) {
      return zero;
    } else if (self >= three) {
      return grad_output;
    } else {
      return static_cast<T>(float(grad_output) * (float(self) / 3.0f + 0.5f));
    }
  }
};

REGISTER_UNARY_OP(hardswish, float, float);
REGISTER_UNARY_OP(hardswish, half, half);
REGISTER_UNARY_OP(hardswish, bfloat, bfloat);

REGISTER_BINARY_OP(hardswish_backward, float, float);
REGISTER_BINARY_OP(hardswish_backward, half, half);
REGISTER_BINARY_OP(hardswish_backward, bfloat, bfloat);

struct elu_functor {
  template <typename T>
  inline T operator()(const T self_, const ELUParams<T> params) {
    using op_T = opmath_t<T>;
    auto alpha = static_cast<op_T>(params.alpha);
    auto scale = static_cast<op_T>(params.scale);
    auto input_scale = static_cast<op_T>(params.input_scale);
    auto self = static_cast<op_T>(self_);
    auto neg_res = alpha * (::metal::precise::exp(self * input_scale) - 1);
    return static_cast<T>(scale * (self < 0 ? neg_res : self));
  }
};

struct elu_backward_functor {
  template <typename T>
  inline T operator()(
      const T grad_output_,
      const T self_,
      ELUBackwardParams<T> params) {
    using op_T = opmath_t<T>;
    auto alpha = static_cast<op_T>(params.alpha);
    auto scale = static_cast<op_T>(params.scale);
    auto input_scale = static_cast<op_T>(params.input_scale);
    auto grad_output = static_cast<op_T>(grad_output_);
    auto self = static_cast<op_T>(self_);

    if (params.is_result) {
      auto neg_coef = input_scale * (self + alpha * scale);
      return static_cast<T>(grad_output * (self <= 0 ? neg_coef : scale));
    } else {
      auto neg_coef = input_scale * alpha * scale *
          ::metal::precise::exp(self * input_scale);
      return static_cast<T>(grad_output * (self <= 0 ? neg_coef : scale));
    }
  }
};

#define REGISTER_ELU_OP(T)            \
  typedef ELUParams<T> ELUParams_##T; \
  REGISTER_UNARY_ALPHA_OP(elu, T, ELUParams_##T, T);

REGISTER_ELU_OP(float);
REGISTER_ELU_OP(half);
REGISTER_ELU_OP(bfloat);

#define REGISTER_ELU_BACKWARD_OP(T)                   \
  typedef ELUBackwardParams<T> ELUBackwardParams_##T; \
  REGISTER_BINARY_ALPHA_OP(elu_backward, T, ELUBackwardParams_##T, T);

REGISTER_ELU_BACKWARD_OP(float);
REGISTER_ELU_BACKWARD_OP(half);
REGISTER_ELU_BACKWARD_OP(bfloat);

struct leaky_relu_functor {
  template <typename T>
  inline T operator()(const T x, const T negative_slope) {
    return float(x) > 0.0f ? x
                           : static_cast<T>(float(x) * float(negative_slope));
  }
};

struct leaky_relu_backward_functor {
  template <typename T>
  inline T operator()(
      const T self,
      const T grad_output,
      const T negative_slope) {
    return float(self) > 0.0f
        ? grad_output
        : static_cast<T>(float(grad_output) * float(negative_slope));
  }
};

REGISTER_UNARY_ALPHA_OP(leaky_relu, float, float, float);
REGISTER_UNARY_ALPHA_OP(leaky_relu, half, half, half);
REGISTER_UNARY_ALPHA_OP(leaky_relu, bfloat, bfloat, bfloat);

REGISTER_BINARY_ALPHA_OP(leaky_relu_backward, float, float, float);
REGISTER_BINARY_ALPHA_OP(leaky_relu_backward, half, half, half);
REGISTER_BINARY_ALPHA_OP(leaky_relu_backward, bfloat, bfloat, bfloat);

struct silu_functor {
  template <typename T>
  inline T operator()(const T x) {
    float xf = float(x);
    return static_cast<T>(xf / (1.0f + ::metal::precise::exp(-xf)));
  }
};

REGISTER_UNARY_OP(silu, float, float);
REGISTER_UNARY_OP(silu, half, half);
REGISTER_UNARY_OP(silu, bfloat, bfloat);
REGISTER_UNARY_OP(silu, int, int);
REGISTER_UNARY_OP(silu, short, short);
REGISTER_UNARY_OP(silu, char, char);
REGISTER_UNARY_OP(silu, uchar, uchar);
REGISTER_UNARY_OP(silu, bool, bool);

struct silu_backward_functor {
  template <typename T>
  inline T operator()(const T grad_output, const T self) {
    float sf = float(self);
    float sig = 1.0f / (1.0f + ::metal::precise::exp(-sf));
    return static_cast<T>(float(grad_output) * sig * (1.0f + sf - sf * sig));
  }
};

REGISTER_BINARY_OP(silu_backward, float, float);
REGISTER_BINARY_OP(silu_backward, half, half);
REGISTER_BINARY_OP(silu_backward, bfloat, bfloat);

template <typename T>
static inline float gelu_dispatch_tanh(float x) {
  if IF_CONSTEXPR (::metal::is_same_v<T, float>) {
    return ::metal::tanh(x);
  } else {
    return ::metal::fast::tanh(x);
  }
}

struct gelu_functor {
  template <typename T>
  inline T operator()(const T x) {
    const float xf = float(x);
    return static_cast<T>(
        0.5f * xf * (1.0f + ::c10::metal::erf(xf * M_SQRT1_2_F)));
  }
};

struct gelu_tanh_functor {
  template <typename T>
  inline T operator()(const T x) {
    const float xf = float(x);
    constexpr float kBeta = M_SQRT2_F * M_2_SQRTPI_F * 0.5f;
    constexpr float kKappa = 0.044715f;
    const float inner = kBeta * (xf + kKappa * xf * xf * xf);
    return static_cast<T>(0.5f * xf * (1.0f + gelu_dispatch_tanh<T>(inner)));
  }
};

REGISTER_UNARY_OP(gelu, float, float);
REGISTER_UNARY_OP(gelu, half, half);
REGISTER_UNARY_OP(gelu, bfloat, bfloat);

REGISTER_UNARY_OP(gelu_tanh, float, float);
REGISTER_UNARY_OP(gelu_tanh, half, half);
REGISTER_UNARY_OP(gelu_tanh, bfloat, bfloat);

struct gelu_backward_functor {
  template <typename T>
  inline T operator()(const T grad, const T self) {
    const float xf = float(self);
    constexpr float kPdfCoeff = M_2_SQRTPI_F * M_SQRT1_2_F * 0.5f;
    const float cdf = 0.5f * (1.0f + ::c10::metal::erf(xf * M_SQRT1_2_F));
    const float pdf = kPdfCoeff * ::metal::exp(-0.5f * xf * xf);
    return static_cast<T>(float(grad) * (cdf + xf * pdf));
  }
};

struct gelu_tanh_backward_functor {
  template <typename T>
  inline T operator()(const T grad, const T self) {
    const float xf = float(self);
    constexpr float kBeta = M_SQRT2_F * M_2_SQRTPI_F * 0.5f;
    constexpr float kKappa = 0.044715f;
    const float x_sq = xf * xf;
    const float inner = kBeta * (xf + kKappa * xf * x_sq);
    const float th = gelu_dispatch_tanh<T>(inner);
    const float dth = 1.0f - th * th;
    const float dinner = kBeta * (1.0f + 3.0f * kKappa * x_sq);
    const float dgelu = 0.5f * (1.0f + th) + 0.5f * xf * dth * dinner;
    return static_cast<T>(float(grad) * dgelu);
  }
};

REGISTER_BINARY_OP(gelu_backward, float, float);
REGISTER_BINARY_OP(gelu_backward, half, half);
REGISTER_BINARY_OP(gelu_backward, bfloat, bfloat);

REGISTER_BINARY_OP(gelu_tanh_backward, float, float);
REGISTER_BINARY_OP(gelu_tanh_backward, half, half);
REGISTER_BINARY_OP(gelu_tanh_backward, bfloat, bfloat);
