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
      // multiplication to propagate Nan, Nan * 0 = Nan.
      return x * T(0);
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
    const auto r = (x + 3.0f) / 6.0f;
    return static_cast<T>(r > 1.0f ? 1.0f : (r < 0.0f ? 0.0f : r));
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

struct mish_functor {
  template <typename T>
  inline T operator()(const T x) {
    float xf = float(x);
    return static_cast<T>(
        xf *
        ::metal::precise::tanh(::c10::metal::log1p(::metal::precise::exp(xf))));
  }
};

REGISTER_UNARY_OP(mish, float, float);
REGISTER_UNARY_OP(mish, half, half);
REGISTER_UNARY_OP(mish, bfloat, bfloat);

struct mish_backward_functor {
  template <typename T>
  inline T operator()(const T grad_output, const T self) {
    float sf = float(self);
    float sig = 1.0f / (1.0f + ::metal::precise::exp(-sf));
    float tsp =
        ::metal::precise::tanh(::c10::metal::log1p(::metal::precise::exp(sf)));
    return static_cast<T>(
        float(grad_output) * (tsp + sf * sig * (1.0f - tsp * tsp)));
  }
};

REGISTER_BINARY_OP(mish_backward, float, float);
REGISTER_BINARY_OP(mish_backward, half, half);
REGISTER_BINARY_OP(mish_backward, bfloat, bfloat);

template <typename T>
static inline float gelu_dispatch_tanh(float x) {
  if IF_CONSTEXPR (::metal::is_same_v<T, float>) {
    return ::metal::tanh(x);
  } else {
    // Clamp to avoid fast::tanh's internals overflowing to NaN,
    // tanh is already saturated here.
    return ::metal::fast::tanh(::metal::clamp(x, -10.0f, 10.0f));
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

struct sigmoid_backward_functor {
  template <typename T, enable_if_t<is_scalar_floating_point_v<T>, bool> = true>
  inline T operator()(const T grad_output, const T output) {
    const float of = float(output);
    return static_cast<T>(float(grad_output) * (1.0f - of) * of);
  }
  template <typename T, enable_if_t<is_complex_v<T>, bool> = true>
  inline T operator()(const T grad_output, const T output) {
    return c10::metal::mul(
        grad_output,
        c10::metal::conj(c10::metal::mul(T(1, 0) - output, output)));
  }
};

REGISTER_BINARY_OP(sigmoid_backward, float, float);
REGISTER_BINARY_OP(sigmoid_backward, half, half);
REGISTER_BINARY_OP(sigmoid_backward, bfloat, bfloat);
REGISTER_BINARY_OP(sigmoid_backward, float2, float2);
REGISTER_BINARY_OP(sigmoid_backward, half2, half2);

struct glu_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    const float bf = float(b);
    return static_cast<T>(float(a) / (1.0f + ::metal::precise::exp(-bf)));
  }
};

REGISTER_BINARY_OP(glu, float, float);
REGISTER_BINARY_OP(glu, half, half);
REGISTER_BINARY_OP(glu, bfloat, bfloat);

// Dense fast path for a contiguous source: the tensor is collapsed around the
// split dim into [outer, 2L] (L = halved-dim * inner dims), so each outer row
// is a contiguous run whose two halves sit L elements apart. Dispatched as a 2D
// grid (x = position in the run, y = outer row), avoiding all per-element index
// math.
template <typename T>
kernel void glu_dense(
    device T* out [[buffer(0)]],
    constant T* input [[buffer(1)]],
    constant long& inner_half [[buffer(2)]],
    uint2 tpig [[thread_position_in_grid]]) {
  const long L = inner_half;
  const long s = static_cast<long>(tpig.y) * 2 * L + tpig.x;
  // Share glu_functor with the strided binary path so both stay in lockstep.
  out[static_cast<long>(tpig.y) * L + tpig.x] =
      glu_functor{}(input[s], input[s + L]);
}

#define REGISTER_GLU_DENSE_OP(DTYPE)                                        \
  template [[host_name("glu_dense_" #DTYPE)]] kernel void glu_dense<DTYPE>( \
      device DTYPE * out [[buffer(0)]],                                     \
      constant DTYPE * input [[buffer(1)]],                                 \
      constant long& inner_half [[buffer(2)]],                              \
      uint2 tpig [[thread_position_in_grid]])

REGISTER_GLU_DENSE_OP(float);
REGISTER_GLU_DENSE_OP(half);
REGISTER_GLU_DENSE_OP(bfloat);

// Shared glu backward math: from a (first half), b (second half), and the
// upstream grad gO, produce the gradients of both halves (.x for the first
// half, .y for the second). Used by both the strided and dense backward kernels
// so they stay in lockstep.
struct glu_backward_functor {
  template <typename T>
  inline vec2type_t<T> operator()(const T a, const T b, const T gO) {
    using op_T = opmath_t<T>;
    const op_T one = 1;
    const op_T sig = one / (one + ::metal::precise::exp(-op_T(b)));
    const op_T g = gO;
    return {
        static_cast<T>(sig * g),
        static_cast<T>((one - sig) * sig * g * op_T(a))};
  }
};

// Fused glu backward, mirroring the CUDA kernel: a single pass over the
// halved iteration shape that reads both input halves (the second via a fixed
// byte offset) and writes both grad halves, computing sigmoid internally.
template <typename T>
kernel void glu_backward(
    device void* grad_input [[buffer(0)]],
    constant void* input [[buffer(1)]],
    constant void* grad_output [[buffer(2)]],
    constant long* sizes [[buffer(3)]],
    constant long* grad_input_strides [[buffer(4)]],
    constant long* input_strides [[buffer(5)]],
    constant long* grad_output_strides [[buffer(6)]],
    constant long& grad_input_byte_offset [[buffer(7)]],
    constant long& input_byte_offset [[buffer(8)]],
    constant uint& ndim [[buffer(9)]],
    uint tid [[thread_position_in_grid]]) {
  long pos[max_ndim];
  pos_from_thread_index(static_cast<long>(tid), pos, sizes, ndim);
  const auto gI_offs = offset_from_coord(pos, grad_input_strides, ndim);
  const auto I_offs = offset_from_coord(pos, input_strides, ndim);
  const auto gO_offs = offset_from_coord(pos, grad_output_strides, ndim);

  const auto grads = glu_backward_functor{}(
      val_at_offs<T>(input, I_offs),
      val_at_offs<T>(input, I_offs + input_byte_offset),
      val_at_offs<T>(grad_output, gO_offs));
  ref_at_offs<T>(grad_input, gI_offs) = grads.x;
  ref_at_offs<T>(grad_input, gI_offs + grad_input_byte_offset) = grads.y;
}

#define REGISTER_GLU_BACKWARD_OP(DTYPE)                      \
  template [[host_name("glu_backward_" #DTYPE)]] kernel void \
  glu_backward<DTYPE>(                                       \
      device void* grad_input [[buffer(0)]],                 \
      constant void* input [[buffer(1)]],                    \
      constant void* grad_output [[buffer(2)]],              \
      constant long* sizes [[buffer(3)]],                    \
      constant long* grad_input_strides [[buffer(4)]],       \
      constant long* input_strides [[buffer(5)]],            \
      constant long* grad_output_strides [[buffer(6)]],      \
      constant long& grad_input_byte_offset [[buffer(7)]],   \
      constant long& input_byte_offset [[buffer(8)]],        \
      constant uint& ndim [[buffer(9)]],                     \
      uint tid [[thread_position_in_grid]])

REGISTER_GLU_BACKWARD_OP(float);
REGISTER_GLU_BACKWARD_OP(half);
REGISTER_GLU_BACKWARD_OP(bfloat);

// Dense backward fast path, mirroring glu_dense: contiguous input/grad_input
// collapsed to [outer, 2L], grad_output to [outer, L]. 2D grid (x = run pos,
// y = outer row).
template <typename T>
kernel void glu_backward_dense(
    device T* grad_input [[buffer(0)]],
    constant T* input [[buffer(1)]],
    constant T* grad_output [[buffer(2)]],
    constant long& inner_half [[buffer(3)]],
    uint2 tpig [[thread_position_in_grid]]) {
  const long L = inner_half;
  const long s = static_cast<long>(tpig.y) * 2 * L + tpig.x;
  const long g = static_cast<long>(tpig.y) * L + tpig.x;
  const auto grads =
      glu_backward_functor{}(input[s], input[s + L], grad_output[g]);
  grad_input[s] = grads.x;
  grad_input[s + L] = grads.y;
}

#define REGISTER_GLU_BACKWARD_DENSE_OP(DTYPE)                      \
  template [[host_name("glu_backward_dense_" #DTYPE)]] kernel void \
  glu_backward_dense<DTYPE>(                                       \
      device DTYPE * grad_input [[buffer(0)]],                     \
      constant DTYPE * input [[buffer(1)]],                        \
      constant DTYPE * grad_output [[buffer(2)]],                  \
      constant long& inner_half [[buffer(3)]],                     \
      uint2 tpig [[thread_position_in_grid]])

REGISTER_GLU_BACKWARD_DENSE_OP(float);
REGISTER_GLU_BACKWARD_DENSE_OP(half);
REGISTER_GLU_BACKWARD_DENSE_OP(bfloat);

struct log_sigmoid_forward_functor {
  template <typename T>
  inline T operator()(const T self) {
    const float x = float(self);
    const float m = ::metal::min(0.0f, x);
    const float z = ::metal::precise::exp(-::metal::abs(x));
    return static_cast<T>(m - log1p(z));
  }
};

REGISTER_UNARY_OP(log_sigmoid_forward, float, float);
REGISTER_UNARY_OP(log_sigmoid_forward, half, half);
REGISTER_UNARY_OP(log_sigmoid_forward, bfloat, bfloat);

struct log_sigmoid_backward_functor {
  template <typename T>
  inline T operator()(const T self, const T grad_output) {
    // d/dx log(sigmoid(x)) = 1 - sigmoid(x) = sigmoid(-x); compute it stably
    // via z = exp(-|x|), splitting on sign(x) to avoid overflow.
    const float in = float(self);
    const float z = ::metal::precise::exp(-::metal::abs(in));
    const float t = z / (1.0f + z);
    return static_cast<T>(float(grad_output) * (in < 0.0f ? 1.0f - t : t));
  }
};

REGISTER_BINARY_OP(log_sigmoid_backward, float, float);
REGISTER_BINARY_OP(log_sigmoid_backward, half, half);
REGISTER_BINARY_OP(log_sigmoid_backward, bfloat, bfloat);
