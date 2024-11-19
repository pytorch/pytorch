#pragma once

#include <ATen/native/DispatchStub.h>
#include <c10/util/Exception.h>
#include <c10/util/string_view.h>

namespace c10 {
class Scalar;
}

namespace at {
struct TensorIterator;
struct TensorIteratorBase;
class TensorBase;
}

namespace at::native {

// These constants control the approximation behavior of gelu function.
enum class GeluType {
  None,             // Baseline Gelu
  Tanh,             // Tahn Gelu Approximation
  END
};

inline GeluType get_gelutype_enum(const c10::string_view approximate) {
  if (approximate == "none") {
    return GeluType::None;
  } else if (approximate == "tanh") {
    return GeluType::Tanh;
  } else {
    TORCH_CHECK(false, "approximate argument must be either none or tanh.");
  }
}

inline std::string gelutype_to_string(const GeluType type) {
  switch(type) {
    case GeluType::None: return "none";
    case GeluType::Tanh: return "tanh";
    default: TORCH_CHECK(false, "unknown GELU type: ", static_cast<int>(type));
  }
}

using structured_activation_fn = void (*)(TensorIteratorBase&);
using structured_activation_backward_fn = void (*)(TensorIteratorBase&);

using activation_fn = void (*)(TensorIterator&);
using activation_backward_fn = void (*)(TensorIterator&);
using softplus_fn = void (*)(TensorIteratorBase&, const c10::Scalar&, const c10::Scalar&);
using softplus_backward_fn = void (*)(TensorIteratorBase&, const c10::Scalar&, const c10::Scalar&);
using threshold_fn = void (*)(TensorIteratorBase&, const c10::Scalar&, const c10::Scalar&);
using hardtanh_backward_fn = void (*)(TensorIterator&, const c10::Scalar&, const c10::Scalar&);
using hardsigmoid_fn = void(*)(TensorIteratorBase&);
using hardsigmoid_backward_fn = void(*)(TensorIteratorBase&);
using hardswish_fn = void(*)(TensorIterator&);
using hardswish_backward_fn = void(*)(TensorIterator&);
using shrink_fn = void (*)(TensorIteratorBase&, const c10::Scalar&);
using softshrink_fn = void (*)(TensorIteratorBase&, const c10::Scalar&);
using shrink_backward_fn = void (*)(TensorIteratorBase&, const c10::Scalar&);
using elu_fn = void (*)(TensorIteratorBase&, const c10::Scalar&, const c10::Scalar&, const c10::Scalar&);
using elu_backward_fn = void (*)(TensorIteratorBase&, const c10::Scalar&, const c10::Scalar&, const c10::Scalar&, bool);
using leaky_relu_fn = void (*)(TensorIteratorBase&, const c10::Scalar&);
using leaky_relu_backward_fn = void (*)(TensorIteratorBase&, const c10::Scalar&);
using log_sigmoid_cpu_fn = void (*)(TensorBase&, TensorBase&, const TensorBase&);
using gelu_fn = void (*)(TensorIteratorBase&, GeluType);
using gelu_backward_fn = void (*)(TensorIteratorBase&, GeluType);
using glu_jvp_fn = void (*)(TensorIteratorBase&);

DECLARE_DISPATCH(elu_fn, elu_stub)
DECLARE_DISPATCH(elu_backward_fn, elu_backward_stub)
DECLARE_DISPATCH(softplus_fn, softplus_stub)
DECLARE_DISPATCH(softplus_backward_fn, softplus_backward_stub)
DECLARE_DISPATCH(log_sigmoid_cpu_fn, log_sigmoid_cpu_stub)
DECLARE_DISPATCH(activation_backward_fn, log_sigmoid_backward_stub)
DECLARE_DISPATCH(threshold_fn, threshold_stub)
DECLARE_DISPATCH(gelu_fn, GeluKernel)
DECLARE_DISPATCH(gelu_backward_fn, GeluBackwardKernel)
DECLARE_DISPATCH(hardtanh_backward_fn, hardtanh_backward_stub)
DECLARE_DISPATCH(hardsigmoid_fn, hardsigmoid_stub)
DECLARE_DISPATCH(hardsigmoid_backward_fn, hardsigmoid_backward_stub)
DECLARE_DISPATCH(hardswish_fn, hardswish_stub)
DECLARE_DISPATCH(hardswish_backward_fn, hardswish_backward_stub)
DECLARE_DISPATCH(shrink_fn, hardshrink_stub)
DECLARE_DISPATCH(softshrink_fn, softshrink_stub)
DECLARE_DISPATCH(shrink_backward_fn, shrink_backward_stub)
DECLARE_DISPATCH(leaky_relu_fn, leaky_relu_stub)
DECLARE_DISPATCH(leaky_relu_backward_fn, leaky_relu_backward_stub)
DECLARE_DISPATCH(structured_activation_fn, glu_stub)
DECLARE_DISPATCH(activation_backward_fn, glu_backward_stub)
DECLARE_DISPATCH(glu_jvp_fn, glu_jvp_stub)
DECLARE_DISPATCH(structured_activation_fn, silu_stub)
DECLARE_DISPATCH(structured_activation_backward_fn, silu_backward_stub)
DECLARE_DISPATCH(structured_activation_fn, mish_stub)
DECLARE_DISPATCH(activation_backward_fn, mish_backward_stub)
DECLARE_DISPATCH(activation_fn, prelu_stub)
DECLARE_DISPATCH(activation_backward_fn, prelu_backward_stub)

} // namespace at::native
