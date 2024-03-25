#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/TensorOperators.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/alpha_dropout_native.h>
#include <ATen/ops/dropout_native.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/feature_alpha_dropout_native.h>
#include <ATen/ops/feature_dropout_native.h>
#include <ATen/ops/native_dropout.h>
#include <ATen/ops/native_dropout_backward_native.h>
#include <ATen/ops/native_dropout_native.h>
#include <ATen/ops/ones_like.h>
#include <ATen/ops/zeros.h>
#endif

namespace at::native {

namespace {

template<bool inplace>
using Ctype = typename std::conditional<inplace, Tensor&, Tensor>::type;

Tensor make_feature_noise(const Tensor& input) {
  auto input_sizes = input.sym_sizes();
  TORCH_CHECK(input.dim() >= 2, "Feature dropout requires at least 2 dimensions in the input");
  c10::SymDimVector sizes;
  sizes.reserve(input.dim());
  sizes.push_back(input_sizes[0]);
  sizes.push_back(input_sizes[1]);
  for (C10_UNUSED const auto i : c10::irange(2, input.dim())) {
    sizes.push_back(1);
  }
  return input.new_empty_symint(sizes);
}

bool is_fused_kernel_acceptable(const Tensor& input, double p) {
  return (input.is_cuda() || input.is_xpu() || input.is_lazy() || input.is_privateuseone()) && p > 0 && p < 1 && input.sym_numel() > 0;
}

// NB: sure, we could have used different overloads here, but I would feel insecure
// knowing that this dispatch depends only on the constness of the references
template<bool inplace>
Tensor& multiply(Tensor& input, const Tensor& noise) {
  static_assert(inplace, "Wrong multiply overload triggered in Dropout.cpp");
  return input.mul_(noise);
}

template<bool inplace>
Tensor multiply(const Tensor& input, const Tensor& noise) {
  static_assert(!inplace, "Wrong multiply overload triggered in Dropout.cpp");
  return input.mul(noise);
}

template<bool feature_dropout, bool alpha_dropout, bool inplace, typename T>
Ctype<inplace> _dropout_impl(T& input, double p, bool train) {
  TORCH_CHECK(p >= 0 && p <= 1, "dropout probability has to be between 0 and 1, but got ", p);
  if (p == 0 || !train || input.sym_numel() == 0) {
    return input;
  }

  if (p == 1) {
    return multiply<inplace>(input, at::zeros({}, input.options()));
  }

  at::Tensor b; // used for alpha_dropout only
  auto noise = feature_dropout ? make_feature_noise(input) : at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  noise.bernoulli_(1 - p);
  if (alpha_dropout) {
    constexpr double alpha = 1.7580993408473766;
    double a = 1. / std::sqrt((alpha * alpha * p + 1) * (1 - p));
    b = noise.add(-1).mul_(alpha * a).add_(alpha * a * p);
    noise.mul_(a);
  } else {
    noise.div_(1 - p);
  }

  if (!alpha_dropout) {
    return multiply<inplace>(input, noise);
  } else {
    return multiply<inplace>(input, noise).add_(b);
  }
}

#define ALIAS_SPECIALIZATION(ALIAS_NAME, IS_FEATURE, IS_ALPHA)                      \
template <bool inplace, typename... Args>                                           \
Ctype<inplace> ALIAS_NAME(Args&&... args) {                                         \
  return _dropout_impl<IS_FEATURE, IS_ALPHA, inplace>(std::forward<Args>(args)...); \
}

ALIAS_SPECIALIZATION(_dropout,               false, false)
ALIAS_SPECIALIZATION(_feature_dropout,       true,  false)
ALIAS_SPECIALIZATION(_alpha_dropout,         false, true )
ALIAS_SPECIALIZATION(_feature_alpha_dropout, true,  true )

} // anonymous namespace

std::tuple<Tensor,Tensor>
native_dropout_cpu(const Tensor& input, double p, c10::optional<bool> train) {
  if (input.numel() == 0) {
    return std::make_tuple(input, at::empty_like(input, input.options()));
  }

  Tensor mask;
  Tensor output;

  if (!train.has_value() || *train) {
    double p1m = 1. - p;
    // Check for probability of zero to avoid divide by zero and NaN results
    double scale = p1m == 0 ? 0. : 1. / p1m;
    mask = at::empty_like(input, input.options().dtype(c10::CppTypeToScalarType<bool>::value));
    mask.bernoulli_(p1m);
    output = input.mul(mask).mul_(scale);
  } else {
    mask = at::ones_like(input, input.options().dtype(c10::CppTypeToScalarType<bool>::value));
    output = input.clone();
  }
  return std::make_tuple(output, mask);
}

Tensor native_dropout_backward(const Tensor& grad, const Tensor& mask, double scale) {
  Tensor result = grad * mask * scale;
  return result;
}

Tensor dropout(const Tensor& input, double p, bool train) {
  auto result = [&]() {
    NoNamesGuard guard;
    // TODO: we can remove this is_nested() code smell in the future
    //       if we find a way to support _dropout for nested tensor
    //       e.g. make it an op (at::_dropout) to use dispatcher?
    if (input.is_nested() || (train && is_fused_kernel_acceptable(input, p))) {
      return std::get<0>(at::native_dropout(input, p, train));
    }
    return _dropout<false>(input, p, train);
  }();
  namedinference::propagate_names(result, input);
  return result;
}

Tensor& dropout_(Tensor& input, double p, bool train) {
  return _dropout<true>(input, p, train);
}

Tensor feature_dropout(const Tensor& input, double p, bool train) {
  return _feature_dropout<false>(input, p, train);
}

Tensor& feature_dropout_(Tensor& input, double p, bool train) {
  return _feature_dropout<true>(input, p, train);
}

Tensor alpha_dropout(const Tensor& input, double p, bool train) {
  return _alpha_dropout<false>(input, p, train);
}

Tensor& alpha_dropout_(Tensor& input, double p, bool train) {
  return _alpha_dropout<true>(input, p, train);
}

Tensor feature_alpha_dropout(const Tensor& input, double p, bool train) {
  return _feature_alpha_dropout<false>(input, p, train);
}

Tensor& feature_alpha_dropout_(Tensor& input, double p, bool train) {
  return _feature_alpha_dropout<true>(input, p, train);
}

} // namespace at::native
