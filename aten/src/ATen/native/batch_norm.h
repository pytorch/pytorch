#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at {

namespace native {

using batch_norm_fn = void (*)(Tensor&, const Tensor&, const Tensor&,
    const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, bool, double);
using batch_norm_collect_stats_fn = void (*)(Tensor&, Tensor&, const Tensor&);
using batch_norm_backward_fn = void(*)(Tensor&, Tensor&, Tensor&, const Tensor&,
        const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, bool, double);

DECLARE_DISPATCH(batch_norm_fn, batch_norm_cpu_stub);
DECLARE_DISPATCH(batch_norm_collect_stats_fn, batch_norm_cpu_collect_stats_stub);
DECLARE_DISPATCH(batch_norm_backward_fn, batch_norm_cpu_backward_stub);

// TensorAccessor when it is defined to work around undefined...
template <typename scalar_t>
static TensorAccessor<scalar_t, 1> conditional_accessor_1d(const Tensor& t) {
  if (! t.defined()) {
    return TensorAccessor<scalar_t, 1>(nullptr, nullptr, nullptr);
  }
  return t.accessor<scalar_t, 1>();
}

template <typename scalar_t>
static scalar_t* conditional_data_ptr(const Tensor& t) {
  return t.defined() ? t.contiguous().data_ptr<scalar_t>()
                     : nullptr;
}

inline ScalarType first_type() {
  return ScalarType::Undefined;
}

template <typename... Args>
inline ScalarType first_type(const Tensor& arg, const Args&... parameters) {
  return arg.defined() ? arg.scalar_type() : first_type(parameters...);
}

template <typename... Args>
inline bool is_mixed_type(const Tensor& input, const Args&... parameters) {
  const auto parameter_type = first_type(parameters...);
  return ((parameter_type != ScalarType::Undefined) &&
          (parameter_type != input.scalar_type()));
}

inline void checkMixedDataTypes(const Tensor& input, const Tensor& weight, const Tensor& bias,
    const Tensor& running_mean, const Tensor& running_var,
    const Tensor& save_mean, const Tensor& save_invstd) {

  // At the moment, the only allowed mixed dtype pattern: input(bfloat16) + weight/bias(float)
  TORCH_CHECK(input.scalar_type() == ScalarType::BFloat16,
      "BatchNorm (CPU) with mixed dtype: expect input to have scalar type of BFloat16");
  TORCH_CHECK(!weight.defined() || weight.scalar_type() == ScalarType::Float,
      "BatchNorm (CPU) with mixed dtype: expect weight either undefined or have scalar type of Float");
  TORCH_CHECK(!bias.defined() || bias.scalar_type() == ScalarType::Float,
      "BatchNorm (CPU) with mixed dtype: expect bias either undefined or have scalar type of Float");
  TORCH_CHECK(!running_mean.defined() || running_mean.scalar_type() == ScalarType::Float,
      "BatchNorm (CPU) with mixed dtype: expect running_mean either undefined or have scalar type of Float");
  TORCH_CHECK(!running_var.defined() || running_var.scalar_type() == ScalarType::Float,
      "BatchNorm (CPU) with mixed dtype: expect running_var either undefined or have scalar type of Float");
  TORCH_CHECK(!save_mean.defined() || save_mean.scalar_type() == ScalarType::Float,
      "BatchNorm (CPU) with mixed dtype: expect save_mean either undefined or have scalar type of Float");
  TORCH_CHECK(!save_invstd.defined() || save_invstd.scalar_type() == ScalarType::Float,
      "BatchNorm (CPU) with mixed dtype: expect save_invstd either undefined or have scalar type of Float");
}

// use float for bfloat16 accumulation
template <typename scalar_t> struct ParamAccType { using type = scalar_t; };
template <> struct ParamAccType<BFloat16> { using type = float; };

template <typename scalar_t>
using param_acc_t = typename ParamAccType<scalar_t>::type;

inline TensorOptions param_options(const Tensor& input) {
  if (input.scalar_type() == ScalarType::BFloat16) {
    return input.options().dtype(kFloat);
  } else {
    return input.options();
  }
}

} // namespace native

} // namespace at
