#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

using batch_norm_fn = void (*)(Tensor&, const Tensor&, const Tensor&,
    const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, bool, double);
using batch_norm_collect_stats_fn = void (*)(Tensor&, Tensor&, const Tensor&);
using batch_norm_backward_fn = void(*)(Tensor&, Tensor&, Tensor&, const Tensor&,
        const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, bool, double);

DECLARE_DISPATCH(batch_norm_fn, batch_norm_cpu_stub)
DECLARE_DISPATCH(batch_norm_collect_stats_fn, batch_norm_cpu_collect_stats_stub)
DECLARE_DISPATCH(batch_norm_backward_fn, batch_norm_cpu_backward_stub)

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
  if constexpr (std::is_const_v<scalar_t>) {
    return t.defined() ? t.contiguous().const_data_ptr<scalar_t>()
                      : nullptr;
  } else {
    return t.defined() ? t.contiguous().data_ptr<scalar_t>()
                      : nullptr;
  }
}

inline void check_batch_norm_inputs(
    const Tensor& input,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& running_mean_opt,
    const std::optional<Tensor>& running_var_opt,
    bool train) {
  TORCH_CHECK(input.dim() >= 2, "Expected input to have at least 2 dimensions, but got ", input.dim());
  const auto num_features = input.sym_sizes()[1];

  auto check_1d = [&](const std::optional<Tensor>& t_opt, const char* name) {
    if (t_opt.has_value() && t_opt->defined()) {
      const auto& t = *t_opt;
      TORCH_CHECK(
          t.dim() == 1,
          name, " should be a 1D tensor, but got ", t.dim(), "D");
      TORCH_CHECK(
          t.sym_numel() == num_features,
          name, " should contain ", num_features, " elements not ", t.sym_numel());
    }
  };

  check_1d(weight_opt, "weight");
  check_1d(bias_opt, "bias");
  check_1d(running_mean_opt, "running_mean");
  check_1d(running_var_opt, "running_var");

  const bool has_running_mean = running_mean_opt.has_value() && running_mean_opt->defined();
  const bool has_running_var = running_var_opt.has_value() && running_var_opt->defined();
  TORCH_CHECK_VALUE(
      has_running_mean == has_running_var,
      "running_mean and running_var must either both be None or neither be None");
  if (!train) {
    TORCH_CHECK_VALUE(
        has_running_mean,
        "running_mean must be defined in evaluation mode");
  }
}

} // namespace at::native
