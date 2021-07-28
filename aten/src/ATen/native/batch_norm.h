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

} // namespace native

} // namespace at
