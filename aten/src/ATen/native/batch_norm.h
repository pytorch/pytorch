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

} // namespace native

} // namespace at
