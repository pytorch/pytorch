#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

#include <tuple>

namespace at { namespace native {

using histogramdd_fn = void(*)(const Tensor&, const c10::optional<Tensor>&, bool, Tensor&, const TensorList&);
using histogram_linear_fn = void(*)(const Tensor&, const c10::optional<Tensor>&, bool, Tensor&, const Tensor&, bool);

DECLARE_DISPATCH(histogramdd_fn, histogramdd_stub);
DECLARE_DISPATCH(histogram_linear_fn, histogram_linear_stub);

}} // namespace at::native
