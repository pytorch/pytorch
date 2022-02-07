#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

#include <tuple>

namespace at { namespace native {

using histogramdd_fn = void(*)(const Tensor&, const c10::optional<Tensor>&, bool, Tensor&, ITensorList);
using histogramdd_linear_fn = void(*)(const Tensor&, const c10::optional<Tensor>&, bool, Tensor&, const TensorList&, bool);

DECLARE_DISPATCH(histogramdd_fn, histogramdd_stub);
DECLARE_DISPATCH(histogramdd_linear_fn, histogramdd_linear_stub);

}} // namespace at::native
