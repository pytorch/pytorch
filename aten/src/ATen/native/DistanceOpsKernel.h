#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at { namespace native {

void pdist_kernel(Tensor& result, const Tensor& self, double p);

void pdist_backward_kernel(Tensor& result, const Tensor& grad, const Tensor& self, const double p, const Tensor& dist);

}} // namespace at::native
