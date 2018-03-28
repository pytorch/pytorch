#pragma once
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <stdexcept>
#include "CapabilityDispatch.h"

namespace at { namespace native {

template <CPUCapability C>
struct sumImplC {
  static void
  function(Tensor& result, const Tensor& self, size_t dim, bool all);
};

template <CPUCapability C>
struct prodImplC {
  static void
  function(Tensor& result, const Tensor& self, size_t dim, bool all);
};

}} // namespace at::native
