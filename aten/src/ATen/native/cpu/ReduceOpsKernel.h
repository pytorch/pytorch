#pragma once
#include "CapabilityDispatch.h"
#include "Parallel.h"
#include "Vec256.h"
#include <stdexcept>

#include "ATen/ATen.h"

namespace at {
namespace native {

template <CPUCapability C> struct sumImplC {
  static void function(Tensor &result, const Tensor &self, size_t dim,
                       bool all);
};

template <CPUCapability C> struct prodImplC {
  static void function(Tensor &result, const Tensor &self, size_t dim,
                       bool all);
};

}
}
