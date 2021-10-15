#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at {
namespace native {

enum class QUANTILE_INTERPOLATION_MODE : uint8_t {
  LINEAR,
  LOWER,
  HIGHER,
  MIDPOINT,
  NEAREST
};

using sort_fn = void(*)(Tensor& values, Tensor& indices, int64_t dim, bool descending, bool stable);
using topk_fn = void(*)(const Tensor&, const Tensor&, const Tensor&, int64_t, int64_t, bool, bool);

DECLARE_DISPATCH(sort_fn, sort_stub);
DECLARE_DISPATCH(topk_fn, topk_stub);

} // namespace native
} // namespace at
