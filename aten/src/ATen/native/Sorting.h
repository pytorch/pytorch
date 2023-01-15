#pragma once

#include <ATen/native/DispatchStub.h>
#include <cstdint>

namespace at {
class TensorBase;
}

namespace at {
namespace native {

enum class QUANTILE_INTERPOLATION_MODE : uint8_t {
  LINEAR,
  LOWER,
  HIGHER,
  MIDPOINT,
  NEAREST
};

using sort_fn = void(*)(const TensorBase&, const TensorBase&, const TensorBase&, int64_t, bool, bool);
using topk_fn = void(*)(const TensorBase&, const TensorBase&, const TensorBase&, int64_t, int64_t, bool, bool);

DECLARE_DISPATCH(sort_fn, sort_stub);
DECLARE_DISPATCH(topk_fn, topk_stub);

void _fill_indices(const TensorBase &indices, int64_t dim);

} // namespace native
} // namespace at
