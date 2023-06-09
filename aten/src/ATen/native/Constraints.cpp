#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <c10/core/Scalar.h>

namespace at {
namespace native {

void _constrain_range_native_cpu(
  const Scalar& size,
  c10::optional<int64_t> min = c10::nullopt,
  c10::optional<int64_t> max = c10::nullopt
) {}

} // namespace meta
} // namespace at
