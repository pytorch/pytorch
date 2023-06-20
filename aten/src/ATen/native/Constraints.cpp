#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <c10/core/Scalar.h>
#include <c10/util/Optional.h>

namespace at {
namespace native {

void sym_constrain_range_cpu(
    const Scalar& size,
    c10::optional<int64_t> min = c10::nullopt,
    c10::optional<int64_t> max = c10::nullopt) {}

Tensor functional_sym_constrain_range_cpu(
    const Scalar& size,
    c10::optional<int64_t> min = c10::nullopt,
    c10::optional<int64_t> max = c10::nullopt,
    const c10::optional<Tensor>& dep_token = c10::nullopt) {
  return c10::value_or_else(dep_token, [] {return at::empty({});});
}

} // namespace native
} // namespace at
