#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <c10/core/Scalar.h>
#include <c10/util/Optional.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/sym_constrain_range_native.h>
#endif

namespace at {
namespace native {

void sym_constrain_range_cpu(
    const Scalar& size,
    c10::optional<int64_t> min,
    c10::optional<int64_t> max) {}

} // namespace native
} // namespace at
