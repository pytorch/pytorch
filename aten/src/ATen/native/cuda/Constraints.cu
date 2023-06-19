#define TORCH_ASSERT_NO_OPERATORS

#include <c10/core/Scalar.h>
#include <c10/util/Optional.h>

namespace at {
namespace native {

void sym_constrain_range_cuda(
    const Scalar& size,
    c10::optional<int64_t> min = c10::nullopt,
    c10::optional<int64_t> max = c10::nullopt) {}

} // namespace native
} // namespace at
