#include <c10/core/WrapDimMinimal.h>

namespace c10 {
namespace detail {

int64_t maybe_wrap_dim_slow(
    int64_t dim,
    int64_t dim_post_expr,
    bool wrap_scalar) {
  TORCH_CHECK_INDEX(
      dim_post_expr >= 0, "Rank cannot be negative but got ", dim_post_expr);

  if (dim_post_expr == 0) {
    TORCH_CHECK_INDEX(
        wrap_scalar,
        "Dimension specified as ",
        dim,
        " but tensor has no dimensions");
    return c10::maybe_wrap_dim(dim, /*dim_post_expr=*/1, /*wrap_scalar=*/false);
  }

  int64_t min = -dim_post_expr;
  int64_t max = dim_post_expr - 1;
  TORCH_CHECK_INDEX(
      min <= dim && dim <= max,
      "Dimension out of range (expected to be in range of [",
      min,
      ", ",
      max,
      "], but got ",
      dim,
      ")");

  TORCH_INTERNAL_ASSERT(
      false, "should never reach here as dim should be out-of-bounds");
}

} // namespace detail
} // namespace c10
