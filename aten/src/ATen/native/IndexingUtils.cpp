#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/IndexingUtils.h>

namespace at::native {

bool canUse32BitIndexMath(const TensorBase& t, int64_t max_elem) {
  const auto strides = t.sym_strides();
  const auto sizes = t.sym_sizes();
  c10::SymInt offset = 0;

  // NOTE: Assumes all strides are positive, which is true for now
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  for (const auto d : c10::irange(t.dim())) {
    if (sizes[d] == 0) {
      // return numel < max_elem
      return 0 < max_elem;
    }
    // here sizes[d] >= 1
    offset += (sizes[d] - 1) * strides[d];
  }

  return offset < max_elem;
}

} // namespace at::native
