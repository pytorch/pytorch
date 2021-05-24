#include <ATen/native/IndexingUtils.h>

namespace at { namespace native {

bool canUse32BitIndexMath(const Tensor& t, int64_t max_elem) {
  int64_t elements = t.numel();
  if (elements >= max_elem) {
    return false;
  }
  if (elements == 0) {
    return max_elem > 0;
  }

  int64_t offset = 0;
  int64_t linearId = elements - 1;

  // NOTE: Assumes all strides are positive, which is true for now
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  for (int i = t.dim() - 1; i >= 0; --i) {
    int64_t curDimIndex = linearId % t.size(i);
    int64_t curDimOffset = curDimIndex * t.stride(i);
    offset += curDimOffset;
    linearId /= t.size(i);
  }

  if (offset >= max_elem) {
    return false;
  }

  return true;
}

}} // namespace at::native
