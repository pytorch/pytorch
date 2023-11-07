#include <ATen/native/vulkan/ops/Common.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/repeat.h>
#endif

#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor tile(const Tensor& self, const IntArrayRef repeats) {
  // If self.size() > len(reps), reps is promoted to self.size() by pre-pending
  // 1’s to it to keep the same behaviour as `numpy.tile`.
  // Thus for a tensor of shape (2, 3, 4, 5), a dims of (2, 2) is treated
  // as (1, 1, 2, 2).
  const int64_t size_diff = self.dim() - static_cast<int64_t>(repeats.size());
  if (size_diff > 0) {
    std::vector<int64_t> new_repeats(size_diff, 1);
    for (const auto i : c10::irange(repeats.size())) {
      new_repeats.emplace_back(repeats[i]);
    }
    return self.repeat(IntArrayRef(new_repeats));
  }
  return self.repeat(repeats);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::tile"), TORCH_FN(tile));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
