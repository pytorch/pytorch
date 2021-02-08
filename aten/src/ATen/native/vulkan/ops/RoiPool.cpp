#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/UpSample.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

std::tuple<at::Tensor, at::Tensor> roi_pool(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width) {
  std::cout << "XXX roi_pool input:" << input << std::endl;
  auto t1 = at::empty({0}, input.options().dtype(at::kLong));
  auto t2 = t1;
  return std::make_tuple(t1, t2);
}

//#ifdef USE_VULKAN_API
TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::roi_pool(Tensor input, Tensor rois, float spatial_scale, int pooled_height, int pooled_width) -> (Tensor, Tensor)"));
}

TORCH_LIBRARY_IMPL(torchvision, Vulkan, m) {
  m.impl("torchvision::roi_pool", TORCH_FN(roi_pool));
}

//#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
