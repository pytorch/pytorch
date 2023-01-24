#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty_strided.h>
#endif

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

Tensor clone(
    const Tensor& src,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  auto memory_format = optional_memory_format.value_or(MemoryFormat::Preserve);
  TORCH_CHECK(
      (c10::MemoryFormat::Preserve == memory_format) ||
          (c10::MemoryFormat::Contiguous == memory_format),
      "Vulkan supports Preserve and Contiguous memory formats");

  Tensor self;
  if (memory_format == MemoryFormat::Preserve) {
    if (src.is_non_overlapping_and_dense()) {
      // Copy all strides, this is marginally faster than calling empty_like
      self = at::empty_strided(src.sizes(), src.strides(), src.options());
    } else {
      self = at::empty_like(src);
    }
  } else {
    self = at::empty_like(src, src.options(), memory_format);
  }

  self.copy_(src);
  return self;
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::clone"), TORCH_FN(clone));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
