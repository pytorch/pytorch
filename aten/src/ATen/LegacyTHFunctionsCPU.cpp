#include <ATen/LegacyTHFunctionsCPU.h>

#include <ATen/ATen.h>
#include <ATen/Utils.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/ExpandUtils.h>
#include <TH/TH.h>
#include <TH/THTensor.hpp>


namespace at {
namespace native {
namespace legacy {
namespace cpu {

namespace {
  ScalarType infer_scalar_type(const Tensor & t) {
    return t.scalar_type();
  }
  // NOLINTNEXTLINE(clang-diagnostic-unused-function)
  ScalarType infer_scalar_type(const TensorList & tl) {
    TORCH_CHECK(tl.size() > 0, "expected a non-empty list of Tensors");
    return tl[0].scalar_type();
  }

  TensorOptions options(ScalarType s) {
    return TensorOptions().dtype(s)
                          .device(DeviceType::CPU)
                          .layout(kStrided);
  }

  Allocator* allocator() {
    return getCPUAllocator();
  }
}

} // namespace th
} // namespace legacy
} // namespace native
} // namespace at
