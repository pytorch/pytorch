#include <ATen/LegacyTHFunctions${Backend}.h>

// ${generated_comment}

#include <ATen/ATen.h>
#include <ATen/Utils.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/${Generator}.h>
#include <ATen/ExpandUtils.h>
#include <ATen/core/EnableNamedTensor.h>
${th_headers}
${extra_cuda_headers}

namespace at {
namespace native {
namespace legacy {
namespace ${namespace} {

namespace {
  ScalarType infer_scalar_type(const Tensor & t) {
    return t.scalar_type();
  }
  ScalarType infer_scalar_type(const TensorList & tl) {
    TORCH_CHECK(tl.size() > 0, "expected a non-empty list of Tensors");
    return tl[0].scalar_type();
  }

  TensorOptions options(ScalarType s) {
    return TensorOptions().dtype(s)
                          .device(DeviceType::${DeviceType})
                          .layout(kStrided)
                          .is_variable(false);
  }

  Allocator* allocator() {
    return ${allocator};
  }
}

${legacy_th_definitions}

} // namespace th
} // namespace legacy
} // namespace native
} // namespace at
