#include <caffe2/core/context_base.h>

#include <c10/util/Logging.h>

namespace at {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_DEFINE_TYPED_REGISTRY(
    ContextRegistry,
    at::DeviceType,
    at::BaseContext,
    std::unique_ptr,
    at::Device);

} // namespace at

namespace caffe2 {

// TODO: rename context.h -> context_cpu.h & context_base.h -> context.h

} // namespace caffe2
