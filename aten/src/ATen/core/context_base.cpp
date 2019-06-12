#include <ATen/core/context_base.h>

namespace at {

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
