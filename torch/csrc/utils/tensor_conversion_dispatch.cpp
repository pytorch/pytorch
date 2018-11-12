#include <Python.h>

#include "tensor_conversion_dispatch.h"

#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/utils/cuda_lazy_init.h"

#include <ATen/DeviceGuard.h>

#include <cstddef>

namespace torch { namespace utils {

at::Tensor dispatch_type_conversion(
    const at::Tensor& self,
    const at::Type& type,
    c10::optional<int32_t> device_index,
    bool non_blocking) {
  if (type.is_cuda()) {
    torch::utils::cuda_lazy_init();
  }
  AutoNoGIL no_gil;

  // TODO: Make this less CUDA specific
  at::Device device = self.device();
  if (device_index && *device_index != -1) {
    device = at::Device(at::kCUDA, *device_index);
  }
  at::DeviceGuard device_guard(device);

  if (self.device().type() == type.device_type()) {
    switch (self.device().type()) {
      case at::DeviceType::CPU:
        // Do nothing, there is only one CPU "device"
        // TODO: Maybe this wouldn't be true with NUMA
        break;
      case at::DeviceType::CUDA:
        if (self.device().index() != at::current_device()) {
          // copy if the devices are different even if the types are the same
          return type.copy(self, non_blocking);
        }
        break;
      default:
        // This assert failed because you tried to use copy() on a non-CUDA
        // device.  We couldn't figure out if this would have resulted in
        // a cross-device copy, because at::current_device() only knows about
        // the current CUDA device.  Fix current_device to take a DeviceType
        // and provide information for things that are not CUDA too!
        AT_ASSERT(0);
    }
  }

  // Don't specialize cross-backend copies
  if (self.type().backend() != type.backend()) {
    return self.toType(type, non_blocking);
  }

  // Dispatch to specialized, traceable cast operators for the JIT. These
  // specialized ops are ATen native and thus have the tracing mechanisms auto-
  // generated, whereas the default case is not traceable since it requires a
  // Type as a parameter/attribute. TODO: support Types in the JIT and remove
  // this once we have that
  switch (type.scalarType()) {
#define DEFINE_CAST_DISPATCH(_1, n, _2)   \
  case at::ScalarType::n: {               \
    return at::_cast_##n(self, non_blocking); \
  } break;
    AT_FORALL_SCALAR_TYPES(DEFINE_CAST_DISPATCH)
#undef DEFINE_CAST_DISPATCH
    default: { return self.toType(type, non_blocking); } break;
  }
}
}} // namespace torch::utils
