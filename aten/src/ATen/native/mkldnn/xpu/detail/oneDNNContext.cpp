#include <ATen/native/mkldnn/xpu/detail/oneDNNContext.h>
#include <ATen/native/mkldnn/xpu/detail/Utils.h>

/* XXX WARNING:
 * Do NOT put any kernels or call any device binaries here!
 * Only maintain oneDNN runtime states in this file.
 * */
namespace at::native{
namespace xpu {
namespace onednn {

using namespace dnnl;

GpuEngineManager& GpuEngineManager::Instance() {
  static GpuEngineManager myInstance;
  return myInstance;
}

GpuStreamManager& GpuStreamManager::Instance() {
  static thread_local GpuStreamManager myInstance;
  return myInstance;
}

} // namespace onednn
} // namespace xpu
} // namespace at::native