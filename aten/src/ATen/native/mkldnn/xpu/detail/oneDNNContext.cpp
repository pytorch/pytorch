#include <ATen/native/mkldnn/xpu/detail/Utils.h>
#include <ATen/native/mkldnn/xpu/detail/oneDNNContext.h>

/* *
 * Do NOT put any kernels or call any device binaries here!
 * Only maintain oneDNN runtime states in this file.
 * */
namespace at::native::onednn {

using namespace dnnl;

GpuEngineManager& GpuEngineManager::Instance() {
  static GpuEngineManager myInstance;
  return myInstance;
}

GpuStreamManager& GpuStreamManager::Instance() {
  static thread_local GpuStreamManager myInstance;
  return myInstance;
}

bool set_onednn_verbose(int level) {
  dnnl::status rs = dnnl::set_verbose(level);
  return rs == dnnl::status::success;
}

} // namespace at::native::onednn
