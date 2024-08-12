#include <ATen/Config.h>

#if AT_MKL_ENABLED()
#include <mkl.h>
#endif

#if AT_ONEDNN_ENABLED()
#include <ATen/native/onednn/ONEDNNCommon.h>
#endif
#include <ATen/native/verbose_wrapper.h>

namespace torch::verbose {

int _mkl_set_verbose(int enable [[maybe_unused]]) {
#if AT_MKL_ENABLED()
  int ret = mkl_verbose(enable);

  // Return 0 when the mkl_verbose function fails to set verbose level.
  // Return 1 on success.
  return ret != -1;
#else
  // Return 0 since oneMKL is not enabled.
  return 0;
#endif
}

int _mkldnn_set_verbose(int level [[maybe_unused]]) {
#if AT_ONEDNN_ENABLED()
  return at::native::set_verbose(level);
#else
  return 0;
#endif
}

} // namespace torch::verbose
