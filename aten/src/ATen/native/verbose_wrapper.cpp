#include <ATen/Config.h>
#include <c10/macros/Export.h>

#if AT_MKL_ENABLED()
#include <mkl.h>
#endif

#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/MKLDNNCommon.h>
#endif

namespace torch {
namespace verbose {

TORCH_API int _mkl_set_verbose(int enable) {
#if AT_MKL_ENABLED()
  return mkl_verbose(enable);
#else
  return 0;
#endif
}

TORCH_API int _mkldnn_set_verbose(int level) {
#if AT_MKLDNN_ENABLED()
  return at::native::set_verbose(level);
#else
  return 0;
#endif
}

} // namespace verbose
} // namespace torch
