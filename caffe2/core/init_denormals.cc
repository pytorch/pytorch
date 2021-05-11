#if defined(__SSE3__)

#include <immintrin.h>

#include "caffe2/core/common.h"
#include "caffe2/core/init.h"

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_DEFINE_int(
    caffe2_ftz,
    false,
    "If true, turn on flushing denormals to zero (FTZ)");
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_DEFINE_int(
    caffe2_daz,
    false,
    "If true, turn on replacing denormals loaded from memory with zero (DAZ)");

namespace caffe2 {

bool Caffe2SetDenormals(int*, char***) {
  if (FLAGS_caffe2_ftz) {
    VLOG(1) << "Setting FTZ";
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  }
  if (FLAGS_caffe2_daz) {
    VLOG(1) << "Setting DAZ";
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  }
  return true;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CAFFE2_INIT_FUNCTION(
    Caffe2SetDenormals,
    &Caffe2SetDenormals,
    "Set denormal settings.");

} // namespace caffe2

#endif
