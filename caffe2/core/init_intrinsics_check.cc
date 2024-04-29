#include "caffe2/core/common.h"
#include "caffe2/core/flags.h"
#include "caffe2/core/init.h"
#include "caffe2/core/logging.h"
#include "caffe2/utils/cpuid.h"

C10_DEFINE_bool(
    caffe2_quit_on_unsupported_cpu_feature,
    false,
    "If set, when Caffe2 is built with a CPU feature (like avx2) but the "
    "current CPU does not support it, quit early. If not set (by default), "
    "log this as an error message and continue execution.");

namespace caffe2 {

inline void QuitIfFeatureUnsupported(
    const bool cpu_has_feature, const string& feature) {
  VLOG(1) << "Caffe2 built with " << feature << ".";
  if (!cpu_has_feature) {
    string err_string =
        "The Caffe2 binary is compiled with CPU feature " + feature +
        ", but your CPU does not support it. This will lead to segfaults "
        "on your machine, such as SIGILL 'illegal instructions' on Linux. "
        "As a result Caffe2 will preemptively quit. Please install or "
        "build a Caffe2 binary with the feature turned off.";
    if (FLAGS_caffe2_quit_on_unsupported_cpu_feature) {
      LOG(FATAL) << err_string;
    } else {
      LOG(ERROR) << err_string;
    }
  }
}

static void WarnIfFeatureUnused(
    const bool cpu_has_feature, const string& feature) {
  VLOG(1) << "Caffe2 not built with " << feature << ".";
  if (cpu_has_feature) {
#ifdef CAFFE2_NO_CROSS_ARCH_WARNING
    // When cross-compiling single binary for multiple archs - turns off the
    // annoying warning
    VLOG(1)
#else
    LOG(ERROR)
#endif
        << "CPU feature " << feature
        << " is present on your machine, "
           "but the Caffe2 binary is not compiled with it. It means you "
           "may not get the full speed of your CPU.";
  }
}

bool Caffe2CheckIntrinsicsFeatures(int*, char***) {

#ifdef __AVX__
  QuitIfFeatureUnsupported(GetCpuId().avx(), "avx");
#else
  WarnIfFeatureUnused(GetCpuId().avx(), "avx");
#endif

#ifdef __AVX2__
  QuitIfFeatureUnsupported(GetCpuId().avx2(), "avx2");
#else
  WarnIfFeatureUnused(GetCpuId().avx2(), "avx2");
#endif

#ifdef __FMA__
  QuitIfFeatureUnsupported(GetCpuId().fma(), "fma");
#else
  WarnIfFeatureUnused(GetCpuId().fma(), "fma");
#endif

  return true;
}

REGISTER_CAFFE2_INIT_FUNCTION(
    Caffe2CheckIntrinsicsFeatures,
    &Caffe2CheckIntrinsicsFeatures,
    "Check intrinsics compatibility between the CPU feature and the binary.");

}  // namespace caffe2
