#include "caffe2/utils/cpuid.h"

namespace caffe2 {

const CpuId& GetCpuId() {
  static CpuId cpuid_singleton;
  return cpuid_singleton;
}

} // namespace caffe2
