#include "caffe2/core/common.h"

namespace caffe2 {

const std::map<string, string>& GetBuildOptions() {
#ifndef CAFFE2_BUILD_STRINGS
#define CAFFE2_BUILD_STRINGS {}
#endif
  static const std::map<string, string> kMap = CAFFE2_BUILD_STRINGS;
  return kMap;
}

} // namespace caffe2
