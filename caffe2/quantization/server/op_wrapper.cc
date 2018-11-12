#include "op_wrapper.h"

namespace caffe2 {

bool ShouldFp32FallbackToNCHW(const OperatorDef& def) {
  if ((def.type() == "Conv" || def.type() == "Int8Conv" ||
       def.type() == "ConvRelu" || def.type() == "Int8ConvRelu") &&
      ArgumentHelper::GetSingleArgument<OperatorDef, std::string>(
          def, "order", "NCHW") == "NHWC") {
    auto kernels =
        ArgumentHelper::GetRepeatedArgument<OperatorDef, int>(def, "kernels");
    if (kernels.size() > 2) {
      return true;
    }
  }
  return false;
}

} // namespace caffe2
