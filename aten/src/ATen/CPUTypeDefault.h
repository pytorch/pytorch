#pragma once
#include <ATen/TypeDefault.h>

namespace at {

struct CAFFE2_API CPUTypeDefault : public TypeDefault {
  CPUTypeDefault()
      : TypeDefault() {}
};

} // namespace at
