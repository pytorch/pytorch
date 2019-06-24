#pragma once
#include <ATen/TypeDefault.h>

namespace at {

struct CAFFE2_API CPUTypeDefault : public TypeDefault {
  CPUTypeDefault(TensorTypeId type_id, bool is_variable, bool is_undefined)
      : TypeDefault(type_id, is_variable, is_undefined) {}
};

} // namespace at
