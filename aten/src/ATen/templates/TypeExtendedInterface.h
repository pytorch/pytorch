#pragma once
#include <ATen/Type.h>

namespace at {

struct CAFFE2_API TypeExtendedInterface : public Type {
  explicit TypeExtendedInterface(TensorTypeId type_id, bool is_variable, bool is_undefined)
      : Type(type_id, is_variable, is_undefined) {}
  ${pure_virtual_extended_type_method_declarations}
};

} // namespace at
