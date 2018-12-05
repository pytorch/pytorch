#pragma once

#include <c10/macros/Macros.h>

namespace caffe2 {
class TypeMeta;
} // namespace caffe2

namespace at {
CAFFE2_API void set_default_dtype(caffe2::TypeMeta dtype);
CAFFE2_API const caffe2::TypeMeta& get_default_dtype();
} // namespace at
