#pragma once

#include <c10/macros/Macros.h>

namespace caffe2 {
class TypeMeta;
} // namespace caffe2

namespace c10 {
C10_API void set_default_dtype(caffe2::TypeMeta dtype);
C10_API const caffe2::TypeMeta& get_default_dtype();
} // namespace c10
