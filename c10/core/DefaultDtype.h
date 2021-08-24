#pragma once

#include <c10/core/ScalarType.h>
#include <c10/core/Device.h>
#include <c10/macros/Macros.h>

namespace caffe2 {
class TypeMeta;
} // namespace caffe2

namespace c10 {
C10_API void set_default_dtype(caffe2::TypeMeta dtype);
C10_API const caffe2::TypeMeta get_default_dtype();
C10_API ScalarType get_default_dtype_as_scalartype();
C10_API const caffe2::TypeMeta get_default_complex_dtype();

// Please don't use these
C10_API void _set_default_device(Device device);
C10_API Device _get_default_device();
} // namespace c10
