#include <ATen/core/typeid.h>
#include <ATen/core/DefaultDtype.h>

namespace at {
static auto default_dtype = caffe2::TypeMeta::Make<float>();

void set_default_dtype(caffe2::TypeMeta dtype) {
  default_dtype = std::move(dtype);
}

const caffe2::TypeMeta& get_default_dtype() {
  return default_dtype;
}
} // namespace at
