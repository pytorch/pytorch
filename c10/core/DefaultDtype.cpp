#include <c10/util/typeid.h>
#include <c10/core/DefaultDtype.h>

namespace c10 {
static auto default_dtype = caffe2::TypeMeta::Make<float>();

void set_default_dtype(caffe2::TypeMeta dtype) {
  default_dtype = std::move(dtype);
}

const caffe2::TypeMeta& get_default_dtype() {
  return default_dtype;
}
} // namespace c10
