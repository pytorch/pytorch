#include <torch/extension.h>

extern "C" const int64_t* CABI_get_tensor_size(int8_t* tensor_ptr) {
  auto variable = reinterpret_cast<THPVariable*>(tensor_ptr);
  return variable->cdata->sizes().data();
}

extern "C" const int64_t* CABI_get_tensor_stride(int8_t* tensor_ptr) {
  auto variable = reinterpret_cast<THPVariable*>(tensor_ptr);
  return variable->cdata->strides().data();
}

extern "C" int64_t CABI_get_tensor_storage_offset(int8_t* tensor_ptr) {
  auto variable = reinterpret_cast<THPVariable*>(tensor_ptr);
  return variable->cdata->storage_offset();
}
