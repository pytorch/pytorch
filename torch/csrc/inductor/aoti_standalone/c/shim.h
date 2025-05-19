#pragma once

// This header mimics APIs in aoti_torch/c/shim.h in a standalone way

#include <torch/csrc/inductor/aoti_standalone/c/shim_device.h>
#include <torch/csrc/inductor/aoti_standalone/c/shim_dtype.h>
#include <torch/csrc/inductor/aoti_standalone/c/shim_layout.h>
#include <torch/csrc/inductor/aoti_standalone/utils.h>
#include <torch/standalone/slim_tensor/slim_tensor.h>

using AtenTensorOpaque = torch::standalone::SlimTensor;
using AtenTensorHandle = torch::standalone::SlimTensor*;

// AOTIProxyExecutorHandle isn't supported in standalone mode.
// Just defining it to void* to make the code compile
using AOTIProxyExecutorHandle = void*;

#ifdef __cplusplus
extern "C" {
#endif
inline bool aoti_torch_grad_mode_is_enabled() {
  return false;
}

inline void aoti_torch_grad_mode_set_enabled(bool enabled) {
  // do nothing
}

inline AOTITorchError aoti_torch_delete_tensor_object(AtenTensorHandle tensor) {
  delete tensor;
  return AOTI_TORCH_SUCCESS;
}

inline AOTITorchError aoti_torch_get_data_ptr(
    AtenTensorHandle tensor,
    void** ret_data_ptr) {
  *ret_data_ptr = tensor->data_ptr();
  return AOTI_TORCH_SUCCESS;
}

inline AOTITorchError aoti_torch_get_dtype(
    AtenTensorHandle tensor,
    int32_t* ret_dtype) {
  *ret_dtype = static_cast<int32_t>(tensor->dtype());
  return AOTI_TORCH_SUCCESS;
}

inline AOTITorchError aoti_torch_get_device_type(
    AtenTensorHandle tensor,
    int32_t* ret_device_type) {
  *ret_device_type = static_cast<int32_t>(tensor->device_type());
  return AOTI_TORCH_SUCCESS;
}

inline AOTITorchError aoti_torch_get_device_index(
    AtenTensorHandle tensor,
    int32_t* ret_device_index) {
  *ret_device_index = static_cast<uint8_t>(tensor->device_index());
  return AOTI_TORCH_SUCCESS;
}

inline AOTITorchError aoti_torch_get_sizes(
    AtenTensorHandle tensor,
    int64_t** ret_sizes) {
  *ret_sizes = (int64_t*)tensor->sizes().data();
  return AOTI_TORCH_SUCCESS;
}

inline AOTITorchError aoti_torch_get_size(
    AtenTensorHandle tensor,
    int64_t d,
    int64_t* ret_size) {
  *ret_size = tensor->size(d);
  return AOTI_TORCH_SUCCESS;
}

inline AOTITorchError aoti_torch_get_strides(
    AtenTensorHandle tensor,
    int64_t** ret_strides) {
  *ret_strides = (int64_t*)tensor->strides().data();
  return AOTI_TORCH_SUCCESS;
}

inline AOTITorchError aoti_torch_get_stride(
    AtenTensorHandle tensor,
    int64_t d,
    int64_t* ret_stride) {
  *ret_stride = tensor->stride(d);
  return AOTI_TORCH_SUCCESS;
}

inline AOTITorchError aoti_torch_get_storage_size(
    AtenTensorHandle tensor,
    int64_t* ret_size) {
  *ret_size = static_cast<int64_t>(tensor->nbytes());
  return AOTI_TORCH_SUCCESS;
}

inline AOTITorchError aoti_torch_get_storage_offset(
    AtenTensorHandle tensor,
    int64_t* ret_storage_offset) {
  *ret_storage_offset = tensor->storage_offset();
  return AOTI_TORCH_SUCCESS;
}

inline AOTITorchError aoti_torch_new_tensor_handle(
    AtenTensorHandle orig_handle,
    AtenTensorHandle* new_handle) {
  *new_handle = new torch::standalone::SlimTensor(*orig_handle);
  return AOTI_TORCH_SUCCESS;
}

inline AOTITorchError aoti_torch_create_tensor_from_blob_v2(
    void* data,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    AtenTensorHandle* ret_new_tensor,
    int32_t layout,
    const uint8_t* opaque_metadata,
    int64_t opaque_metadata_size) {
  torch::standalone::ArrayRef sizes(sizes_ptr, ndim);
  torch::standalone::ArrayRef strides(strides_ptr, ndim);
  *ret_new_tensor = new torch::standalone::SlimTensor(create_tensor_from_blob(
      data,
      sizes,
      strides,
      static_cast<c10::ScalarType>(dtype),
      {static_cast<c10::DeviceType>(device_type),
       static_cast<c10::DeviceIndex>(device_index)},
      storage_offset));
  return AOTI_TORCH_SUCCESS;
}

inline AOTITorchError aoti_torch_empty_strided(
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    AtenTensorHandle* ret_new_tensor) {
  torch::standalone::ArrayRef sizes(sizes_ptr, ndim);
  torch::standalone::ArrayRef strides(strides_ptr, ndim);
  *ret_new_tensor = new torch::standalone::SlimTensor(create_empty_tensor(
      sizes,
      strides,
      static_cast<c10::ScalarType>(dtype),
      {static_cast<c10::DeviceType>(device_type),
       static_cast<c10::DeviceIndex>(device_index)},
      0));
  return AOTI_TORCH_SUCCESS;
}

inline AOTITorchError aoti_torch__reinterpret_tensor(
    AtenTensorHandle self,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t offset_increment,
    AtenTensorHandle* ret_new_tensor) {
  torch::standalone::ArrayRef sizes(sizes_ptr, ndim);
  torch::standalone::ArrayRef strides(strides_ptr, ndim);
  *ret_new_tensor = new torch::standalone::SlimTensor(
      self->storage(),
      sizes,
      strides,
      self->dtype(),
      self->storage_offset() + offset_increment);
  return AOTI_TORCH_SUCCESS;
}

inline AOTITorchError aoti_torch_as_strided(
    AtenTensorHandle self,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    AtenTensorHandle* ret) {
  torch::standalone::ArrayRef sizes(sizes_ptr, self->dim());
  torch::standalone::ArrayRef strides(strides_ptr, self->dim());
  *ret = new torch::standalone::SlimTensor(
      self->storage(), sizes, strides, self->dtype(), self->storage_offset());
  return AOTI_TORCH_SUCCESS;
}

inline AOTITorchError aoti_torch_clone(
    AtenTensorHandle self,
    AtenTensorHandle* ret) {
  torch::standalone::SlimTensor tmp_tensor = create_empty_tensor(
      self->sizes(),
      self->strides(),
      self->dtype(),
      {self->device_type(), self->device_index()},
      0);
  tmp_tensor.copy_(*self);
  *ret = new torch::standalone::SlimTensor(tmp_tensor);
  return AOTI_TORCH_SUCCESS;
}

inline AOTITorchError aoti_torch_clone_preserve_strides(
    AtenTensorHandle self,
    AtenTensorHandle* ret) {
  int64_t needed_size = 1;
  for (size_t i = 0; i < self->dim(); i++) {
    if (self->size(i) == 0) {
      needed_size = 0;
      break;
    }
    needed_size += (self->size(i) - 1) * self->stride(i);
  }
  torch::standalone::SlimTensor tmp_tensor = *self;
  tmp_tensor.as_strided_({needed_size}, {1}, 0);
  aoti_torch_clone(&tmp_tensor, ret);
  (*ret)->as_strided_(self->sizes(), self->strides(), self->storage_offset());
  return AOTI_TORCH_SUCCESS;
}

#ifdef __cplusplus
} // extern "C"
#endif
