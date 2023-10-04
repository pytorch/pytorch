#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/proxy_executor.h>
#include <torch/csrc/inductor/aoti_torch/tensor_converter.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <torch/csrc/inductor/inductor_ops.h>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else

#include <ATen/ops/_addmm_activation.h>
#include <ATen/ops/_scaled_dot_product_flash_attention.h>
#include <ATen/ops/addmm.h>
#include <ATen/ops/as_strided.h>
#include <ATen/ops/bmm.h>
#include <ATen/ops/convolution.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/mm.h>

#endif

using namespace torch::aot_inductor;

namespace {
static c10::Device c10_device(int32_t device_type, int32_t device_index) {
  if (device_type == aoti_torch_device_type_cpu()) {
    return c10::Device(static_cast<c10::DeviceType>(device_type));
  } else {
    return c10::Device(
        static_cast<c10::DeviceType>(device_type),
        static_cast<c10::DeviceIndex>(device_index));
  }
}
} // namespace

int32_t aoti_torch_device_type_cpu() {
  return (int32_t)c10::DeviceType::CPU;
}

int32_t aoti_torch_device_type_cuda() {
  return (int32_t)c10::DeviceType::CUDA;
}

int32_t aoti_torch_dtype_bfloat16() {
  return (int32_t)c10::ScalarType::BFloat16;
}

int32_t aoti_torch_dtype_float16() {
  return (int32_t)c10::ScalarType::Half;
}

int32_t aoti_torch_dtype_float32() {
  return (int32_t)c10::ScalarType::Float;
}

int32_t aoti_torch_dtype_float64() {
  return (int32_t)c10::ScalarType::Double;
}

int32_t aoti_torch_dtype_uint8() {
  return (int32_t)c10::ScalarType::Byte;
}

int32_t aoti_torch_dtype_int8() {
  return (int32_t)c10::ScalarType::Char;
}

int32_t aoti_torch_dtype_int16() {
  return (int32_t)c10::ScalarType::Short;
}

int32_t aoti_torch_dtype_int32() {
  return (int32_t)c10::ScalarType::Int;
}

int32_t aoti_torch_dtype_int64() {
  return (int32_t)c10::ScalarType::Long;
}

AOTITorchError aoti_torch_delete_tensor_object(AtenTensorHandle tensor) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* t = tensor_handle_to_tensor_pointer(tensor);
    delete t;
  });
}

AOTITorchError aoti_torch_get_data_ptr(
    AtenTensorHandle tensor,
    void** ret_data_ptr) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* t = tensor_handle_to_tensor_pointer(tensor);
    *ret_data_ptr = t->data_ptr();
  });
}

AOTITorchError aoti_torch_get_sizes(
    AtenTensorHandle tensor,
    int64_t** ret_sizes) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* t = tensor_handle_to_tensor_pointer(tensor);
    *ret_sizes = const_cast<int64_t*>(t->sizes().data());
  });
}

AOTITorchError aoti_torch_get_strides(
    AtenTensorHandle tensor,
    int64_t** ret_strides) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* t = tensor_handle_to_tensor_pointer(tensor);
    *ret_strides = const_cast<int64_t*>(t->strides().data());
  });
}

AOTITorchError aoti_torch__reinterpret_tensor(
    AtenTensorHandle self,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t offset_increment,
    AtenTensorHandle* ret_new_tensor) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* self_tensor = tensor_handle_to_tensor_pointer(self);
    c10::IntArrayRef sizes(sizes_ptr, ndim);
    c10::IntArrayRef strides(strides_ptr, ndim);
    at::Tensor* new_tensor =
        new at::Tensor(torch::inductor::_reinterpret_tensor(
            *self_tensor, sizes, strides, offset_increment));
    *ret_new_tensor = tensor_pointer_to_tensor_handle(new_tensor);
  });
}

// TODO: implement a more efficient version instead of calling into aten
AOTITorchError aoti_torch_empty_strided(
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    AtenTensorHandle* ret_new_tensor) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    c10::IntArrayRef sizes(sizes_ptr, ndim);
    c10::IntArrayRef strides(strides_ptr, ndim);
    c10::Device device = c10_device(device_type, device_index);
    c10::TensorOptions options = c10::TensorOptions().device(device).dtype(
        static_cast<c10::ScalarType>(dtype));
    at::Tensor* new_tensor =
        new at::Tensor(at::empty_strided(sizes, strides, options));
    *ret_new_tensor = tensor_pointer_to_tensor_handle(new_tensor);
  });
}

AOTITorchError aoti_torch_create_tensor_from_blob(
    void* data,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    AtenTensorHandle* ret_new_tensor) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    c10::IntArrayRef sizes(sizes_ptr, ndim);
    c10::IntArrayRef strides(strides_ptr, ndim);
    c10::Device device = c10_device(device_type, device_index);
    c10::TensorOptions options = c10::TensorOptions().device(device).dtype(
        static_cast<c10::ScalarType>(dtype));
    at::Tensor* new_tensor = new at::Tensor(at::for_blob(data, sizes)
                                                .strides(strides)
                                                .storage_offset(storage_offset)
                                                .options(options)
                                                .make_tensor());
    *ret_new_tensor = tensor_pointer_to_tensor_handle(new_tensor);
  });
}

AOTITorchError aoti_torch__scaled_dot_product_flash_attention(
    AtenTensorHandle query,
    AtenTensorHandle key,
    AtenTensorHandle value,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    double scale,
    AtenTensorHandle* ret0, // returns new reference
    AtenTensorHandle* ret1, // returns new reference
    AtenTensorHandle* ret2, // returns new reference
    AtenTensorHandle* ret3, // returns new reference
    int64_t* ret4,
    int64_t* ret5,
    AtenTensorHandle* ret6, // returns new reference
    AtenTensorHandle* ret7, // returns new reference
    AtenTensorHandle* ret8 // returns new reference
) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* query_tensor = tensor_handle_to_tensor_pointer(query);
    at::Tensor* key_tensor = tensor_handle_to_tensor_pointer(key);
    at::Tensor* value_tensor = tensor_handle_to_tensor_pointer(value);
    auto [r0, r1, r2, r3, r4, r5, r6, r7, r8] =
        at::_scaled_dot_product_flash_attention(
            *query_tensor,
            *key_tensor,
            *value_tensor,
            dropout_p,
            is_causal,
            return_debug_mask,
            scale);

    at::Tensor* ret0_tensor = new at::Tensor(std::move(r0));
    *ret0 = tensor_pointer_to_tensor_handle(ret0_tensor);
    at::Tensor* ret1_tensor = new at::Tensor(std::move(r1));
    *ret1 = tensor_pointer_to_tensor_handle(ret1_tensor);
    // ret2 and ret3 may be null
    if (ret2) {
      at::Tensor* ret2_tensor = new at::Tensor(std::move(r2));
      *ret2 = tensor_pointer_to_tensor_handle(ret2_tensor);
    }
    if (ret3) {
      at::Tensor* ret3_tensor = new at::Tensor(std::move(r3));
      *ret3 = tensor_pointer_to_tensor_handle(ret3_tensor);
    }
    *ret4 = r4;
    *ret5 = r5;
    at::Tensor* ret6_tensor = new at::Tensor(std::move(r6));
    *ret6 = tensor_pointer_to_tensor_handle(ret6_tensor);
    at::Tensor* ret7_tensor = new at::Tensor(std::move(r7));
    *ret7 = tensor_pointer_to_tensor_handle(ret7_tensor);
    at::Tensor* ret8_tensor = new at::Tensor(std::move(r8));
    *ret8 = tensor_pointer_to_tensor_handle(ret8_tensor);
  });
}

AOTITorchError aoti_torch_new_uninitialized_tensor(AtenTensorHandle* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* out_tensor = new at::Tensor();
    *ret = tensor_pointer_to_tensor_handle(out_tensor);
  });
}

// TODO: implement a more efficient version instead of calling into aten
AOTITorchError aoti_torch_tensor_copy_(
    AtenTensorHandle src,
    AtenTensorHandle dst) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* src_tensor = tensor_handle_to_tensor_pointer(src);
    at::Tensor* dst_tensor = tensor_handle_to_tensor_pointer(dst);
    dst_tensor->copy_(*src_tensor);
  });
}

// TODO: implement a more efficient version instead of calling into aten
AOTITorchError aoti_torch_addmm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat1,
    AtenTensorHandle mat2,
    float beta,
    float alpha) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* out_tensor = tensor_handle_to_tensor_pointer(out);
    at::Tensor* self_tensor = tensor_handle_to_tensor_pointer(self);
    at::Tensor* mat1_tensor = tensor_handle_to_tensor_pointer(mat1);
    at::Tensor* mat2_tensor = tensor_handle_to_tensor_pointer(mat2);
    at::addmm_out(
        *out_tensor, *self_tensor, *mat1_tensor, *mat2_tensor, beta, alpha);
  });
}

// TODO: implement a more efficient version instead of calling into aten
AOTITorchError aoti_torch_bmm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat2) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* out_tensor = tensor_handle_to_tensor_pointer(out);
    at::Tensor* self_tensor = tensor_handle_to_tensor_pointer(self);
    at::Tensor* mat2_tensor = tensor_handle_to_tensor_pointer(mat2);
    at::bmm_out(*out_tensor, *self_tensor, *mat2_tensor);
  });
}

// TODO: implement a more efficient version instead of calling into aten
AOTITorchError aoti_torch_mm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat2) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* out_tensor = tensor_handle_to_tensor_pointer(out);
    at::Tensor* self_tensor = tensor_handle_to_tensor_pointer(self);
    at::Tensor* mat2_tensor = tensor_handle_to_tensor_pointer(mat2);
    at::mm_out(*out_tensor, *self_tensor, *mat2_tensor);
  });
}

// Function to check existence of inf and NaN
AOTITorchError aoti_check_inf_and_nan(AtenTensorHandle tensor) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* check_tensor = tensor_handle_to_tensor_pointer(tensor);
    auto flattened = check_tensor->view({-1});

    for (int64_t i = 0; i < flattened.numel(); i++) {
      auto value = flattened[i].item<float>();
      if (std::isinf(value) || std::isnan(value)) {
        assert(false);
      }
    }
  });
}

// ProxyExecutor
AOTITorchError aoti_torch_proxy_executor_call_function(
    AOTIProxyExecutorHandle proxy_executor,
    int extern_node_index,
    int num_ints,
    int64_t* flatten_int_args,
    int num_tensors,
    AtenTensorHandle* flatten_tensor_args) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    ProxyExecutor* executor = reinterpret_cast<ProxyExecutor*>(proxy_executor);
    executor->call_function(
        extern_node_index,
        num_ints,
        flatten_int_args,
        num_tensors,
        flatten_tensor_args);
  });
}
