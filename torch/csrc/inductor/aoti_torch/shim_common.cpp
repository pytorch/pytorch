#include <c10/core/DeviceType.h>
#include <c10/core/GradMode.h>
#include <c10/core/Layout.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/mkldnn_tensor.h>
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
#include <ATen/ops/_embedding_bag.h>
#include <ATen/ops/_fft_c2c.h>
#include <ATen/ops/_scaled_dot_product_efficient_attention.h>
#include <ATen/ops/_scaled_dot_product_flash_attention.h>
#include <ATen/ops/_scaled_mm.h>
#include <ATen/ops/addmm.h>
#include <ATen/ops/as_strided.h>
#include <ATen/ops/bmm.h>
#include <ATen/ops/convolution.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/fbgemm_linear_fp16_weight_fp32_activation.h>
#include <ATen/ops/fbgemm_pack_gemm_matrix_fp16.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/index_put.h>
#include <ATen/ops/mm.h>
#include <ATen/ops/nonzero.h>
#include <ATen/ops/scalar_tensor.h>
#include <ATen/ops/scatter.h>
#include <ATen/ops/scatter_reduce.h>
#include <ATen/ops/view_as_real_ops.h>
#include <ATen/ops/view_ops.h>

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

#define AOTI_TORCH_DTYPE_IMPL(dtype, stype) \
  int32_t aoti_torch_dtype_##dtype() {      \
    return (int32_t)c10::ScalarType::stype; \
  }

AOTI_TORCH_DTYPE_IMPL(float8_e5m2, Float8_e5m2)
AOTI_TORCH_DTYPE_IMPL(float8_e4m3fn, Float8_e4m3fn)
AOTI_TORCH_DTYPE_IMPL(float8_e5m2fnuz, Float8_e5m2fnuz)
AOTI_TORCH_DTYPE_IMPL(float8_e4m3fnuz, Float8_e4m3fnuz)
AOTI_TORCH_DTYPE_IMPL(bfloat16, BFloat16)
AOTI_TORCH_DTYPE_IMPL(float16, Half)
AOTI_TORCH_DTYPE_IMPL(float32, Float)
AOTI_TORCH_DTYPE_IMPL(float64, Double)
AOTI_TORCH_DTYPE_IMPL(uint8, Byte)
AOTI_TORCH_DTYPE_IMPL(uint16, UInt16)
AOTI_TORCH_DTYPE_IMPL(uint32, UInt32)
AOTI_TORCH_DTYPE_IMPL(uint64, UInt64)
AOTI_TORCH_DTYPE_IMPL(int8, Char)
AOTI_TORCH_DTYPE_IMPL(int16, Short)
AOTI_TORCH_DTYPE_IMPL(int32, Int)
AOTI_TORCH_DTYPE_IMPL(int64, Long)
AOTI_TORCH_DTYPE_IMPL(bool, Bool)
AOTI_TORCH_DTYPE_IMPL(complex32, ComplexHalf)
AOTI_TORCH_DTYPE_IMPL(complex64, ComplexFloat)
AOTI_TORCH_DTYPE_IMPL(complex128, ComplexDouble)
#undef AOTI_TORCH_DTYPE_IMPL

int32_t aoti_torch_layout_strided() {
  return (int32_t)at::kStrided;
}

int32_t aoti_torch_layout__mkldnn() {
  return (int32_t)at::kMkldnn;
}

#define AOTI_TORCH_ITEM_IMPL(dtype, ctype)                     \
  AOTITorchError aoti_torch_item_##dtype(                      \
      AtenTensorHandle tensor, ctype* ret_value) {             \
    AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({               \
      at::Tensor* t = tensor_handle_to_tensor_pointer(tensor); \
      *ret_value = t->item().to<ctype>();                      \
    });                                                        \
  }

AOTI_TORCH_ITEM_IMPL(float32, float)
AOTI_TORCH_ITEM_IMPL(float64, double)
AOTI_TORCH_ITEM_IMPL(uint8, uint8_t)
AOTI_TORCH_ITEM_IMPL(uint16, uint16_t)
AOTI_TORCH_ITEM_IMPL(uint32, uint32_t)
AOTI_TORCH_ITEM_IMPL(uint64, uint64_t)
AOTI_TORCH_ITEM_IMPL(int8, int8_t)
AOTI_TORCH_ITEM_IMPL(int16, int16_t)
AOTI_TORCH_ITEM_IMPL(int32, int32_t)
AOTI_TORCH_ITEM_IMPL(int64, int64_t)
AOTI_TORCH_ITEM_IMPL(bool, bool)
#undef AOTI_TORCH_ITEM_IMPL

#define AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(dtype, ctype, ttype)                  \
  AOTITorchError aoti_torch_scalar_to_tensor_##dtype(                          \
      ctype value, AtenTensorHandle* ret_new_tensor) {                         \
    AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({                               \
      *ret_new_tensor =                                                        \
          new_tensor_handle(at::scalar_tensor(value, c10::ScalarType::ttype)); \
    });                                                                        \
  }

AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(float32, float, Float)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(float64, double, Double)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(uint8, uint8_t, Byte)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(uint16, uint16_t, UInt16)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(uint32, uint32_t, UInt32)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(uint64, uint64_t, UInt64)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(int8, int8_t, Char)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(int16, int16_t, Short)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(int32, int32_t, Int)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(int64, int64_t, Long)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(bool, bool, Bool)
#undef AOTI_TORCH_SCALAR_TO_TENSOR_IMPL

bool aoti_torch_grad_mode_is_enabled() {
  return c10::GradMode::is_enabled();
}

void aoti_torch_grad_mode_set_enabled(bool enabled) {
  return c10::GradMode::set_enabled(enabled);
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
    if (t->is_mkldnn()) {
      *ret_data_ptr = data_ptr_from_mkldnn(t);
    } else {
      *ret_data_ptr = t->data_ptr();
    }
  });
}

AOTITorchError aoti_torch_get_storage_size(
    AtenTensorHandle tensor,
    int64_t* ret_size) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* t = tensor_handle_to_tensor_pointer(tensor);
    *ret_size = t->storage().nbytes();
  });
}

AOTITorchError aoti_torch_get_dim(AtenTensorHandle tensor, int64_t* ret_dim) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* t = tensor_handle_to_tensor_pointer(tensor);
    *ret_dim = t->dim();
  });
}

AOTITorchError aoti_torch_get_numel(
    AtenTensorHandle tensor,
    int64_t* ret_numel) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* t = tensor_handle_to_tensor_pointer(tensor);
    *ret_numel = t->numel();
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

AOTITorchError aoti_torch_get_size(
    AtenTensorHandle tensor,
    int64_t d,
    int64_t* ret_size) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* t = tensor_handle_to_tensor_pointer(tensor);
    *ret_size = t->size(d);
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

AOTITorchError aoti_torch_get_stride(
    AtenTensorHandle tensor,
    int64_t d,
    int64_t* ret_stride) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* t = tensor_handle_to_tensor_pointer(tensor);
    *ret_stride = t->stride(d);
  });
}

AOTITorchError aoti_torch_get_dtype(
    AtenTensorHandle tensor,
    int32_t* ret_dtype) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* t = tensor_handle_to_tensor_pointer(tensor);
    *ret_dtype = static_cast<int32_t>(t->scalar_type());
  });
}

AOTITorchError aoti_torch_get_device_type(
    AtenTensorHandle tensor,
    int32_t* ret_device_type) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* t = tensor_handle_to_tensor_pointer(tensor);
    *ret_device_type = static_cast<int32_t>(t->device().type());
  });
}

AOTITorchError aoti_torch_get_device_index(
    AtenTensorHandle tensor,
    int32_t* ret_device_index) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* t = tensor_handle_to_tensor_pointer(tensor);
    *ret_device_index = t->device().index();
  });
}

AOTITorchError aoti_torch_get_storage_offset(
    AtenTensorHandle tensor,
    int64_t* ret_storage_offset) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* t = tensor_handle_to_tensor_pointer(tensor);
    *ret_storage_offset = t->storage_offset();
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
    *ret_new_tensor = new_tensor_handle(torch::inductor::_reinterpret_tensor(
        *self_tensor, sizes, strides, offset_increment));
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
    if (c10::DeviceType(device_type) == c10::DeviceType::CPU) {
      *ret_new_tensor = new_tensor_handle(at::detail::empty_strided_cpu(
          sizes, strides, static_cast<c10::ScalarType>(dtype)));
    } else {
      c10::Device device = c10_device(device_type, device_index);
      c10::TensorOptions options = c10::TensorOptions().device(device).dtype(
          static_cast<c10::ScalarType>(dtype));
      *ret_new_tensor =
          new_tensor_handle(at::empty_strided(sizes, strides, options));
    }
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
    *ret_new_tensor = new_tensor_handle(
        // data == nullptr can happen for a 0-size tensor
        (data != nullptr) ? at::for_blob(data, sizes)
                                .strides(strides)
                                .storage_offset(storage_offset)
                                .options(options)
                                .make_tensor()
                          : at::empty_strided(sizes, strides, options));
  });
}

AOTITorchError aoti_torch_create_tensor_from_blob_v2(
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
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    if (layout == static_cast<int32_t>(at::kMkldnn)) {
      c10::IntArrayRef sizes(sizes_ptr, ndim);
      c10::IntArrayRef strides(strides_ptr, ndim);
      c10::Device device = c10_device(device_type, device_index);
      // get a mkldnn tensor wrapped by a torch Tensor(OpaqueTensorImpl),
      // which used by later mkldnn op.
      *ret_new_tensor = new_tensor_handle(mkldnn_tensor_from_data_ptr(
          data,
          sizes,
          static_cast<c10::ScalarType>(dtype),
          device,
          opaque_metadata,
          opaque_metadata_size));
    } else {
      aoti_torch_create_tensor_from_blob(
          data,
          ndim,
          sizes_ptr,
          strides_ptr,
          storage_offset,
          dtype,
          device_type,
          device_index,
          ret_new_tensor);
    }
  });
}

AOTI_TORCH_EXPORT AOTITorchError aoti_torch__embedding_bag(
    AtenTensorHandle weight,
    AtenTensorHandle indices,
    AtenTensorHandle offsets,
    int32_t scale_grad_by_freq,
    int32_t mode,
    int32_t sparse,
    AtenTensorHandle per_sample_weights, // optional argument
    int32_t include_last_offset,
    int32_t padding_idx,
    AtenTensorHandle* ret0, // returns new reference
    AtenTensorHandle* ret1, // returns new reference
    AtenTensorHandle* ret2, // returns new reference
    AtenTensorHandle* ret3 // returns new reference
) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto [r0, r1, r2, r3] = at::_embedding_bag(
        *tensor_handle_to_tensor_pointer(weight),
        *tensor_handle_to_tensor_pointer(indices),
        *tensor_handle_to_tensor_pointer(offsets),
        scale_grad_by_freq,
        mode,
        sparse,
        pointer_to_optional(
            tensor_handle_to_tensor_pointer(per_sample_weights)),
        include_last_offset,
        padding_idx);

    *ret0 = new_tensor_handle(std::move(r0));
    *ret1 = new_tensor_handle(std::move(r1));
    *ret2 = new_tensor_handle(std::move(r2));
    *ret3 = new_tensor_handle(std::move(r3));
  });
}

AOTI_TORCH_EXPORT AOTITorchError aoti_torch__fft_c2c(
    AtenTensorHandle self,
    const int64_t* dim_ptr,
    int64_t dim_size,
    int64_t normalization,
    int32_t forward,
    AtenTensorHandle* ret // returns new reference
) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto dim = c10::IntArrayRef(dim_ptr, dim_size);
    *ret = new_tensor_handle(at::_fft_c2c(
        *tensor_handle_to_tensor_pointer(self), dim, normalization, forward));
  });
}

AOTITorchError aoti_torch__scaled_dot_product_flash_attention_v2(
    AtenTensorHandle query,
    AtenTensorHandle key,
    AtenTensorHandle value,
    double dropout_p,
    int is_causal,
    int return_debug_mask,
    double* scale, // optional argument
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
    auto optional_scale = pointer_to_optional(scale);
    auto [r0, r1, r2, r3, r4, r5, r6, r7, r8] =
        at::_scaled_dot_product_flash_attention(
            *query_tensor,
            *key_tensor,
            *value_tensor,
            dropout_p,
            is_causal,
            return_debug_mask,
            optional_scale);

    *ret0 = new_tensor_handle(std::move(r0));
    *ret1 = new_tensor_handle(std::move(r1));
    // ret2 and ret3 may be null
    if (ret2) {
      *ret2 = new_tensor_handle(std::move(r2));
    }
    if (ret3) {
      *ret3 = new_tensor_handle(std::move(r3));
    }
    *ret4 = r4.expect_int();
    *ret5 = r5.expect_int();
    *ret6 = new_tensor_handle(std::move(r6));
    *ret7 = new_tensor_handle(std::move(r7));
    *ret8 = new_tensor_handle(std::move(r8));
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
  return aoti_torch__scaled_dot_product_flash_attention_v2(
      query,
      key,
      value,
      dropout_p,
      is_causal,
      return_debug_mask,
      &scale,
      ret0,
      ret1,
      ret2,
      ret3,
      ret4,
      ret5,
      ret6,
      ret7,
      ret8);
}

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch__scaled_dot_product_efficient_attention(
    AtenTensorHandle query,
    AtenTensorHandle key,
    AtenTensorHandle value,
    AtenTensorHandle attn_bias, // optional argument
    int compute_log_sumexp,
    double dropout_p,
    int is_causal,
    double* scale, // optional argument
    AtenTensorHandle* ret0, // returns new reference
    AtenTensorHandle* ret1, // returns new reference
    AtenTensorHandle* ret2, // returns new reference
    AtenTensorHandle* ret3 // returns new reference
) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* query_tensor = tensor_handle_to_tensor_pointer(query);
    at::Tensor* key_tensor = tensor_handle_to_tensor_pointer(key);
    at::Tensor* value_tensor = tensor_handle_to_tensor_pointer(value);
    auto optional_attn_bias =
        pointer_to_optional(tensor_handle_to_tensor_pointer(attn_bias));
    auto optional_scale = pointer_to_optional(scale);
    auto [r0, r1, r2, r3] = at::_scaled_dot_product_efficient_attention(
        *query_tensor,
        *key_tensor,
        *value_tensor,
        optional_attn_bias,
        compute_log_sumexp,
        dropout_p,
        is_causal,
        optional_scale);
    *ret0 = new_tensor_handle(std::move(r0));
    *ret1 = new_tensor_handle(std::move(r1));
    *ret2 = new_tensor_handle(std::move(r2));
    *ret3 = new_tensor_handle(std::move(r3));
  });
}

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_convolution(
    AtenTensorHandle input,
    AtenTensorHandle weight,
    AtenTensorHandle bias, // optional argument
    const int64_t* stride_ptr,
    int64_t stride_size,
    const int64_t* padding_ptr,
    int64_t padding_size,
    const int64_t* dilation_ptr,
    int64_t dilation_size,
    int transposed,
    const int64_t* output_padding_ptr,
    int64_t output_padding_size,
    int64_t groups,
    AtenTensorHandle* out // returns new reference
) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* input_tensor = tensor_handle_to_tensor_pointer(input);
    at::Tensor* weight_tensor = tensor_handle_to_tensor_pointer(weight);
    at::Tensor* bias_tensor = tensor_handle_to_tensor_pointer(bias);
    auto optional_bias = pointer_to_optional(bias_tensor);
    c10::IntArrayRef stride(stride_ptr, stride_size);
    c10::IntArrayRef padding(padding_ptr, padding_size);
    c10::IntArrayRef dilation(dilation_ptr, dilation_size);
    c10::IntArrayRef output_padding(output_padding_ptr, output_padding_size);

    *out = new_tensor_handle(at::convolution(
        *input_tensor,
        *weight_tensor,
        optional_bias,
        stride,
        padding,
        dilation,
        static_cast<bool>(transposed),
        output_padding,
        groups));
  });
}

AOTITorchError aoti_torch_new_uninitialized_tensor(AtenTensorHandle* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* out_tensor = new at::Tensor();
    *ret = tensor_pointer_to_tensor_handle(out_tensor);
  });
}

AOTITorchError aoti_torch__scaled_mm(
    AtenTensorHandle self,
    AtenTensorHandle mat2,
    AtenTensorHandle bias,
    int32_t* out_dtype,
    AtenTensorHandle scale_a,
    AtenTensorHandle scale_b,
    AtenTensorHandle scale_result,
    int8_t use_fast_accum,
    AtenTensorHandle* ret0,
    AtenTensorHandle* ret1) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* self_tensor = tensor_handle_to_tensor_pointer(self);
    at::Tensor* mat2_tensor = tensor_handle_to_tensor_pointer(mat2);
    at::Tensor* bias_tensor = tensor_handle_to_tensor_pointer(bias);
    at::Tensor* scale_a_tensor = tensor_handle_to_tensor_pointer(scale_a);
    at::Tensor* scale_b_tensor = tensor_handle_to_tensor_pointer(scale_b);
    at::Tensor* scale_result_tensor =
        tensor_handle_to_tensor_pointer(scale_result);
    auto [r0, r1] = at::_scaled_mm(
        *self_tensor,
        *mat2_tensor,
        pointer_to_optional(bias_tensor),
        pointer_to_optional<c10::ScalarType>(out_dtype),
        pointer_to_optional(scale_a_tensor),
        pointer_to_optional(scale_b_tensor),
        pointer_to_optional(scale_result_tensor),
        use_fast_accum);
    *ret0 = new_tensor_handle(std::move(r0));
    *ret1 = new_tensor_handle(std::move(r1));
  });
}

// TODO: implement a more efficient version instead of calling into aten
AOTITorchError aoti_torch_tensor_copy_(
    AtenTensorHandle src,
    AtenTensorHandle dst) {
  return aoti_torch_copy_(dst, src, /*non_blocking=*/0);
}

AOTITorchError aoti_torch_assign_tensors(
    AtenTensorHandle src,
    AtenTensorHandle dst) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* src_tensor = tensor_handle_to_tensor_pointer(src);
    at::Tensor* dst_tensor = tensor_handle_to_tensor_pointer(dst);
    *dst_tensor = *src_tensor;
  });
}

AOTITorchError aoti_torch_assign_tensors_out(
    AtenTensorHandle src,
    AtenTensorHandle* ret_dst) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* src_tensor_ptr = tensor_handle_to_tensor_pointer(src);
    at::Tensor dst_tensor = *src_tensor_ptr;
    *ret_dst = new_tensor_handle(std::move(dst_tensor));
  });
}

AOTITorchError aoti_torch_clone(AtenTensorHandle self, AtenTensorHandle* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* self_tensor = tensor_handle_to_tensor_pointer(self);
    *ret = new_tensor_handle(self_tensor->clone());
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

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_copy_(
    AtenTensorHandle self,
    AtenTensorHandle src,
    int32_t non_blocking) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    tensor_handle_to_tensor_pointer(self)->copy_(
        *tensor_handle_to_tensor_pointer(src), non_blocking);
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

AOTITorchError aoti_torch_cpu_wrapped_fbgemm_pack_gemm_matrix_fp16(
    AtenTensorHandle weight,
    AtenTensorHandle* out) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* weight_tensor = tensor_handle_to_tensor_pointer(weight);

    *out = new_tensor_handle(at::fbgemm_pack_gemm_matrix_fp16(*weight_tensor));
  });
}

AOTITorchError aoti_torch_cpu_wrapped_fbgemm_linear_fp16_weight(
    AtenTensorHandle input,
    AtenTensorHandle weight,
    AtenTensorHandle bias,
    int64_t out_channel,
    AtenTensorHandle* out) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* input_tensor = tensor_handle_to_tensor_pointer(input);
    at::Tensor* weight_tensor = tensor_handle_to_tensor_pointer(weight);
    at::Tensor* bias_tensor = tensor_handle_to_tensor_pointer(bias);

    *out = new_tensor_handle(at::fbgemm_linear_fp16_weight_fp32_activation(
        *input_tensor, *weight_tensor, *bias_tensor));
  });
}

AOTITorchError aoti_torch_nonzero(
    AtenTensorHandle self,
    AtenTensorHandle* out) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* self_tensor = tensor_handle_to_tensor_pointer(self);
    *out = new_tensor_handle(at::nonzero(*self_tensor));
  });
}

AOTITorchError aoti_torch_repeat_interleave_Tensor(
    AtenTensorHandle repeats,
    int64_t* output_size,
    AtenTensorHandle* out) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* repeats_tensor = tensor_handle_to_tensor_pointer(repeats);
    *out = new_tensor_handle(at::_ops::repeat_interleave_Tensor::call(
        *repeats_tensor, pointer_to_optional<c10::SymInt>(output_size)));
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

AOTITorchError aoti_torch_scatter_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    int64_t dim,
    AtenTensorHandle index,
    AtenTensorHandle src) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* out_tensor = tensor_handle_to_tensor_pointer(out);
    at::Tensor* self_tensor = tensor_handle_to_tensor_pointer(self);
    at::Tensor* index_tensor = tensor_handle_to_tensor_pointer(index);
    at::Tensor* src_tensor = tensor_handle_to_tensor_pointer(src);
    at::scatter_out(*out_tensor, *self_tensor, dim, *index_tensor, *src_tensor);
  });
}

AOTITorchError aoti_torch_scatter_reduce_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    int64_t dim,
    AtenTensorHandle index,
    AtenTensorHandle src,
    const char* reduce,
    int32_t include_self) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* out_tensor = tensor_handle_to_tensor_pointer(out);
    at::Tensor* self_tensor = tensor_handle_to_tensor_pointer(self);
    at::Tensor* index_tensor = tensor_handle_to_tensor_pointer(index);
    at::Tensor* src_tensor = tensor_handle_to_tensor_pointer(src);
    at::scatter_reduce_out(
        *out_tensor,
        *self_tensor,
        dim,
        *index_tensor,
        *src_tensor,
        reduce,
        (bool)include_self);
  });
}

AOTITorchError aoti_torch_index_put_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    const AtenTensorHandle* indices,
    const uint32_t num_indices,
    const AtenTensorHandle values,
    bool accumulate) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    c10::List<std::optional<at::Tensor>> indices_;
    indices_.reserve(num_indices);
    for (size_t i = 0; i < num_indices; i++) {
      indices_.emplace_back(
          pointer_to_optional(tensor_handle_to_tensor_pointer(indices[i])));
    }
    at::Tensor* out_tensor = tensor_handle_to_tensor_pointer(out);
    at::Tensor* self_tensor = tensor_handle_to_tensor_pointer(self);
    at::Tensor* values_tensor = tensor_handle_to_tensor_pointer(values);
    at::index_put_out(
        *out_tensor, *self_tensor, indices_, *values_tensor, accumulate);
  });
}

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_view_as_real(
    AtenTensorHandle self,
    AtenTensorHandle* ret // returns new reference
) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    *ret = new_tensor_handle(
        at::_ops::view_as_real::call(*tensor_handle_to_tensor_pointer(self)));
  });
}

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_view_dtype(
    AtenTensorHandle self,
    int32_t dtype,
    AtenTensorHandle* ret // returns new reference
) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* self_tensor = tensor_handle_to_tensor_pointer(self);
    *ret = new_tensor_handle(at::_ops::view_dtype::call(
        *self_tensor, static_cast<c10::ScalarType>(dtype)));
  });
}

AOTI_TORCH_EXPORT void aoti_torch_print_tensor_handle(
    AtenTensorHandle self,
    const char* msg) {
  at::Tensor* t = tensor_handle_to_tensor_pointer(self);
  std::cout << "[";
  if (msg) {
    std::cout << msg;
  }
  std::cout << "]:" << *t << "\n";
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

void aoti_torch_check(
    bool cond,
    const char* func,
    const char* file,
    uint32_t line,
    const char* msg) {
  if (C10_UNLIKELY_OR_CONST(!cond)) {
    ::c10::detail::torchCheckFail(func, file, line, msg);
  }
}

AOTITorchError aoti_torch__alloc_from_pool(
    AtenTensorHandle self,
    int64_t offset_bytes,
    int32_t dtype,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    AtenTensorHandle* ret_new_tensor) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* self_tensor = tensor_handle_to_tensor_pointer(self);
    c10::IntArrayRef sizes(sizes_ptr, ndim);
    c10::IntArrayRef strides(strides_ptr, ndim);
    *ret_new_tensor = new_tensor_handle(torch::inductor::_alloc_from_pool(
        *self_tensor,
        offset_bytes,
        static_cast<c10::ScalarType>(dtype),
        sizes,
        strides));
  });
}
