#ifndef AOTI_TORCH_SHIM
#define AOTI_TORCH_SHIM

#include <stddef.h>
#include <stdint.h>

// This header defines a stable C API for certain ATen functionality in
// libtorch. The AOTInductor compiled model.so will only refer to this header
// instead of other headers from aten/c10, which means it will NOT be able to
// directly use any data structures or call functions from libtorch.
//
// What problems are we trying to solve here?  Direct use of aten/c10 APIs
// means use of C++ APIs on a library that doesn't have any ABI compatibility
// guarantees.  However, we want model.so to remain usable across updates
// to the PyTorch C++ libraries, which requires a stable ABI.  By introducing
// a C shim layer, we can minimize the surface that will cause breakage. The
// corresponding software stack can be illustrated as follows:
//
// |--------------------------------|
// |     inference service code     |
// |--------------------------------|
// |           model.so             |
// |--------------|-----------------|
// |           <c shim>             |
// |          libtorch.so           |
// |--------------------------------|
//
// The general guidelines for the C API:
//
//  - No exceptions, return an explicit error code to be checked at call site
//  - Only pointers (AtenTensorHandle counts), integers and floats in headers
//
// If you want to make changes to this header, you MUST MAINTAIN ABI
// compatibility.  Typically, this means you will have to add a _v2 version
// of a function that you, e.g., want to add a new function parameter to, and
// maintain the old and new versions of the APIs until all old model.so
// go out of use.

#ifdef __GNUC__
#define AOTI_TORCH_EXPORT __attribute__((__visibility__("default")))
#else // !__GNUC__
#ifdef _WIN32
// PyTorch2 doesn't currently work on Windows. Exporting these APIs can lead
// to symbol clashes at link time if libtorch is included in a DLL and binary
// that depends on the DLL. As a short term fix, we don't export the symbols.
// In the long term, this will need to be addressed when Windows is supported.
// #define AOTI_TORCH_EXPORT __declspec(dllexport)
#define AOTI_TORCH_EXPORT
#else // !_WIN32
#define AOTI_TORCH_EXPORT
#endif // _WIN32
#endif // __GNUC__

// The following files are implemented in a header-only way and are guarded by
// test/cpp/aoti_abi_check
#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e4m3fnuz.h>
#include <c10/util/Float8_e5m2.h>
#include <c10/util/Float8_e5m2fnuz.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>

#ifdef __cplusplus
extern "C" {
#endif

// AtenTensorHandle represents an abstract notion of Tensor that can be passed
// between model.so and libtorch.so.  The contents of the structure itself
// are private; model.so is not allowed to access any fields directly, it must
// go through functions defined in this ABI.  Under the hood, this is
// represented as at::Tensor*, but we reserve the right to change this (and in
// fact, we probably should change it to at::TensorImpl* at least).
//
// An AtenTensorHandle can be owning (please check the API reference for exact
// ownership/borrow semantics).  If you have an owning AtenTensorHandle
// in model.so, you are obligated to aoti_torch_delete_tensor_object when you
// are done.  You can use the helper C++ class RAIIAtenTensorHandle
// (see aot_runtime/model.h) to ensure the deallocator is called in RAII style
// (note that RAIIAtenTensorHandle is private to model.so, and never crosses
// the ABI boundary.)
struct AtenTensorOpaque;
using AtenTensorHandle = AtenTensorOpaque*;

struct AtenGeneratorOpaque;
using AtenGeneratorHandle = AtenGeneratorOpaque*;

struct AOTIProxyExecutorOpaque;
using AOTIProxyExecutorHandle = AOTIProxyExecutorOpaque*;

using AOTITorchError = int32_t;
#define AOTI_TORCH_SUCCESS 0
#define AOTI_TORCH_FAILURE 1

// Getter functions for retrieving various constants from the runtime, that
// can subsequently be passed to other aoti_* functions.  By hiding these
// behind functions, the precise value of device/dtype is NOT part of the
// ABI contract.  (In practice, aten/c10 is pretty good about not renumbering
// these, so we probably could later switch to having these in the ABI, if
// desired for perf reasons.)
AOTI_TORCH_EXPORT int32_t aoti_torch_device_type_cpu();
AOTI_TORCH_EXPORT int32_t aoti_torch_device_type_cuda();
AOTI_TORCH_EXPORT int32_t aoti_torch_device_type_privateuse1();

AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_float8_e5m2();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_float8_e4m3fn();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_float8_e5m2fnuz();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_float8_e4m3fnuz();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_bfloat16();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_float16();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_float32();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_float64();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_uint8();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_uint16();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_uint32();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_uint64();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_int8();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_int16();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_int32();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_int64();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_bool();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_complex32();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_complex64();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_complex128();

AOTI_TORCH_EXPORT int32_t aoti_torch_layout_strided();
AOTI_TORCH_EXPORT int32_t aoti_torch_layout__mkldnn();

// Functions for converting a single-element tensor to a scalar value
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_item_float16(AtenTensorHandle tensor, c10::Half* ret_value);
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_item_float32(AtenTensorHandle tensor, float* ret_value);
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_item_float64(AtenTensorHandle tensor, double* ret_value);
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_item_uint8(AtenTensorHandle tensor, uint8_t* ret_value);
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_item_uint16(AtenTensorHandle tensor, uint16_t* ret_value);
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_item_uint32(AtenTensorHandle tensor, uint32_t* ret_value);
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_item_uint64(AtenTensorHandle tensor, uint64_t* ret_value);
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_item_int8(AtenTensorHandle tensor, int8_t* ret_value);
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_item_int16(AtenTensorHandle tensor, int16_t* ret_value);
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_item_int32(AtenTensorHandle tensor, int32_t* ret_value);
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_item_int64(AtenTensorHandle tensor, int64_t* ret_value);
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_item_bool(AtenTensorHandle tensor, bool* ret_value);
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_item_bfloat16(AtenTensorHandle tensor, c10::BFloat16* ret_value);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_item_complex64(
    AtenTensorHandle tensor,
    c10::complex<float>* ret_value);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_item_float8_e4m3fn(
    AtenTensorHandle tensor,
    c10::Float8_e4m3fn* ret_value);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_item_float8_e5m2(
    AtenTensorHandle tensor,
    c10::Float8_e5m2* ret_value);

// Functions for wrapping a scalar value to a single-element tensor
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_scalar_to_tensor_float32(
    float value,
    AtenTensorHandle* ret_new_tensor);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_scalar_to_tensor_float64(
    double value,
    AtenTensorHandle* ret_new_tensor);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_scalar_to_tensor_uint8(
    uint8_t value,
    AtenTensorHandle* ret_new_tensor);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_scalar_to_tensor_uint16(
    uint16_t value,
    AtenTensorHandle* ret_new_tensor);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_scalar_to_tensor_uint32(
    uint32_t value,
    AtenTensorHandle* ret_new_tensor);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_scalar_to_tensor_uint64(
    uint64_t value,
    AtenTensorHandle* ret_new_tensor);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_scalar_to_tensor_int8(
    int8_t value,
    AtenTensorHandle* ret_new_tensor);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_scalar_to_tensor_int16(
    int16_t value,
    AtenTensorHandle* ret_new_tensor);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_scalar_to_tensor_int32(
    int32_t value,
    AtenTensorHandle* ret_new_tensor);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_scalar_to_tensor_int64(
    int64_t value,
    AtenTensorHandle* ret_new_tensor);
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_scalar_to_tensor_bool(bool value, AtenTensorHandle* ret_new_tensor);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_scalar_to_tensor_complex64(
    c10::complex<float> value,
    AtenTensorHandle* ret_new_tensor);

AOTI_TORCH_EXPORT bool aoti_torch_grad_mode_is_enabled();
AOTI_TORCH_EXPORT void aoti_torch_grad_mode_set_enabled(bool enabled);

// Free the tensor object
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_delete_tensor_object(AtenTensorHandle tensor);

// Get a pointer to the underlying storage data
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_get_data_ptr(
    AtenTensorHandle tensor,
    void** ret_data_ptr // returns borrowed reference
);

// Get the nbytes of the underlying storage
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_get_storage_size(AtenTensorHandle tensor, int64_t* ret_size);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_get_dim(AtenTensorHandle tensor, int64_t* ret_dim);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_get_numel(AtenTensorHandle tensor, int64_t* ret_numel);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_get_storage_numel(AtenTensorHandle tensor, int64_t* ret_numel);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_get_sizes(
    AtenTensorHandle tensor,
    int64_t** ret_sizes // returns borrowed reference
);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_get_size(AtenTensorHandle tensor, int64_t d, int64_t* ret_size);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_get_strides(
    AtenTensorHandle tensor,
    int64_t** ret_strides // returns borrowed reference
);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_get_stride(AtenTensorHandle tensor, int64_t d, int64_t* ret_stride);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_get_dtype(AtenTensorHandle tensor, int32_t* ret_dtype);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_get_device_type(AtenTensorHandle tensor, int32_t* ret_device_type);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_get_device_index(AtenTensorHandle tensor, int32_t* ret_device_index);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_get_storage_offset(
    AtenTensorHandle tensor,
    int64_t* ret_storage_offset);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch__alloc_from_pool(
    AtenTensorHandle self,
    int64_t offset_bytes,
    int32_t dtype,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    AtenTensorHandle* ret_new_tensor);

// This function will create a new tensor object and its pointer is returned
// through *out. The caller is responsible for wrapping the tensor pointer
// with RAIIAtenTensorHandle which will call aoti_torch_delete_tensor_object
// when going out of scope.
AOTI_TORCH_EXPORT AOTITorchError aoti_torch__reinterpret_tensor(
    AtenTensorHandle self,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset,
    AtenTensorHandle* ret_new_tensor // returns new reference
);

// This function will create a new tensor object and its pointer is returned
// through *out. The caller is responsible for wrapping the tensor pointer
// with RAIIAtenTensorHandle which will call aoti_torch_delete_tensor_object
// when going out of scope.
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_empty_strided(
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    AtenTensorHandle* ret_new_tensor // returns new reference
);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_create_tensor_from_blob(
    void* data,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    AtenTensorHandle* ret // returns new reference
);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_create_tensor_from_blob_v2(
    void* data,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    AtenTensorHandle* ret, // returns new reference
    int32_t layout,
    const uint8_t* opaque_metadata,
    int64_t opaque_metadata_size);

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
);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch__fft_c2c(
    AtenTensorHandle self,
    const int64_t* dim_ptr,
    int64_t dim_size,
    int64_t normalization,
    int32_t forward,
    AtenTensorHandle* ret // returns new reference
);

// This version is deprecated. We will remove it later
AOTI_TORCH_EXPORT AOTITorchError aoti_torch__scaled_dot_product_flash_attention(
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
);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch__scaled_dot_product_flash_attention_v2(
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
);

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
);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch__scaled_mm(
    AtenTensorHandle self,
    AtenTensorHandle mat2,
    AtenTensorHandle bias,
    int32_t* out_dtype,
    AtenTensorHandle scale_a,
    AtenTensorHandle scale_b,
    AtenTensorHandle scale_result,
    int8_t use_fast_accum,
    AtenTensorHandle* ret0,
    AtenTensorHandle* ret1);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch__scaled_mm_v2(
    AtenTensorHandle self,
    AtenTensorHandle mat2,
    AtenTensorHandle scale_a,
    AtenTensorHandle scale_b,
    AtenTensorHandle bias,
    AtenTensorHandle scale_result,
    int32_t* out_dtype,
    int8_t use_fast_accum,
    AtenTensorHandle* ret0);

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
    AtenTensorHandle* ret // returns new reference
);

// This function will create a new uninitialized tensor object
// and its pointer is returned through *ret.
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_new_uninitialized_tensor(AtenTensorHandle* ret);

// WARNING: This will be deprecated. Use aoti_torch_copy_ instead.
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_tensor_copy_(AtenTensorHandle src, AtenTensorHandle dst);

// Make the tensor referred to by dst an alias for the tensor referred
// to by src. The two tensors must still be deleted with
// aoti_torch_delete_tensor separately (or not) as before the call.
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_assign_tensors(AtenTensorHandle src, AtenTensorHandle dst);

// Make a shallow copy of the tensor referred to by src and assign
// it to the handle in the ret_dst. This is similar to the above
// aoti_torch_assign_tensors function, but creates and sets the
// ret_dst from within.
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_assign_tensors_out(AtenTensorHandle src, AtenTensorHandle* ret_dst);

// This function will create a new tensor object and its pointer is returned
// through *ret. The caller is responsible for wrapping the tensor pointer
// with RAIIAtenTensorHandle which will call aoti_torch_delete_tensor_object
// when going out of scope.
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_clone(AtenTensorHandle self, AtenTensorHandle* ret);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_addmm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat1,
    AtenTensorHandle mat2,
    float beta,
    float alpha);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_bmm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat2);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_copy_(
    AtenTensorHandle self,
    AtenTensorHandle src,
    int32_t non_blocking);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_mm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat2);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch__mm_plus_mm_out(
    AtenTensorHandle out,
    AtenTensorHandle a,
    AtenTensorHandle b,
    AtenTensorHandle c,
    AtenTensorHandle d);

// This will soon be deprecated after ao_quantization is complete.
// Please refrain from using this or increasing callsites.
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_cpu_wrapped_fbgemm_pack_gemm_matrix_fp16(
    AtenTensorHandle weight,
    AtenTensorHandle* out);

// This will soon be deprecated after ao_quantization is complete.
// Please refrain from using this or increasing callsites.
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu__wrapped_linear_prepack(
    AtenTensorHandle weight,
    AtenTensorHandle weight_scale,
    AtenTensorHandle weight_zero_point,
    AtenTensorHandle bias,
    AtenTensorHandle* out);

// This will soon be deprecated after ao_quantization is complete.
// Please refrain from using this or increasing callsites.
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_cpu_wrapped_fbgemm_linear_fp16_weight(
    AtenTensorHandle input,
    AtenTensorHandle weight,
    AtenTensorHandle bias,
    int64_t out_channel,
    AtenTensorHandle* out);

// This will soon be deprecated after ao_quantization is complete.
// Please refrain from using this or increasing callsites.
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_cpu__wrapped_quantized_linear_prepacked(
    AtenTensorHandle input,
    AtenTensorHandle input_scale,
    AtenTensorHandle input_zero_point,
    AtenTensorHandle weight,
    AtenTensorHandle out_scale,
    AtenTensorHandle out_zeropoint,
    int64_t out_channel,
    AtenTensorHandle* out);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_nonzero(AtenTensorHandle self, AtenTensorHandle* out);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_zero_(AtenTensorHandle self);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_repeat_interleave_Tensor(
    AtenTensorHandle repeats,
    int64_t* output_size,
    AtenTensorHandle* out);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_check_inf_and_nan(const char* tensor_name, AtenTensorHandle tensor);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_scatter_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    int64_t dim,
    AtenTensorHandle index,
    AtenTensorHandle src);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_scatter_reduce_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    int64_t dim,
    AtenTensorHandle index,
    AtenTensorHandle src,
    const char* reduce,
    int32_t include_self);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_index_put_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    const AtenTensorHandle* indices,
    const uint32_t num_indices,
    const AtenTensorHandle values,
    bool accumulate);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_view_as_real(
    AtenTensorHandle self,
    AtenTensorHandle* ret // returns new reference
);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_view_dtype(
    AtenTensorHandle self,
    int32_t dtype,
    AtenTensorHandle* ret // returns new reference
);

AOTI_TORCH_EXPORT void aoti_torch_print_tensor_handle(
    AtenTensorHandle self,
    const char* msg);

// When AOTI debug printer option is enabled, this function will be invoked to
// torch pickle save the intermediate tensor for debugging purpose.
AOTI_TORCH_EXPORT void aoti_torch_save_tensor_handle(
    AtenTensorHandle self,
    const char* tensor_name,
    const char* launch_prefix,
    const char* kernel_name);

#ifdef USE_CUDA

struct CUDAGuardOpaque;
using CUDAGuardHandle = CUDAGuardOpaque*;

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_create_cuda_guard(
    int32_t device_index,
    CUDAGuardHandle* ret_guard // returns new reference
);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_delete_cuda_guard(CUDAGuardHandle guard);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_cuda_guard_set_index(CUDAGuardHandle guard, int32_t device_index);

struct CUDAStreamGuardOpaque;
using CUDAStreamGuardHandle = CUDAStreamGuardOpaque*;

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_create_cuda_stream_guard(
    void* stream,
    int32_t device_index,
    CUDAStreamGuardHandle* ret_guard // returns new reference
);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_delete_cuda_stream_guard(CUDAStreamGuardHandle guard);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_get_current_cuda_stream(int32_t device_index, void** ret_stream);

#endif // USE_CUDA

// See `ProxyExecutor Design Note` in ir.py for more details
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_proxy_executor_call_function(
    AOTIProxyExecutorHandle proxy_executor,
    int extern_node_index,
    int num_ints,
    int64_t* flatten_int_args,
    int num_tensors,
    AtenTensorHandle* flatten_tensor_args);

AOTI_TORCH_EXPORT void aoti_torch_check(
    bool cond,
    const char* func,
    const char* file,
    uint32_t line,
    const char* msg);

#ifdef STRIP_ERROR_MESSAGES
#define AOTI_TORCH_CHECK(cond, ...)              \
  if (!(cond)) {                                 \
    aoti_torch_check(                            \
        false,                                   \
        __func__,                                \
        __FILE__,                                \
        static_cast<uint32_t>(__LINE__),         \
        TORCH_CHECK_MSG(cond, "", __VA_ARGS__)); \
  }
#else
#define AOTI_TORCH_CHECK(cond, ...)                \
  if (!(cond)) {                                   \
    aoti_torch_check(                              \
        false,                                     \
        __func__,                                  \
        __FILE__,                                  \
        static_cast<uint32_t>(__LINE__),           \
        TORCH_CHECK_MSG(cond, "", ##__VA_ARGS__)); \
  }
#endif

#ifdef __cplusplus
} // extern "C"

template <typename T>
int32_t aoti_torch_dtype() = delete;

#define DEFINE_DTYPE_SPECIALIZATION(ctype, typename) \
  template <>                                        \
  inline int32_t aoti_torch_dtype<ctype>() {         \
    return aoti_torch_dtype_##typename();            \
  }

namespace c10 {
struct BFloat16;
struct Half;
struct Float8_e4m3fn;
struct Float8_e5m2;
} // namespace c10

DEFINE_DTYPE_SPECIALIZATION(c10::BFloat16, bfloat16)
DEFINE_DTYPE_SPECIALIZATION(c10::Half, float16)
DEFINE_DTYPE_SPECIALIZATION(c10::complex<float>, complex64)
DEFINE_DTYPE_SPECIALIZATION(float, float32)
DEFINE_DTYPE_SPECIALIZATION(double, float64)
DEFINE_DTYPE_SPECIALIZATION(uint8_t, uint8)
DEFINE_DTYPE_SPECIALIZATION(int8_t, int8)
DEFINE_DTYPE_SPECIALIZATION(int16_t, int16)
DEFINE_DTYPE_SPECIALIZATION(int32_t, int32)
DEFINE_DTYPE_SPECIALIZATION(int64_t, int64)
DEFINE_DTYPE_SPECIALIZATION(bool, bool)
DEFINE_DTYPE_SPECIALIZATION(c10::Float8_e4m3fn, float8_e4m3fn)
DEFINE_DTYPE_SPECIALIZATION(c10::Float8_e5m2, float8_e5m2)

#endif

#endif // AOTI_TORCH_SHIM
