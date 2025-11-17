#ifndef AOTI_TORCH_SHIM
#define AOTI_TORCH_SHIM

#include <torch/csrc/inductor/aoti_torch/c/macros.h>
#include <torch/csrc/inductor/aoti_torch/c/shim_deprecated.h>

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

// The following files are implemented in a header-only way and are guarded by
// test/cpp/aoti_abi_check
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>

#ifdef __cplusplus
extern "C" {
#endif

// Getter functions for retrieving various constants from the runtime, that
// can subsequently be passed to other aoti_* functions.  By hiding these
// behind functions, the precise value of device/dtype is NOT part of the
// ABI contract.  (In practice, aten/c10 is pretty good about not renumbering
// these, so we probably could later switch to having these in the ABI, if
// desired for perf reasons.)
AOTI_TORCH_EXPORT int32_t aoti_torch_device_type_cpu();
AOTI_TORCH_EXPORT int32_t aoti_torch_device_type_cuda();
AOTI_TORCH_EXPORT int32_t aoti_torch_device_type_meta();
AOTI_TORCH_EXPORT int32_t aoti_torch_device_type_xpu();
AOTI_TORCH_EXPORT int32_t aoti_torch_device_type_mps();
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
AOTI_TORCH_EXPORT size_t aoti_torch_dtype_element_size(int32_t dtype);

AOTI_TORCH_EXPORT int32_t aoti_torch_layout_strided();
AOTI_TORCH_EXPORT int32_t aoti_torch_layout_sparse_coo();
AOTI_TORCH_EXPORT int32_t aoti_torch_layout_sparse_csr();
AOTI_TORCH_EXPORT int32_t aoti_torch_layout_sparse_csc();
AOTI_TORCH_EXPORT int32_t aoti_torch_layout_sparse_bsr();
AOTI_TORCH_EXPORT int32_t aoti_torch_layout_sparse_bsc();
AOTI_TORCH_EXPORT int32_t aoti_torch_layout__mkldnn();
AOTI_TORCH_EXPORT int32_t aoti_torch_layout_jagged();

AOTI_TORCH_EXPORT int32_t aoti_torch_memory_format_contiguous_format();
AOTI_TORCH_EXPORT int32_t aoti_torch_memory_format_channels_last();
AOTI_TORCH_EXPORT int32_t aoti_torch_memory_format_channels_last_3d();
AOTI_TORCH_EXPORT int32_t aoti_torch_memory_format_preserve_format();

// Get TORCH_ABI_VERSION of the built libtorch.so
AOTI_TORCH_EXPORT uint64_t aoti_torch_abi_version();

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
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_scalar_to_tensor_complex128(
    c10::complex<double> value,
    AtenTensorHandle* ret_new_tensor);

AOTI_TORCH_EXPORT bool aoti_torch_grad_mode_is_enabled();
AOTI_TORCH_EXPORT void aoti_torch_grad_mode_set_enabled(bool enabled);

// Free the tensor object
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_delete_tensor_object(AtenTensorHandle tensor);

// c10::IValue <int64_t> object conversion
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_int64_to_ivalue(int64_t val, C10IValueHandle* ivalue);

// c10::IValue <const char** > object conversions
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_strlist_to_ivalue(
    const char** val,
    int64_t len,
    C10IValueHandle* ivalue);

// c10::IValue <const char* > object conversions
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_str_to_ivalue(const char* val, C10IValueHandle* ivalue);

// c10::IValue <at::Tensor> object conversions
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_tensor_to_ivalue(AtenTensorHandle val, C10IValueHandle* ivalue);

// Free the c10::IValue object
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_delete_c10_value_object(C10IValueHandle handle);

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

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_get_layout(AtenTensorHandle tensor, int32_t* ret_layout);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_get_storage_offset(
    AtenTensorHandle tensor,
    int64_t* ret_storage_offset);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_is_contiguous(AtenTensorHandle tensor, bool* ret_is_contiguous);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_is_defined(AtenTensorHandle tensor, bool* ret_is_defined);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_new_tensor_handle(
    AtenTensorHandle orig_handle,
    AtenTensorHandle* new_handle);

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

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_empty_strided_pinned(
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    AtenTensorHandle* ret_new_tensor // returns new reference
);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_as_strided(
    AtenTensorHandle self,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    AtenTensorHandle* ret);

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

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_clone_preserve_strides(AtenTensorHandle self, AtenTensorHandle* ret);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_copy_(
    AtenTensorHandle self,
    AtenTensorHandle src,
    int32_t non_blocking);

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
    AtenTensorHandle bias, // optional argument
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

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_zero_(AtenTensorHandle self);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_check_inf_and_nan(const char* tensor_name, AtenTensorHandle tensor);

struct AtenRecordFunctionOpaque;
using AtenRecordFunctionHandle = AtenRecordFunctionOpaque*;

struct IValueMapOpaque;
using IValueMapHandle = IValueMapOpaque*;

AOTI_TORCH_EXPORT AOTITorchError aoti_record_function_start(
    const char* name,
    IValueMapHandle kwargs,
    const C10IValueHandle* inputs,
    const uint64_t n_inputs,
    AtenRecordFunctionHandle* guard);

AOTI_TORCH_EXPORT AOTITorchError
aoti_record_function_end(AtenRecordFunctionHandle guard);

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

// helpers for converting between StableIValue and actual IValues
using StableIValue = uint64_t;

class TorchLibraryOpaque;
using TorchLibraryHandle = TorchLibraryOpaque*;

// stable corollary to torch::Library constructor with Kind::IMPL
// will create a new torch::Library object on the heap
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_library_init_impl(
    const char* ns,
    const char* k,
    const char* file,
    uint32_t line,
    TorchLibraryHandle* ret_new_torch_lib);

// stable corollary to torch::Library constructor with Kind::DEF
// will create a new torch::Library object on the heap
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_library_init_def(
    const char* ns,
    const char* file,
    uint32_t line,
    TorchLibraryHandle* ret_new_torch_lib);

// stable corollary to torch::Library constructor with Kind::FRAGMENT
// will create a new torch::Library object on the heap
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_library_init_fragment(
    const char* ns,
    const char* file,
    uint32_t line,
    TorchLibraryHandle* ret_new_torch_lib);

// stable corollary to torch::Library method m.impl(), should be
// called from StableLibrary
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_library_impl(
    TorchLibraryHandle self,
    const char* name,
    void (*fn)(StableIValue*, uint64_t, uint64_t));

// stable corollary to torch::Library method m.def(), should be
// called from StableLibrary
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_library_def(TorchLibraryHandle self, const char* schema);

// the above stable constructors for torch::Library add Library objects
// to the heap. if you are calling those functions directly, please use
// this function to free the Library's memory. The more user friendly
// alternative is to use StableLibrary, which will free its handle upon
// destruction
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_delete_library_object(TorchLibraryHandle tlh);

// calls the op overload defined by a given opName, overloadName, and a
// stack of StableIValues. This call will populate any return values of the
// op into the stack in their StableIValue form, with ret0 at index 0, ret1
// at index 1, and so on.
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_call_dispatcher(
    const char* opName,
    const char* overloadName,
    StableIValue* stack);

// Has the same semantic as aoti_torch_call_dispatcher, but takes an
// additional argument for the extension build version. This is
// needed for backward compatibility when calling native functions via
// the dispatcher. The caller should pass in its build version (not target
// version).
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_call_dispatcher_v2(
    const char* opName,
    const char* overloadName,
    StableIValue* stack,
    uint64_t extension_build_version);

// Device-generic guard for managing device context
struct DeviceGuardOpaque;
using DeviceGuardHandle = DeviceGuardOpaque*;

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_create_device_guard(
    int32_t device_index,
    DeviceGuardHandle* ret_guard // returns new reference
);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_delete_device_guard(DeviceGuardHandle guard);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_device_guard_set_index(
    DeviceGuardHandle guard,
    int32_t device_index);

// Device-generic stream for managing stream objects
struct StreamOpaque;
using StreamHandle = StreamOpaque*;

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_delete_stream(StreamHandle stream);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_stream_id(StreamHandle stream, int64_t* ret_stream_id);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_get_current_stream(
    int32_t device_index,
    StreamHandle* ret_stream // returns new reference
);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_get_current_device_index(int32_t* ret_device_index);

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

// CUDA memory allocation using CUDACachingAllocator
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cuda_caching_allocator_raw_alloc(
    uint64_t nbytes,
    void** ret_ptr // returns raw GPU memory pointer
);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_cuda_caching_allocator_raw_delete(void* ptr);

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

AOTI_TORCH_EXPORT void aoti_torch_warn(
    const char* func,
    const char* file,
    uint32_t line,
    const char* msg);

#ifdef DISABLE_WARN
#define AOTI_TORCH_WARN(...) ((void)0);
#else
#define AOTI_TORCH_WARN(...) \
  aoti_torch_warn(           \
      __func__, __FILE__, static_cast<uint32_t>(__LINE__), #__VA_ARGS__);
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

#endif
#endif // AOTI_TORCH_SHIM
