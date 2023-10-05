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
#define AOTI_TORCH_EXPORT __declspec(dllexport)
#else // !_WIN32
#define AOTI_TORCH_EXPORT
#endif // _WIN32
#endif // __GNUC__

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

AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_bfloat16();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_float16();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_float32();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_float64();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_uint8();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_int8();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_int16();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_int32();
AOTI_TORCH_EXPORT int32_t aoti_torch_dtype_int64();

// Free the tensor object
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_delete_tensor_object(AtenTensorHandle tensor);

// Get a pointer to the underlying storage data
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_get_data_ptr(
    AtenTensorHandle tensor,
    void** ret_data_ptr // returns borrowed reference
);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_get_sizes(
    AtenTensorHandle tensor,
    int64_t** ret_size // returns borrowed reference
);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_get_strides(
    AtenTensorHandle tensor,
    int64_t** ret_strides // returns borrowed reference
);

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

// This function will create a new uninitialized tensor object
// and its pointer is returned through *ret.
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_new_uninitialized_tensor(AtenTensorHandle* ret);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_tensor_copy_(AtenTensorHandle src, AtenTensorHandle dst);

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

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_mm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat2);

#ifdef USE_CUDA

struct CUDAStreamGuardOpaque;
using CUDAStreamGuardHandle = CUDAStreamGuardOpaque*;

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_create_cuda_stream_guard(
    void* stream,
    int32_t device_index,
    CUDAStreamGuardHandle* ret_guard // returns new reference
);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_delete_cuda_stream_guard(CUDAStreamGuardHandle guard);
#endif

// See `ProxyExecutor Design Note` in ir.py for more details
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_proxy_executor_call_function(
    AOTIProxyExecutorHandle proxy_executor,
    int extern_node_index,
    int num_ints,
    int64_t* flatten_int_args,
    int num_tensors,
    AtenTensorHandle* flatten_tensor_args);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // AOTI_TORCH_SHIM
