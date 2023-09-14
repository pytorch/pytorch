#ifndef AOTI_TORCH_SHIM
#define AOTI_TORCH_SHIM

#include <stddef.h>
#include <stdint.h>

// This header defines a stable C API for certain ATen functionality in
// libtorch. The AOTInductor compiled model.so will only refer to this header
// instead of other headers from aten/c10, which means it will NOT be able to
// directly use any data structures or call functions from libtorch.
//
// What problems are we trying to solve here? If we Ahead-of-Time compile a
// model using AOTInductor and directly call aten ops or use aten/c10 data
// structures in the generated code, we will end up with ABI compatibility
// breakage. By introducing a C shim layer, we can minimize the surface that
// will cause breakage. The corresponding software stack can be illustrated
// as follows:
//
// |--------------------------------|
// |     inference service code     |
// |--------------------------------|
// |           model.so             |
// |--------------|-----------------|
// |           <c shim>             |
// |          libtorch.so           |
// |--------------------------------|

#ifdef __GNUC__
#define AOTI_TORCH_EXPORT __attribute__((__visibility__("default")))
#else // !__GNUC__
#ifdef _WIN32
#define AOTI_TORCH_EXPORT __declspec(dllexport)
#else // !_WIN32
#define AOTI_TORCH_EXPORT
#endif // _WIN32
#endif // __GNUC__

#ifdef __GNUC__
#define AOTI_TORCH_NOINLINE __attribute__((noinline))
#elif _MSC_VER
#define AOTI_TORCH_NOINLINE __declspec(noinline)
#else
#define AOTI_TORCH_NOINLINE
#endif

// AtenTensorHandle is basically an at::Tensor pointer. We pass AtenTensorHandle
// instead of raw at::Tensor aross the boundary between model.so and libtorch.so
// to avoid accidental ABI breakage.
//
// In terms of the ownership of at::Tensor objects, model.so owns them:
// For fallback ops that allocate new at::Tensor objects, e.g. empty_strided
// and convolution (which has no out-variant), the fallback functions will
// allocate those at::Tensor objects and return their raw pointers. The caller,
// generated code in model.so, is responsible for wrapping those raw pointers
// into RAIIAtenTensorHandle (see aot_runtime/model.h) which will take care of
// freeing those allocated at::Tensor objects.
struct AtenTensorOpaque {};
using AtenTensorHandle = AtenTensorOpaque*;

#ifdef __cplusplus
extern "C" {
#endif

using AOTITorchError = int32_t;
#define AOTI_TORCH_SUCCESS 0
#define AOTI_TORCH_FAILURE 1

// WARNING: Change the following signatures will break ABI compatibility
AOTI_TORCH_EXPORT AOTI_TORCH_NOINLINE int32_t aoti_torch_device_type_cpu();

AOTI_TORCH_EXPORT AOTI_TORCH_NOINLINE int32_t aoti_torch_device_type_cuda();

AOTI_TORCH_EXPORT AOTI_TORCH_NOINLINE int32_t aoti_torch_dtype_bfloat16();

AOTI_TORCH_EXPORT AOTI_TORCH_NOINLINE int32_t aoti_torch_dtype_float16();

AOTI_TORCH_EXPORT AOTI_TORCH_NOINLINE int32_t aoti_torch_dtype_float32();

AOTI_TORCH_EXPORT AOTI_TORCH_NOINLINE int32_t aoti_torch_dtype_float64();

AOTI_TORCH_EXPORT AOTI_TORCH_NOINLINE int32_t aoti_torch_dtype_uint8();

AOTI_TORCH_EXPORT AOTI_TORCH_NOINLINE int32_t aoti_torch_dtype_int8();

AOTI_TORCH_EXPORT AOTI_TORCH_NOINLINE int32_t aoti_torch_dtype_int16();

AOTI_TORCH_EXPORT AOTI_TORCH_NOINLINE int32_t aoti_torch_dtype_int32();

AOTI_TORCH_EXPORT AOTI_TORCH_NOINLINE int32_t aoti_torch_dtype_int64();

// Free the tensor object
AOTI_TORCH_EXPORT AOTI_TORCH_NOINLINE AOTITorchError
aoti_torch_delete_tensor_object(AtenTensorHandle tensor);

// Get a pointer to the underlying storage data
AOTI_TORCH_EXPORT AOTI_TORCH_NOINLINE AOTITorchError
aoti_torch_get_data_ptr(AtenTensorHandle tensor, void** data_ptr);

// This function will create a new tensor object and its pointer is returned
// through *out. The caller is responsible for wrapping the tensor pointer
// with RAIIAtenTensorHandle which will call aoti_torch_delete_tensor_object
// when going out of scope.
AOTI_TORCH_EXPORT AOTI_TORCH_NOINLINE AOTITorchError
aoti_torch__reinterpret_tensor(
    AtenTensorHandle* out,
    AtenTensorHandle self,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset);

// This function will create a new tensor object and its pointer is returned
// through *out. The caller is responsible for wrapping the tensor pointer
// with RAIIAtenTensorHandle which will call aoti_torch_delete_tensor_object
// when going out of scope.
AOTI_TORCH_EXPORT AOTI_TORCH_NOINLINE AOTITorchError aoti_torch_empty_strided(
    AtenTensorHandle* out,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index);

AOTI_TORCH_EXPORT AOTI_TORCH_NOINLINE AOTITorchError
aoti_torch_tensor_copy_(AtenTensorHandle src, AtenTensorHandle dst);

AOTI_TORCH_EXPORT AOTI_TORCH_NOINLINE AOTITorchError aoti_torch_addmm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat1,
    AtenTensorHandle mat2,
    float beta,
    float alpha);

AOTI_TORCH_EXPORT AOTI_TORCH_NOINLINE AOTITorchError aoti_torch_bmm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat2);

AOTI_TORCH_EXPORT AOTI_TORCH_NOINLINE AOTITorchError aoti_torch_mm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat2);

#ifdef USE_CUDA
// Assuming CUDA ABI is stable, and any major CUDA upgrade will require a
// rebuild of model.so
AOTI_TORCH_EXPORT AOTI_TORCH_NOINLINE AOTITorchError
aoti_torch_set_current_cuda_stream(void* stream, int32_t device_index);
#endif

#ifdef __cplusplus
} // extern "C"
#endif

#endif // AOTI_TORCH_SHIM
