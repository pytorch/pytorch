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
// will cause breakage.

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

#ifdef __cplusplus
extern "C" {
#endif

struct AOTITensorOpaque {};
using AOTITensorHandle = AOTITensorOpaque*;

struct AtenTensorOpaque {};
using AtenTensorHandle = AtenTensorOpaque*;

struct AOTITensorManagerOpaque {};
using AOTITensorManagerHandle = AOTITensorManagerOpaque*;

enum AOTITorchError : int32_t {
  Success = 0,
  Failure = 1,
};

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

AOTI_TORCH_EXPORT AOTI_TORCH_NOINLINE AOTITensorHandle
aoti_torch_optional_tensor();

AOTI_TORCH_EXPORT AOTI_TORCH_NOINLINE AOTITorchError
aoti_torch_create_tensor_manager(
    AOTITensorManagerHandle* tensor_manager,
    int64_t reserve_size);

AOTI_TORCH_EXPORT AOTI_TORCH_NOINLINE AOTITorchError
aoti_torch_destroy_tensor_manager(AOTITensorManagerHandle tensor_manager);

// Convert input/output aten tensors to AOTInductorTensor
AOTI_TORCH_EXPORT AOTI_TORCH_NOINLINE AOTITorchError
aoti_torch_extern_tensor_to_aoti_tensor(
    AOTITensorManagerHandle tensor_manager,
    AOTITensorHandle* aoti_tensor,
    AtenTensorHandle aten_tensor);

// Free the underlying storage data
AOTI_TORCH_EXPORT AOTI_TORCH_NOINLINE AOTITorchError
aoti_torch_free_tensor_storage(AOTITensorHandle aoti_tensor);

// Get a pointer to the underlying storage data
AOTI_TORCH_EXPORT AOTI_TORCH_NOINLINE AOTITorchError
aoti_torch_get_data_ptr(AOTITensorHandle aoti_tensor, void** data_ptr);

// Return AOTITensorHandle instead of AOTITorchError makes it easier to create
// an AOTITensor for a view op
AOTI_TORCH_EXPORT AOTI_TORCH_NOINLINE AOTITensorHandle
aoti_torch__reinterpret_tensor(
    AOTITensorManagerHandle tensor_manager,
    AOTITensorHandle self,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset);

// Allocate a new AOTInductorTensor with a new tensor storage created
AOTI_TORCH_EXPORT AOTI_TORCH_NOINLINE AOTITorchError aoti_torch_empty_strided(
    AOTITensorManagerHandle tensor_manager,
    AOTITensorHandle* out,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index);

AOTI_TORCH_EXPORT AOTI_TORCH_NOINLINE AOTITorchError
aoti_torch_tensor_copy_(AOTITensorHandle src, AOTITensorHandle dst);

AOTI_TORCH_EXPORT AOTI_TORCH_NOINLINE AOTITorchError aoti_torch_addmm_out(
    AOTITensorHandle out,
    AOTITensorHandle self,
    AOTITensorHandle mat1,
    AOTITensorHandle mat2,
    float beta,
    float alpha);

AOTI_TORCH_EXPORT AOTI_TORCH_NOINLINE AOTITorchError aoti_torch_bmm_out(
    AOTITensorHandle out,
    AOTITensorHandle self,
    AOTITensorHandle mat2);

// aten::convolution does not have an out variant, so we will create an out
// AOTITensor and return through *out
AOTI_TORCH_EXPORT AOTI_TORCH_NOINLINE AOTITorchError aoti_torch_convolution(
    AOTITensorManagerHandle tensor_manager,
    AOTITensorHandle* out,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    AOTITensorHandle input,
    AOTITensorHandle weight,
    AOTITensorHandle bias,
    const int64_t* stride,
    size_t len_stride,
    const int64_t* padding,
    size_t len_padding,
    const int64_t* dilation,
    size_t len_dilation,
    int32_t transposed,
    const int64_t* output_padding,
    size_t len_output_padding,
    int32_t groups);

AOTI_TORCH_EXPORT AOTI_TORCH_NOINLINE AOTITorchError aoti_torch_mm_out(
    AOTITensorHandle out,
    AOTITensorHandle self,
    AOTITensorHandle mat2);

#ifdef USE_CUDA
AOTI_TORCH_EXPORT AOTI_TORCH_NOINLINE AOTITorchError
aoti_torch_set_current_cuda_stream(void* stream, int32_t device_index);
#endif

#ifdef __cplusplus
}
#endif

#endif // AOTI_TORCH_SHIM
