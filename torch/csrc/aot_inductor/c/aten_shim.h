#pragma once

#include <stdint.h>
#include <torch/csrc/aot_inductor/c/utils.h>

// What problems are we trying to solve here with a C shim layer?
//
// If we Ahead-of-Time compile a model using AOTInductor and directly call aten
// ops or use aten/c10 data structures in the generated code, we will end up
// with ABI compatibility issues. People working on aten/c10 C++ code may not
// know there are some AOTInductor compiled model code relying on aten/c10 code
// to be ABI stable. By introducing a C shim layer here, we are minimizing the
// surface that could cause ABI breakage. AOTInductor generated code will
// NOT be allowed to call any of aten/c10 C++ APIs. It is only allowed to call
// functions declared in this header file instead.

// WARNING: Change the following types can break ABI compatibility
// Typedef these to be exactly the same as c10 since C enum types can not be
// specified as int8_t. We will assert the data type length at run time.
typedef int8_t AOTInductorDeviceType;
typedef int8_t AOTInductorDeviceIndex;
typedef int8_t AOTInductorScalarType;

// WARNING: Change the following enum values can break ABI compatibility
// Define these enum values to be exactly the same as c10. Again, we will check
// these at run time.
#define kAOTInductorCPU 0
#define kAOTInductorCUDA 1

#define kAOTInductorByte 0
#define kAOTInductorChar 1
#define kAOTInductorShort 2
#define kAOTInductorInt 3
#define kAOTInductorLong 4
#define kAOTInductorHalf 5
#define kAOTInductorFloat 6
#define kAOTInductorDouble 7
#define kAOTInductorComplexHalf 8
#define kAOTInductorComplexFloat 9
#define kAOTInductorComplexDouble 10
#define kAOTInductorBool 11
#define kAOTInductorQInt8 12
#define kAOTInductorQUInt8 13
#define kAOTInductorQInt32 14
#define kAOTInductorBFloat16 15

#ifdef __cplusplus
extern "C" {
#endif

// WARNING: Change the following signatures can break ABI compatibility
AOT_INDUCTOR_EXPORT void aot_inductor_initialize();

AOT_INDUCTOR_EXPORT void aot_inductor_destroy();

AOT_INDUCTOR_EXPORT void aot_inductor_free(AOTInductorTensorHandle aot_tensor);

AOT_INDUCTOR_EXPORT void* aot_inductor_data_ptr(
    AOTInductorTensorHandle aot_tensor);

AOT_INDUCTOR_EXPORT AOTInductorTensorHandle
convert_input_output_to_aot_tensor(AtenTensorHandle aten_tensor);

AOT_INDUCTOR_EXPORT AOTInductorTensorHandle aot_inductor_empty_strided(
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    AOTInductorDeviceType device_type,
    AOTInductorDeviceIndex device_index,
    AOTInductorScalarType dtype);

AOT_INDUCTOR_EXPORT AOTInductorTensorHandle aot_inductor_as_strided(
    AOTInductorTensorHandle self,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset);

AOT_INDUCTOR_EXPORT AOTInductorTensorHandle aot_inductor_addmm_out(
    AOTInductorTensorHandle out,
    AOTInductorTensorHandle self,
    AOTInductorTensorHandle mat1,
    AOTInductorTensorHandle mat2,
    float beta,
    float alpha);

AOT_INDUCTOR_EXPORT AOTInductorTensorHandle aot_inductor__addmm_activation(
    AOTInductorTensorHandle self,
    AOTInductorTensorHandle mat1,
    AOTInductorTensorHandle mat2,
    float beta,
    float alpha,
    uint8_t use_gelu);

AOT_INDUCTOR_EXPORT AOTInductorTensorHandle aot_inductor_bmm_out(
    AOTInductorTensorHandle out,
    AOTInductorTensorHandle self,
    AOTInductorTensorHandle mat2);

AOT_INDUCTOR_EXPORT AOTInductorTensorHandle
aot_inductor_copy_(AOTInductorTensorHandle src, AOTInductorTensorHandle dst);

AOT_INDUCTOR_EXPORT AOTInductorTensorHandle aot_inductor_mm_out(
    AOTInductorTensorHandle out,
    AOTInductorTensorHandle self,
    AOTInductorTensorHandle mat2);

#ifdef USE_CUDA
AOT_INDUCTOR_EXPORT void aot_inductor_set_current_cuda_stream(
    void* stream,
    AOTInductorDeviceIndex device_index);
#endif

#ifdef __cplusplus
}
#endif
