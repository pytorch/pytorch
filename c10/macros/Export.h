#ifndef C10_MACROS_EXPORT_H_
#define C10_MACROS_EXPORT_H_

#ifndef C10_USING_CUSTOM_GENERATED_MACROS
#include <c10/macros/cmake_macros.h>
#endif // C10_USING_CUSTOM_GENERATED_MACROS

#include <torch/headeronly/macros/Export.h>

// This one is being used by libtorch.so
#ifdef CAFFE2_BUILD_MAIN_LIB
#define TORCH_API C10_EXPORT
#else
#define TORCH_API C10_IMPORT
#endif

// You may be wondering: Whose brilliant idea was it to split torch_cuda into
// two pieces with confusing names?
// Once upon a time, there _was_ only TORCH_CUDA_API. All was happy until we
// tried to compile PyTorch for CUDA 11.1, which ran into relocation marker
// issues when linking big binaries.
// (https://github.com/pytorch/pytorch/issues/39968) We had two choices:
//    (1) Stop supporting so many GPU architectures
//    (2) Do something else
// We chose #2 and decided to split the behemoth that was torch_cuda into two
// smaller libraries, one with most of the core kernel functions (torch_cuda_cu)
// and the other that had..well..everything else (torch_cuda_cpp). The idea was
// this: instead of linking our static libraries (like the hefty
// libcudnn_static.a) with another huge library, torch_cuda, and run into pesky
// relocation marker issues, we could link our static libraries to a smaller
// part of torch_cuda (torch_cuda_cpp) and avoid the issues.

// libtorch_cuda_cu.so
#ifdef TORCH_CUDA_CU_BUILD_MAIN_LIB
#define TORCH_CUDA_CU_API C10_EXPORT
#elif defined(BUILD_SPLIT_CUDA)
#define TORCH_CUDA_CU_API C10_IMPORT
#endif

// libtorch_cuda_cpp.so
#ifdef TORCH_CUDA_CPP_BUILD_MAIN_LIB
#define TORCH_CUDA_CPP_API C10_EXPORT
#elif defined(BUILD_SPLIT_CUDA)
#define TORCH_CUDA_CPP_API C10_IMPORT
#endif

// libtorch_cuda.so (where torch_cuda_cu and torch_cuda_cpp are a part of the
// same api)
#ifdef TORCH_CUDA_BUILD_MAIN_LIB
#define TORCH_CUDA_CPP_API C10_EXPORT
#define TORCH_CUDA_CU_API C10_EXPORT
#elif !defined(BUILD_SPLIT_CUDA)
#define TORCH_CUDA_CPP_API C10_IMPORT
#define TORCH_CUDA_CU_API C10_IMPORT
#endif

#if defined(TORCH_HIP_BUILD_MAIN_LIB)
#define TORCH_HIP_CPP_API C10_EXPORT
#define TORCH_HIP_API C10_EXPORT
#else
#define TORCH_HIP_CPP_API C10_IMPORT
#define TORCH_HIP_API C10_IMPORT
#endif

#if defined(TORCH_XPU_BUILD_MAIN_LIB)
#define TORCH_XPU_API C10_EXPORT
#else
#define TORCH_XPU_API C10_IMPORT
#endif

// Enums only need to be exported on windows for non-CUDA files
#if defined(_WIN32) && defined(__CUDACC__)
#define C10_API_ENUM C10_API
#else
#define C10_API_ENUM
#endif

#endif // C10_MACROS_EXPORT_H_
