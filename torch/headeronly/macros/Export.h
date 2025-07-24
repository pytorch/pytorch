#pragma once

#ifndef C10_MACROS_EXPORT_H_
#define C10_MACROS_EXPORT_H_

#ifndef C10_USING_CUSTOM_GENERATED_MACROS
#include <torch/headeronly/macros/cmake_macros.h>
#endif // C10_USING_CUSTOM_GENERATED_MACROS

/* Header file to define the common scaffolding for exported symbols.
 *
 * Export is by itself a quite tricky situation to deal with, and if you are
 * hitting this file, make sure you start with the background here:
 * - Linux: https://gcc.gnu.org/wiki/Visibility
 * - Windows:
 * https://docs.microsoft.com/en-us/cpp/cpp/dllexport-dllimport?view=vs-2017
 *
 * Do NOT include this file directly. Instead, use c10/macros/Macros.h
 */

// You do not need to edit this part of file unless you are changing the core
// pytorch export abstractions.
//
// This part defines the C10 core export and import macros. This is controlled
// by whether we are building shared libraries or not, which is determined
// during build time and codified in c10/core/cmake_macros.h.
// When the library is built as a shared lib, EXPORT and IMPORT will contain
// visibility attributes. If it is being built as a static lib, then EXPORT
// and IMPORT basically have no effect.

// As a rule of thumb, you should almost NEVER mix static and shared builds for
// libraries that depend on c10. AKA, if c10 is built as a static library, we
// recommend everything dependent on c10 to be built statically. If c10 is built
// as a shared library, everything dependent on it should be built as shared. In
// the PyTorch project, all native libraries shall use the macro
// C10_BUILD_SHARED_LIB to check whether pytorch is building shared or static
// libraries.

// For build systems that do not directly depend on CMake and directly build
// from the source directory (such as Buck), one may not have a cmake_macros.h
// file at all. In this case, the build system is responsible for providing
// correct macro definitions corresponding to the cmake_macros.h.in file.
//
// In such scenarios, one should define the macro
//     C10_USING_CUSTOM_GENERATED_MACROS
// to inform this header that it does not need to include the cmake_macros.h
// file.

#ifdef _WIN32
#define C10_HIDDEN
#if defined(C10_BUILD_SHARED_LIBS)
#define C10_EXPORT __declspec(dllexport)
#define C10_IMPORT __declspec(dllimport)
#else
#define C10_EXPORT
#define C10_IMPORT
#endif
#else // _WIN32
#if defined(__GNUC__)
#define C10_EXPORT __attribute__((__visibility__("default")))
#define C10_HIDDEN __attribute__((__visibility__("hidden")))
#else // defined(__GNUC__)
#define C10_EXPORT
#define C10_HIDDEN
#endif // defined(__GNUC__)
#define C10_IMPORT C10_EXPORT
#endif // _WIN32

#ifdef NO_EXPORT
#undef C10_EXPORT
#define C10_EXPORT
#endif

// Definition of an adaptive XX_API macro, that depends on whether you are
// building the library itself or not, routes to XX_EXPORT and XX_IMPORT.
// Basically, you will need to do this for each shared library that you are
// building, and the instruction is as follows: assuming that you are building
// a library called libawesome.so. You should:
// (1) for your cmake target (usually done by "add_library(awesome, ...)"),
//     define a macro called AWESOME_BUILD_MAIN_LIB using
//     target_compile_options.
// (2) define the AWESOME_API macro similar to the one below.
// And in the source file of your awesome library, use AWESOME_API to
// annotate public symbols.

// Here, for the C10 library, we will define the macro C10_API for both import
// and export.

// This one is being used by libc10.so
#ifdef C10_BUILD_MAIN_LIB
#define C10_API C10_EXPORT
#else
#define C10_API C10_IMPORT
#endif

// This one is being used by libtorch.so
#ifdef CAFFE2_BUILD_MAIN_LIB
#define TORCH_API C10_EXPORT
#else
#define TORCH_API C10_IMPORT
#endif

// You may be wondering why we have TORCH_CUDA_CPP_API and TORCH_CUDA_CU_API
// belonging to the same library instead of just one TORCH_CUDA_API. Well, it
// can indeed just be one TORCH_CUDA_API (and used to be)! TORCH_CUDA_CPP_API
// and TORCH_CUDA_CU_API are artifacts of when we needed a split build to
// avoid relocation marker linking errors. The context is as follows:
//
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

// libtorch_cuda.so (where torch_cuda_cu and torch_cuda_cpp are a part of the
// same api)
#ifdef TORCH_CUDA_BUILD_MAIN_LIB
#define TORCH_CUDA_CPP_API C10_EXPORT
#define TORCH_CUDA_CU_API C10_EXPORT
#else
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
