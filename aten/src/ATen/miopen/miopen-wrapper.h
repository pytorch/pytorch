#pragma once

// Enable MIOpen Beta APIs including miopenSetTensorDescriptorV2 which supports
// 64-bit tensor dimensions/strides for large tensors (numel > INT32_MAX).
// Reference: https://github.com/ROCm/MIOpen/pull/2838
#ifndef MIOPEN_BETA_API
#define MIOPEN_BETA_API 1
#endif

#include <miopen/miopen.h>
#include <miopen/version.h>

// MIOpen 3.5.2 (ROCm 7.14) introduced miopenMathType_t and the conv math-type
// attribute; TF32 conv cannot be requested at all before that.
#define MIOPEN_HAS_TF32_MATH_TYPE \
  (MIOPEN_VERSION_MAJOR > 3 || (MIOPEN_VERSION_MAJOR == 3 && \
   (MIOPEN_VERSION_MINOR > 5 || (MIOPEN_VERSION_MINOR == 5 && MIOPEN_VERSION_PATCH >= 2))))

// MIOpen 3.6.1 (ROCm 10.1) fixed deterministic perf-config enumeration for the
// grouped xdlops solvers; see ROCm/rocm-libraries#10240 (ALMIOPEN-2359).
#define MIOPEN_HAS_DETERMINISTIC_TF32 \
  (MIOPEN_VERSION_MAJOR > 3 || (MIOPEN_VERSION_MAJOR == 3 && \
   (MIOPEN_VERSION_MINOR > 6 || (MIOPEN_VERSION_MINOR == 6 && MIOPEN_VERSION_PATCH >= 1))))

#if MIOPEN_VERSION_MAJOR > 3 || (MIOPEN_VERSION_MAJOR == 3 && MIOPEN_VERSION_MINOR >= 4)
// miopen 3.4 moved find mode from private header to public header
#else
// from miopen_internal.h
extern "C" {

typedef enum
{
    miopenConvolutionFindModeNormal        = 1, /*!< Normal mode */
} miopenConvolutionFindMode_t;

miopenStatus_t miopenSetConvolutionFindMode(
    miopenConvolutionDescriptor_t convDesc,
    miopenConvolutionFindMode_t findMode);
}
#endif
