#pragma once

// Enable MIOpen Beta APIs including miopenSetTensorDescriptorV2 which supports
// 64-bit tensor dimensions/strides for large tensors (numel > INT32_MAX).
// Reference: https://github.com/ROCm/MIOpen/pull/2838
#ifndef MIOPEN_BETA_API
#define MIOPEN_BETA_API 1
#endif

#include <miopen/miopen.h>
#include <miopen/version.h>

// miopenMathType_t exists in ROCm 7.12 (MIOpen 3.5.1) and is absent from
// ROCm 7.11, which reports the same MIOpen version. Do not test for 3.5.1.
// Requesting TF32 before MIOpen 3.6.2 misses the tuned fp32 database and
// can fall through to the naive kernel. hipconv TF32 kernels landed in 3.6.2
// for gfx950 and gfx1250. gfx942 is not that path.
#define MIOPEN_HAS_TF32_CONV \
  (MIOPEN_VERSION_MAJOR > 3 || (MIOPEN_VERSION_MAJOR == 3 && \
   (MIOPEN_VERSION_MINOR > 6 || (MIOPEN_VERSION_MINOR == 6 && MIOPEN_VERSION_PATCH >= 2))))

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
