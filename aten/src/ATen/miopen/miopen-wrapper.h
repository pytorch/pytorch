#pragma once

// Enable MIOpen Beta APIs including miopenSetTensorDescriptorV2 which supports
// 64-bit tensor dimensions/strides for large tensors (numel > INT32_MAX).
// Reference: https://github.com/ROCm/MIOpen/pull/2838
#ifndef MIOPEN_BETA_API
#define MIOPEN_BETA_API 1
#endif

#include <miopen/miopen.h>
#include <miopen/version.h>

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
