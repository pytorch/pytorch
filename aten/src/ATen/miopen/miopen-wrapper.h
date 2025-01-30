#pragma once

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
