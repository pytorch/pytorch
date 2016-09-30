/*************************************************************************
 * Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/


// Dynamically handle dependencies on external libraries (other than cudart).

#ifndef SRC_LIBWRAP_H_
#define SRC_LIBWRAP_H_

#include "core.h"

typedef struct nvmlDevice_st* nvmlDevice_t;

/**
 * Generic enable/disable enum.
 */
typedef enum nvmlEnableState_enum
{
    NVML_FEATURE_DISABLED    = 0,     //!< Feature disabled
    NVML_FEATURE_ENABLED     = 1      //!< Feature enabled
} nvmlEnableState_t;

ncclResult_t wrapSymbols(void);

ncclResult_t wrapNvmlInit(void);
ncclResult_t wrapNvmlShutdown(void);
ncclResult_t wrapNvmlDeviceGetHandleByPciBusId(const char* pciBusId, nvmlDevice_t* device);
ncclResult_t wrapNvmlDeviceGetIndex(nvmlDevice_t device, unsigned* index);
ncclResult_t wrapNvmlDeviceSetCpuAffinity(nvmlDevice_t device);
ncclResult_t wrapNvmlDeviceClearCpuAffinity(nvmlDevice_t device);
ncclResult_t wrapNvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t *device);

#endif // End include guard

