/*************************************************************************
 * Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ************************************************************************/

#include "libwrap.h"
#include <dlfcn.h>
#include "core.h"

typedef enum { SUCCESS = 0 } RetCode;
int symbolsLoaded = 0;

static RetCode (*nvmlInternalInit)(void);
static RetCode (*nvmlInternalShutdown)(void);
static RetCode (*nvmlInternalDeviceGetHandleByPciBusId)(const char* pciBusId, nvmlDevice_t* device);
static RetCode (*nvmlInternalDeviceGetIndex)(nvmlDevice_t device, unsigned* index);
static RetCode (*nvmlInternalDeviceSetCpuAffinity)(nvmlDevice_t device);
static RetCode (*nvmlInternalDeviceClearCpuAffinity)(nvmlDevice_t device);
static const char* (*nvmlInternalErrorString)(RetCode r);

static CUresult (*cuInternalGetErrorString)(CUresult error, const char** pStr);
static CUresult (*cuInternalIpcGetMemHandle)(CUipcMemHandle* pHandle, CUdeviceptr dptr);
static CUresult (*cuInternalIpcOpenMemHandle)(CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int  Flags);
static CUresult (*cuInternalIpcCloseMemHandle)(CUdeviceptr dptr);


ncclResult_t wrapSymbols(void) {

  if (symbolsLoaded)
    return ncclSuccess;
 
  static void* nvmlhandle = NULL;
  static void* cuhandle = NULL;
  void* tmp;
  void** cast;

  nvmlhandle=dlopen("libnvidia-ml.so", RTLD_NOW);
  if (!nvmlhandle) {
    WARN("Failed to open libnvidia-ml.so");
    goto teardown;
  }

  cuhandle = dlopen("libcuda.so", RTLD_NOW);
  if (!cuhandle) {
    WARN("Failed to open libcuda.so");
    goto teardown;
  }

  #define LOAD_SYM(handle, symbol, funcptr) do {         \
    cast = (void**)&funcptr;                             \
    tmp = dlsym(handle, symbol);                         \
    if (tmp == NULL) {                                   \
      WARN("dlsym failed on %s - %s", symbol, dlerror()); \
      goto teardown;                                     \
    }                                                    \
    *cast = tmp;                                         \
  } while (0)

  LOAD_SYM(nvmlhandle, "nvmlInit", nvmlInternalInit);
  LOAD_SYM(nvmlhandle, "nvmlShutdown", nvmlInternalShutdown);
  LOAD_SYM(nvmlhandle, "nvmlDeviceGetHandleByPciBusId", nvmlInternalDeviceGetHandleByPciBusId);
  LOAD_SYM(nvmlhandle, "nvmlDeviceGetIndex", nvmlInternalDeviceGetIndex);
  LOAD_SYM(nvmlhandle, "nvmlDeviceSetCpuAffinity", nvmlInternalDeviceSetCpuAffinity);
  LOAD_SYM(nvmlhandle, "nvmlDeviceClearCpuAffinity", nvmlInternalDeviceClearCpuAffinity);
  LOAD_SYM(nvmlhandle, "nvmlErrorString", nvmlInternalErrorString);

  LOAD_SYM(cuhandle, "cuGetErrorString", cuInternalGetErrorString);
  LOAD_SYM(cuhandle, "cuIpcGetMemHandle", cuInternalIpcGetMemHandle);
  LOAD_SYM(cuhandle, "cuIpcOpenMemHandle", cuInternalIpcOpenMemHandle);
  LOAD_SYM(cuhandle, "cuIpcCloseMemHandle", cuInternalIpcCloseMemHandle);
  
  symbolsLoaded = 1;
  return ncclSuccess;

  teardown:
  nvmlInternalInit = NULL;
  nvmlInternalShutdown = NULL;
  nvmlInternalDeviceGetHandleByPciBusId = NULL;
  nvmlInternalDeviceGetIndex = NULL;
  nvmlInternalDeviceSetCpuAffinity = NULL;
  nvmlInternalDeviceClearCpuAffinity = NULL;
  
  cuInternalGetErrorString = NULL;
  cuInternalIpcGetMemHandle = NULL;
  cuInternalIpcOpenMemHandle = NULL;
  cuInternalIpcCloseMemHandle = NULL;

  if (cuhandle   != NULL) dlclose(cuhandle);
  if (nvmlhandle != NULL) dlclose(nvmlhandle);
  return ncclSystemError;
}


ncclResult_t wrapNvmlInit(void) {
  if (nvmlInternalInit == NULL) {
    WARN("lib wrapper not initilaized.");
    return ncclLibWrapperNotSet;
  }
  RetCode ret = nvmlInternalInit();
  if (ret != SUCCESS) {
    WARN("nvmlInit() failed: %s",
      nvmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrapNvmlShutdown(void) {
  if (nvmlInternalShutdown == NULL) {
    WARN("lib wrapper not initilaized.");
    return ncclLibWrapperNotSet;
  }
  RetCode ret = nvmlInternalShutdown();
  if (ret != SUCCESS) {
    WARN("nvmlShutdown() failed: %s ",
      nvmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrapNvmlDeviceGetHandleByPciBusId(const char* pciBusId, nvmlDevice_t* device) {
  if (nvmlInternalDeviceGetHandleByPciBusId == NULL) {
    WARN("lib wrapper not initilaized.");
    return ncclLibWrapperNotSet;
  }
  RetCode ret = nvmlInternalDeviceGetHandleByPciBusId(pciBusId, device);
  if (ret != SUCCESS) {
    WARN("nvmlDeviceGetHandleByPciBusId() failed: %s ",
      nvmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrapNvmlDeviceGetIndex(nvmlDevice_t device, unsigned* index) {
  if (nvmlInternalDeviceGetIndex == NULL) {
    WARN("lib wrapper not initilaized.");
    return ncclLibWrapperNotSet;
  }
  RetCode ret = nvmlInternalDeviceGetIndex(device, index);
  if (ret != SUCCESS) {
    WARN("nvmlDeviceGetIndex() failed: %s ",
      nvmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrapNvmlDeviceSetCpuAffinity(nvmlDevice_t device) {
  if (nvmlInternalDeviceSetCpuAffinity == NULL) {
    WARN("lib wrapper not initilaized.");
    return ncclLibWrapperNotSet;
  }
  RetCode ret = nvmlInternalDeviceSetCpuAffinity(device);
  if (ret != SUCCESS) {
    WARN("nvmlDeviceSetCpuAffinity() failed: %s ",
      nvmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrapNvmlDeviceClearCpuAffinity(nvmlDevice_t device) {
  if (nvmlInternalInit == NULL) {
    WARN("lib wrapper not initilaized.");
    return ncclLibWrapperNotSet;
  }
  RetCode ret = nvmlInternalDeviceClearCpuAffinity(device);
  if (ret != SUCCESS) {
    WARN("nvmlDeviceClearCpuAffinity() failed: %s ",
      nvmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrapCuIpcGetMemHandle(CUipcMemHandle* pHandle, CUdeviceptr dptr) {
  if (cuInternalIpcGetMemHandle == NULL) {
    WARN("lib wrapper not initilaized.");
    return ncclLibWrapperNotSet;
  }
  CUresult ret = cuInternalIpcGetMemHandle(pHandle, dptr);
  if (ret != CUDA_SUCCESS) {
    const char* reason = NULL;
    cuInternalGetErrorString(ret, &reason);
    if (reason != NULL)
      WARN("cuInternalIpcGetMemHandle() failed: %s ", reason);
    else
      WARN("cuInternalIpcGetMemHandle() failed: %d ", ret);
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrapCuIpcOpenMemHandle(CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int  Flags) {
  if (cuInternalIpcOpenMemHandle == NULL) {
    WARN("lib wrapper not initilaized.");
    return ncclLibWrapperNotSet;
  }
  CUresult ret = cuInternalIpcOpenMemHandle(pdptr, handle, Flags);
  if (ret != CUDA_SUCCESS) {
    const char* reason = NULL;
    cuInternalGetErrorString(ret, &reason);
    if (reason != NULL)
      WARN("cuInternalIpcOpenMemHandle() failed: %s ", reason);
    else
      WARN("cuInternalIpcOpenMemHandle() failed: %d ", ret);
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrapCuIpcCloseMemHandle(CUdeviceptr dptr) {
  if (cuInternalIpcCloseMemHandle == NULL) {
    WARN("lib wrapper not initilaized.");
    return ncclLibWrapperNotSet;
  }
  CUresult ret = cuInternalIpcCloseMemHandle(dptr);
  if (ret != CUDA_SUCCESS) {
    const char* reason = NULL;
    cuInternalGetErrorString(ret, &reason);
    if (reason != NULL)
      WARN("cuInternalIpcCloseMemHandle() failed: %s ", reason);
    else
      WARN("cuInternalIpcCloseMemHandle() failed: %d ", ret);
    return ncclSystemError;
  }
  return ncclSuccess;
}

