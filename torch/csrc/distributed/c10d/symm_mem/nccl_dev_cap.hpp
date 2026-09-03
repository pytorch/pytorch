#pragma once

#if USE_NCCL

#include <nccl.h>
#include <torch/csrc/cuda/nccl.h>

// RCCL symmetric memory requires the 2.30.4 API and an nccl_device.h that host
// translation units can compile. Header presence and version macros alone do
// not establish host compatibility, so CMake probes the installed header.
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 27, 0) &&   \
    (!defined(USE_ROCM) ||                           \
     (NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 4) && \
      defined(RCCL_DEVICE_HEADER_HOST_COMPATIBLE)))
#define NCCL_HAS_SYMMEM_SUPPORT
#endif

// 2.28.4 is the first release with the usable symmetric-memory device API: the
// device-side LSA barrier (ncclLsaBarrierSession) landed in 2.28.4, alongside
// ncclGetLsaPointer and the device communicator. Earlier 2.28.x ship an
// incomplete nccl_device.h, so gate on 2.28.4.
#if defined(NCCL_HAS_SYMMEM_SUPPORT) && \
    NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 4)
#define NCCL_HAS_SYMMEM_DEVICE_SUPPORT
#include <nccl_device.h>
#endif

// Host-side device-communicator setup was added in NCCL 2.29.
#if defined(NCCL_HAS_SYMMEM_DEVICE_SUPPORT) && \
    NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
#define NCCL_HAS_DEVCOMM
#define NCCL_HAS_ONE_SIDED_API
#endif

// Device-side reduce/copy APIs were completed in NCCL 2.29.7.
#if defined(NCCL_HAS_SYMMEM_DEVICE_SUPPORT) && \
    NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 7)
#define NCCL_DEVICE_HAS_REDUCE_COPY
#endif

// Host-side CFT (Compute Fabric Transport) logical-endpoint queries:
// ncclGetPeerDeviceLeInfo / ncclGetMultimemDeviceLeInfo. They resolve a window
// offset into the `(leId, leOffset)` pair that the device-side `ncclCft`
// put/get/red family consumes, so a custom kernel can drive CFT without
// building a ncclDevComm. The LEs only exist if the communicator was created
// with `ncclConfig_t::hostCftMode` enabled (see NCCL_HAS_HOST_CFT_MODE).
#if defined(NCCL_HAS_SYMMEM_DEVICE_SUPPORT) && \
    NCCL_VERSION_CODE >= NCCL_VERSION(2, 31, 2)
#define NCCL_HAS_HOST_CFT
#endif
#endif // USE_NCCL
