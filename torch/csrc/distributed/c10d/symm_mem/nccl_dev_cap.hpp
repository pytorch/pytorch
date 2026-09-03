#pragma once

#if USE_NCCL

#include <nccl.h>
#include <torch/csrc/cuda/nccl.h>

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 27, 0)
#define NCCL_HAS_SYMMEM_SUPPORT
#endif

// 2.28.4 is the first release with the usable symmetric-memory device API: the
// device-side LSA barrier (ncclLsaBarrierSession) landed in 2.28.4, alongside
// ncclGetLsaPointer and the device communicator. Earlier 2.28.x ship an
// incomplete nccl_device.h, so gate on 2.28.4.
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 4)
#if !defined(USE_ROCM)
#define NCCL_HAS_SYMMEM_DEVICE_SUPPORT
#include <nccl_device.h>
#endif
#endif

#if defined(NCCL_HAS_SYMMEM_DEVICE_SUPPORT) && \
    NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
#define NCCL_HAS_ONE_SIDED_API
#endif

#if defined(NCCL_HAS_SYMMEM_DEVICE_SUPPORT) && \
    NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 7)
#define NCCL_DEVICE_HAS_REDUCE_COPY
#endif
#endif // USE_NCCL
