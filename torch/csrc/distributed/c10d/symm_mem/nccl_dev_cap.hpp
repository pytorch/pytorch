#pragma once

#if USE_NCCL

#include <nccl.h>
#include <torch/csrc/cuda/nccl.h>

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 27, 0)
#define NCCL_HAS_SYMMEM_SUPPORT
#endif

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
#if !defined(USE_ROCM)
#define NCCL_HAS_SYMMEM_DEVICE_SUPPORT
#include <nccl_device.h>
#endif
#endif

// Host-side device-communicator setup: ncclDevCommCreate together with
// ncclDevCommRequirements / NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER. These land
// in NCCL 2.29 (verified: absent in 2.28.9, present in 2.29.2), later than the
// device-side kernel symbols, so ops that construct a ncclDevComm gate on this
// rather than NCCL_HAS_SYMMEM_DEVICE_SUPPORT.
#if defined(NCCL_HAS_SYMMEM_DEVICE_SUPPORT) && \
    NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
#define NCCL_HAS_DEVCOMM
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
