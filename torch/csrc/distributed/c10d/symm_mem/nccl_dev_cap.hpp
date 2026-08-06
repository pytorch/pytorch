#pragma once

#if USE_NCCL

#include <nccl.h>
#include <torch/csrc/cuda/nccl.h>

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 27, 0)
#define NCCL_HAS_SYMMEM_SUPPORT
#endif

// ROCm/RCCL exposes the device-side LSA peer-pointer helper (ncclGetLsaPointer
// in <nccl_device.h>) but not the host-side ncclDevComm*/ncclGetPeerDevicePointer
// APIs that gate NCCL_HAS_SYMMEM_DEVICE_SUPPORT. Enable a device-kernel-based
// peer-pointer resolution path for symmetric memory on ROCm so intra-node
// symm-mem kernels (e.g. one_shot_all_reduce) get valid peer pointers.
#if defined(USE_ROCM) && defined(NCCL_HAS_SYMMEM_SUPPORT)
#define NCCL_HAS_LSA_PEER_PTR
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

// Host-side device-communicator setup: ncclDevCommCreate together with
// ncclDevCommRequirements / NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER. These land
// in NCCL 2.29 (verified: absent in 2.28.9, present in 2.29.2), later than the
// device-side kernel symbols, so ops that construct a ncclDevComm gate on this
// rather than NCCL_HAS_SYMMEM_DEVICE_SUPPORT.
#if defined(NCCL_HAS_SYMMEM_DEVICE_SUPPORT) && \
    NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
#define NCCL_HAS_DEVCOMM
#endif

#if (defined(NCCL_HAS_SYMMEM_DEVICE_SUPPORT) || defined(USE_ROCM)) && \
    NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
#define NCCL_HAS_ONE_SIDED_API
#endif

// Device-side reduce/copy API (ncclLsaReduceSum, ncclMultimemReduceSum,
// ncclLsaBarrierSession). On ROCm this is enabled via NCCL_HAS_LSA_PEER_PTR:
// RCCL >= 2.29.7 ships these device symbols and exports the host-side
// ncclDevCommCreate/Destroy in librccl.so, so the reduce-scatter/all-to-all
// kernels can be compiled with hipcc (see nccl_reduce_scatter_offset.cu for the
// HIP-specific shims required to include <nccl_device.h>).
#if (defined(NCCL_HAS_SYMMEM_DEVICE_SUPPORT) || defined(NCCL_HAS_LSA_PEER_PTR)) && \
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
