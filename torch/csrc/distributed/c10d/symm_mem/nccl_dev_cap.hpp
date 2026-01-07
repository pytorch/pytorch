#pragma once

#if USE_NCCL

#include <nccl.h>
#include <torch/csrc/cuda/nccl.h>

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 27, 0)
#define NCCL_HAS_SYMMEM_SUPPORT
#endif

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
#define NCCL_HAS_SYMMEM_DEVICE_SUPPORT
#include <nccl_device.h>
#endif
#endif // USE_NCCL
