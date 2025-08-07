#pragma once

namespace c10d::symmetric_memory {

constexpr size_t signal_pad_size = 2048;

#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
using HandleType = CUmemGenericAllocationHandle;
#elif defined(USE_ROCM)
using HandleType = hipMemGenericAllocationHandle_t;
#else
using HandleType = void*;
#endif

} // namespace c10d::symmetric_memory
