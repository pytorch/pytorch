#include <c10/core/CopyBytes.h>

#include "caffe2/core/hip/context_gpu.h"

namespace at {
REGISTER_COPY_BYTES_FUNCTION(
    DeviceType::CUDA,
    DeviceType::CUDA,
    caffe2::HIPContext::CopyBytesSync,
    caffe2::HIPContext::CopyBytesAsync);

REGISTER_COPY_BYTES_FUNCTION(
    DeviceType::CUDA,
    DeviceType::CPU,
    caffe2::HIPContext::CopyBytesSync,
    caffe2::HIPContext::CopyBytesAsync);

REGISTER_COPY_BYTES_FUNCTION(
    DeviceType::CPU,
    DeviceType::CUDA,
    caffe2::HIPContext::CopyBytesSync,
    caffe2::HIPContext::CopyBytesAsync);
} // namespace at
