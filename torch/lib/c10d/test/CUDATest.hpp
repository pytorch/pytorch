#pragma once

#include <ATen/cuda/CUDAStream.h>

namespace c10d {
namespace test {

void cudaSleep(at::cuda::CUDAStream& stream, uint64_t clocks);

int cudaNumDevices();

} // namespace test
} // namespace c10d
