#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "../CUDAUtils.hpp"

namespace c10d {
namespace test {

void cudaSleep(CUDAStream& stream, uint64_t clocks);

int cudaNumDevices();

} // namespace test
} // namespace c10d
