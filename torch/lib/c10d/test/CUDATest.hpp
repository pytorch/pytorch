#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "../CUDAUtils.hpp"

namespace c10d {
namespace test {

void cudaSleep(CUDAStream& stream, size_t clocks);

int cudaNumDevices();

} // namespace test
} // namespace c10d
