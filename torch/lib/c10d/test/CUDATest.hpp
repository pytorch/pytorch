#pragma once

#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>

namespace c10d {
namespace test {

void cudaSleep(at::hip::HIPStreamMasqueradingAsCUDA& stream, uint64_t clocks);

int cudaNumDevices();

} // namespace test
} // namespace c10d
