#pragma once

// This file contains utility functions common for CUDA, which can be used by
// ProcessGroupNCCL or SymmetricMemory.

namespace c10d::cuda {

bool deviceSupportsMulticast(int device_idx);

} // namespace c10d::cuda
