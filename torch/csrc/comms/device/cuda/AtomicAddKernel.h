// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy

#include <cstdint>

namespace torch::comms {

cudaError_t launchAtomicAdd(
    cudaStream_t stream,
    uint64_t* d_counter,
    uint64_t amount = 1ULL);

} // namespace torch::comms
