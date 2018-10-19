#pragma once

#include <ATen/ATen.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace c10d {
class ProcessGroup;
} // namespace c10d

namespace c10d {
void distBroadcastCoalesced(
    std::vector<at::Tensor>& tensors,
    int64_t bufferSize,
    ProcessGroup& processGroup);

void syncParams(
    ProcessGroup& processGroup,
    std::vector<std::vector<at::Tensor>>& parameterData,
    std::vector<std::vector<at::Tensor>>& bufferData,
    const std::vector<int64_t>& devices,
    int64_t broadcastBucketSize,
    bool broadcastBuffers);
} // namespace c10d
