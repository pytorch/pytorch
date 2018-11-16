#pragma once

#include <c10d/ProcessGroup.hpp>

#include <ATen/ATen.h>
#include "c10/util/Optional.h"

#include <cstddef>
#include <memory>
#include <tuple>
#include <vector>

namespace c10d {

std::vector<std::vector<at::Tensor>> bucketTensors(
    std::vector<at::Tensor>& tensors,
    int64_t bucketSize,
    bool fineGrained = false);

void distBroadcastCoalesced(
    ProcessGroup& processGroup,
    std::vector<at::Tensor>& tensors,
    int64_t bufferSize,
    bool fineGrained = false);

void syncParams(
    ProcessGroup& processGroup,
    std::vector<std::vector<at::Tensor>>& parameterData,
    std::vector<std::vector<at::Tensor>>& bufferData,
    const std::vector<int64_t>& devices,
    int64_t broadcastBucketSize,
    bool broadcastBuffers);

std::tuple<std::shared_ptr<ProcessGroup::Work>, at::Tensor> queueReduction(
    ProcessGroup& processGroup,
    std::vector<std::vector<at::Tensor>>& gradsBatch,
    const std::vector<int64_t>& devices);

void syncReduction(
    std::shared_ptr<ProcessGroup::Work>& reductionWork,
    std::vector<at::Tensor>& gradsBatch,
    at::Tensor& gradsBatchCoalesced);

} // namespace c10d
