#pragma once

#include "../common/CommandChannel.hpp"
#include "../../base/Tensor.hpp"

#include <memory>

namespace thd { namespace worker {
extern std::unique_ptr<WorkerCommandChannel> workerCommandChannel;
extern std::unordered_map<unsigned long long, std::unique_ptr<Tensor>> workerTensors;
}} // namespace worker, thd
