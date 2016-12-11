#pragma once

#include "../common/CommandChannel.hpp"
#include "../../base/DataChannel.hpp"
#include "../../base/Storage.hpp"
#include "../../base/Tensor.hpp"

#include <memory>

namespace thd { namespace worker {
extern std::unique_ptr<WorkerCommandChannel> workerCommandChannel;
extern std::unordered_map<object_id_type, std::unique_ptr<Tensor>> workerTensors;
extern std::unordered_map<object_id_type, std::unique_ptr<Storage>> workerStorages;
}} // namespace worker, thd
