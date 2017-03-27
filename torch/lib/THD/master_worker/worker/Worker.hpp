#pragma once

#include "../common/CommandChannel.hpp"
#include "../../base/DataChannel.hpp"

#include <THPP/Storage.hpp>
#include <THPP/Tensor.hpp>

#include <memory>

namespace thd { namespace worker {
extern std::unique_ptr<WorkerCommandChannel> workerCommandChannel;
extern std::unordered_map<object_id_type, std::unique_ptr<thpp::Tensor>>
  workerTensors;
extern std::unordered_map<object_id_type, std::unique_ptr<thpp::Storage>>
  workerStorages;
}} // namespace worker, thd
