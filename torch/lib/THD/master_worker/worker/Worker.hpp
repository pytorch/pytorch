#pragma once

#include "../common/CommandChannel.hpp"

#include <memory>

namespace thd { namespace worker {
extern std::unique_ptr<WorkerCommandChannel> workerCommandChannel;
}} // namespace worker, thd
