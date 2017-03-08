#pragma once

#include "../../base/channels/command_channel/CommandChannel.hpp"
#include "../../base/channels/data_channel/DataChannel.hpp"

#include <memory>

namespace thd {
namespace master {

extern std::unique_ptr<MasterCommandChannel> masterCommandChannel;
extern uint64_t nextTensorId;   //TODO wrap it somehow to auto-increment

} // namespace master
} // namespace thd
