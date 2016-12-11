#pragma once

#include "../common/CommandChannel.hpp"
#include "../../base/DataChannel.hpp"

#include <memory>

namespace thd {
namespace master {

extern std::unique_ptr<MasterCommandChannel> masterCommandChannel;
extern uint64_t nextTensorId;   //TODO wrap it somehow to auto-increment

} // namespace master
} // namespace thd
