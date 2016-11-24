#pragma once

#include "../common/CommandChannel.hpp"

#include <memory>

namespace thd {
namespace master {

extern std::unique_ptr<MasterCommandChannel> masterCommandChannel;
extern uint64_t tensorCount;

} // namespace master
} // namespace thd
