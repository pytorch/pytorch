#pragma once

#include <memory>
#include <THD/process_group/General.h>
#include <THD/base/DataChannel.hpp>

namespace thd {
extern std::unique_ptr<DataChannel> dataChannel;
} // namespace thd
