#pragma once

#include "base/channels/data_channel/DataChannel.hpp"
#include "General.h"
#include <memory>

namespace thd {
extern std::unique_ptr<DataChannel> dataChannel;
} // namespace thd
