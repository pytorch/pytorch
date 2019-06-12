#pragma once

#include <memory>
#include "General.h"
#include "base/DataChannel.hpp"

namespace thd {
extern std::unique_ptr<DataChannel> dataChannel;
} // namespace thd
