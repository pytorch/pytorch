#pragma once

#include "../../base/DataChannel.h"
#include "../../base/ChannelUtils.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace thd {
namespace master {

struct WorkerState {
  WorkerState() : copy_mutex() {}

  std::mutex copy_mutex;
};

struct THDState {
  static std::vector<WorkerState> s_workers;
  thread_local static rank_type s_current_worker;
  static std::uint64_t s_nextId;
};

} // namespace master
} // namespace thd
