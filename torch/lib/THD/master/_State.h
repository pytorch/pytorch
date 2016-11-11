#pragma once

#include <vector>

namespace thd {
namespace master {

struct WorkerState {
};

struct THDState {
  static std::vector<WorkerState> s_workers;
  thread_local static size_t s_current_worker;
};

} // namespace master
} // namespace thd
