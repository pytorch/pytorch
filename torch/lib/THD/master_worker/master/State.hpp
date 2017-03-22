#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace thd {
namespace master {

struct WorkerState {
};

struct THDState {
  static std::vector<WorkerState> s_workers;
  thread_local static std::size_t s_current_worker;
  static std::uint64_t s_nextId;
};

} // namespace master
} // namespace thd
