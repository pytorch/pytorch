#include "State.hpp"

#include <cstddef>
#include <cstdint>

namespace thd {
namespace master {

std::vector<WorkerState> THDState::s_workers;
thread_local rank_type THDState::s_current_worker = 1;
std::uint64_t THDState::s_nextId = 0;

} // namespace master
} // namespace thd
