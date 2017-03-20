#include "State.hpp"

#include <cstddef>
#include <cstdint>

namespace thd {
namespace master {

thread_local size_t THDState::s_current_worker = 1;
std::uint64_t THDState::s_nextId = 0;
std::string THDState::s_error = "";

} // namespace master
} // namespace thd
