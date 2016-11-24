#include "State.hpp"

#include <cstddef>

namespace thd {
namespace master {

thread_local size_t THDState::s_current_worker = 1;

} // namespace master
} // namespace thd
