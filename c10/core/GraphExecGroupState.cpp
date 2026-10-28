#include <c10/core/GraphExecGroupState.h>

namespace c10 {

namespace {

thread_local std::optional<SafePyObject> graph_exec_group_tls;

} // namespace

const std::optional<SafePyObject>& GraphExecGroupState::get_tls_state() {
  return graph_exec_group_tls;
}

void GraphExecGroupState::set_tls_state(
    const std::optional<SafePyObject>& state) {
  graph_exec_group_tls = state;
}

} // namespace c10
