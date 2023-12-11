#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

namespace c10d_functional {

void register_process_group(
    const std::string& tag,
    c10::intrusive_ptr<c10d::ProcessGroup> pg);

c10::intrusive_ptr<c10d::ProcessGroup> resolve_process_group(
    const std::string& tag);

} // namespace c10d_functional
