#pragma once

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

namespace c10d {

C10_EXPORT void set_thread_isolation_mode(bool enable);

bool get_thread_isolation_mode();

C10_EXPORT void register_process_group(
    const std::string& group_name,
    c10::intrusive_ptr<c10d::ProcessGroup> group);

C10_EXPORT c10::intrusive_ptr<c10d::ProcessGroup> resolve_process_group(
    const std::string& group_name);

C10_EXPORT void unregister_process_group(const std::string& group_name);

C10_EXPORT void unregister_all_process_groups();

} // namespace c10d
