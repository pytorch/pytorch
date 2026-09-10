#pragma once

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include <functional>
#include <string>

namespace c10d {

C10_EXPORT void set_thread_isolation_mode(bool enable);

bool get_thread_isolation_mode();

C10_EXPORT void register_process_group(
    const std::string& group_name,
    const c10::intrusive_ptr<c10d::ProcessGroup>& group);

C10_EXPORT c10::intrusive_ptr<c10d::ProcessGroup> resolve_process_group(
    const std::string& group_name);

C10_EXPORT bool is_process_group_registered(const std::string& group_name);

// Runs when a process group is unregistered, with that group's name. Lets a
// consumer holding per-group state drop it at the moment the name stops
// referring to that group, rather than discovering later that it does not.
//
// A hook rather than a direct call because this file builds without CUDA and
// the consumers do not: symmetric memory's counters live in a CUDA-only
// translation unit. Collapsing this into a call would need an ifdef here.
C10_EXPORT void register_group_unregister_hook(
    std::function<void(const std::string&)> hook);

C10_EXPORT void unregister_process_group(const std::string& group_name);

C10_EXPORT void unregister_all_process_groups();

} // namespace c10d
