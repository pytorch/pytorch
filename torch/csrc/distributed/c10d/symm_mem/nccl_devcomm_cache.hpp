#pragma once

#include <c10/core/Device.h>
#include <string>

namespace c10d::symmetric_memory {

// Cache of NCCL/RCCL device communicators (ncclDevComm) keyed by
// (device, group_name, key). Implemented in NCCLSymmetricMemory.cu, the one
// TU where <nccl_device.h> is includable: RCCL's version instantiates device
// builtins that do not exist in host-only compiles, so the type cannot leak
// into headers consumed by ProcessGroupNCCL.cpp and friends.
//
// Entries are erased when the owning process group tears down
// (release_nccl_devcomms_for_group), so a recreated group can never observe
// a communicator built for its predecessor. Erased entries are reclaimed by
// the communicator itself; survivors are destroyed at process exit.
//
// Returns a pointer to a cache-owned ncclDevComm; callers compiled against
// <nccl_device.h> cast it to ncclDevComm*.
void* get_or_create_nccl_devcomm(
    const c10::Device& device,
    const std::string& group_name,
    const std::string& key,
    int lsa_barrier_count,
    bool lsa_multimem);

void release_nccl_devcomms_for_group(
    const c10::Device& device,
    const std::string& group_name);

} // namespace c10d::symmetric_memory
