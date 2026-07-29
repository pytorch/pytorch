// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Installs the NCCL comm-registration hooks (see NCCLCommRegistrationHook.hpp)
// so that a backend which publishes a host ncclComm_t -- e.g. the nccl2
// ProcessGroupNCCL -- lands it in NCCLDevCommManager for symmetric memory,
// without the backend having to depend on the manager header. This is the one
// place the opaque void* comm from the hook is cast back to ncclComm_t.
//
// Gated on NCCL_HAS_SYMMEM_SUPPORT (via nccl_dev_cap.hpp), matching the other
// symm_mem NCCL translation units: the macro is only defined when NCCL is built
// with symmetric-memory support, so a build without it compiles this to an
// empty TU and producers' publish/retire calls stay no-ops.

#include <torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp>

#ifdef NCCL_HAS_SYMMEM_SUPPORT

#include <torch/csrc/distributed/c10d/NCCLCommRegistrationHook.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_devcomm_manager.hpp>

namespace c10d::symmetric_memory {
namespace {

// Installed at load time (before any process group is created).
const bool kNcclCommRegistrationHooksInstalled = [] {
  ::c10d::setNCCLCommRegistrationHooks(
      {// on_register
       [](const std::string& group_name, void* comm, c10::Device device) {
         NCCLDevCommManager::get(device).register_comm(
             group_name, reinterpret_cast<ncclComm_t>(comm));
       },
       // on_unregister
       [](const std::string& group_name, void* comm, c10::Device device) {
         NCCLDevCommManager::get(device).unregister_comm(
             group_name, reinterpret_cast<ncclComm_t>(comm));
       }});
  return true;
}();

} // namespace
} // namespace c10d::symmetric_memory

#endif // NCCL_HAS_SYMMEM_SUPPORT
