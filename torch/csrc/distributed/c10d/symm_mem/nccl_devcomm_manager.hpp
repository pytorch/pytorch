#pragma once

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

#ifdef NCCL_HAS_SYMMEM_DEVICE_SUPPORT

namespace c10d::symmetric_memory {

// Manages NCCL device communicators for symmetric memory operations.
// This is a singleton class that maintains a registry of device communicators
// organized by process group name and an optional key (typically the caller
// function name). This allows different functions within the same process group
// to use different device communicators, which is useful for concurrent
// collective operations.
//
// The registry uses a two-level map structure:
// - First level: keyed by process group name
// - Second level: keyed by an optional key (defaults to caller function name)
//
// Device communicators are stored by value in the registry, but methods return
// references wrapped in std::optional for safe access.
class NCCLDevCommManager {
 public:
  // Constructor
  // @param device The CUDA device this manager is associated with
  explicit NCCLDevCommManager(const c10::Device device) : device_(device) {}

  // Get the singleton instance for the given device.
  // This ensures there's only one manager per device. If called with a
  // different device than the one used to create the singleton, it will throw.
  // @param device The CUDA device to get the manager for
  // @return Reference to the singleton manager instance
  static NCCLDevCommManager& get(const c10::Device device) {
    static NCCLDevCommManager manager(device);
    TORCH_CHECK_VALUE(
        manager.device_ == device,
        "Detected use of NCCLDevCommManager on multiple devices. This is not supported.");
    return manager;
  }

  // Get an NCCL device communicator for a group, for the caller function.  By
  // default, we search for the device communicator using the caller function
  // name as the key.  If you previously registered a device communicator with a
  // different key, you should provide that key instead.
  // Returns std::nullopt if the device communicator is not found.
  // Example:
  // void foo(const std::string& group_name) {
  //   // Try to get first.
  //   auto devcomm_opt = get_devcomm(group_name);
  //   if (!devcomm_opt) {
  //     // Not found, create then register.
  //     ncclDevComm devcomm = ncclDevCommCreate(...);
  //     devcomm_opt = register_devcomm(group_name, devcomm);
  //   }
  //   ncclDevComm& devcomm_ref = *devcomm_opt;
  //   // Use devcomm_ref
  // }
  std::optional<std::reference_wrapper<ncclDevComm>> get_devcomm(
      const std::string& group_name,
      const std::string& key = __builtin_FUNCTION()) {
    std::lock_guard<std::mutex> lock(mutex_);
    // First, look up the group in the registry
    auto group_it = devcomm_registry_.find(group_name);
    if (group_it == devcomm_registry_.end()) {
      return std::nullopt;
    }
    // Then, look up the key within that group's map
    auto key_it = group_it->second.find(key);
    if (key_it == group_it->second.end()) {
      return std::nullopt;
    }
    // Return a reference wrapper to the device communicator
    // Using reference_wrapper because std::optional cannot hold references
    // directly
    return std::make_optional(std::ref(key_it->second));
  }

  // Get a host-side NCCL communicator for a group.
  // This is the regular host-side communicator, not the device communicator.
  // @param group_name The process group name
  // @return The host-side NCCL communicator
  // @throws TORCH_CHECK if the communicator is not found
  ncclComm_t get_comm(const std::string& group_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = group_to_comm_.find(group_name);
    if (it == group_to_comm_.end()) {
      TORCH_CHECK(
          false,
          "NCCL host communicator for group ",
          group_name,
          " not found. Have you rendezvoused any tensor with this group?");
    }
    return it->second;
  }

  // Register a device communicator for a group. If `key` is not
  // specified, we use the caller function name as the default `key`, to
  // distinguish between different collective functions within the same group.
  // You can provide your own `key` if your function uses two different
  // device communicators on the same group at the same time, for example,
  // when concurrent collective operations are used.
  // Returns a reference to the newly registered device communicator.
  // @throws TORCH_CHECK if the device communicator is already registered for
  //         the given group and key combination.
  // Example:
  // void foo(const std::string& group_name) {
  //   // Try to get first.
  //   auto devcomm_opt = get_devcomm(group_name);
  //   if (!devcomm_opt) {
  //     // Not found, create then register.
  //     ncclDevComm devcomm = ncclDevCommCreate(...);
  //     devcomm_opt = register_devcomm(group_name, devcomm);
  //   }
  //   ncclDevComm& devcomm_ref = *devcomm_opt;
  //   // Use devcomm_ref
  // }
  // void bar(const std::string& group_name) {
  //   ncclDevComm devcomm0 = ncclDevCommCreate(...);
  //   ncclDevComm devcomm1 = ncclDevCommCreate(...);
  //   // You can provide your own `key` if you want to, for example, to
  //   // distinguish between concurrent collective operations.
  //   register_devcomm(group_name, devcomm0, "bar0");
  //   register_devcomm(group_name, devcomm1, "bar1");
  // }
  std::optional<std::reference_wrapper<ncclDevComm>> register_devcomm(
      const std::string& group_name,
      ncclDevComm devcomm,
      const std::string& key = __builtin_FUNCTION()) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Ensure the group exists in the registry, creating an empty map if needed
    auto [group_it, inserted] = devcomm_registry_.try_emplace(
        group_name, std::unordered_map<std::string, ncclDevComm>());
    auto& group_map = group_it->second;
    // Try to insert the device communicator with the given key
    // Use std::move to avoid copying the device communicator
    auto [key_it, key_inserted] =
        group_map.try_emplace(key, std::move(devcomm));
    if (!key_inserted) {
      // Already registered - this is a programming error, so throw
      TORCH_CHECK(
          false,
          "NCCL device communicator for group ",
          group_name,
          " with key ",
          key,
          " already registered.");
    }
    // Return a reference to the newly registered device communicator
    return std::make_optional(std::ref(key_it->second));
  }

  // Register a host-side NCCL communicator for a group.
  // This should be called before registering any device communicators for the
  // same group, as device communicators need the host communicator for cleanup.
  // @param group_name The process group name
  // @param comm The host-side NCCL communicator to register
  // @throws TORCH_CHECK if the group is already registered with a different
  // communicator.
  void register_comm(const std::string& group_name, ncclComm_t comm) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto [it, inserted] = group_to_comm_.try_emplace(group_name, comm);
    // If the communicator is already registered, check if it is the same one.
    // If not, throw an error.
    TORCH_CHECK(
        inserted || it->second == comm, // this is just a pointer comparison
        "NCCL host communicator for group ",
        group_name,
        " already registered.");
  }

  // Destructor: Clean up all registered device communicators.
  // This is a best-effort cleanup. If the CUDA context has already been
  // destroyed, the cleanup will be skipped. All errors are caught and ignored
  // to prevent exceptions from propagating during destruction.
  ~NCCLDevCommManager() noexcept {
    // Best effort to destroy the device communicators. Skip if CUDA context has
    // exited.
    try {
      c10::cuda::CUDAGuard guard(device_);
      // Make sure all kernels have completed before destroying the device
      // communicator. This is important to ensure no kernels are still using
      // the device communicator when we destroy it.
      C10_CUDA_CHECK(cudaDeviceSynchronize());
      // Iterate through all groups and their device communicators
      for (auto& [group_name, group_map] : devcomm_registry_) {
        // Find the host communicator for the group.
        // Device communicators need the host communicator for destruction.
        auto comm_it = group_to_comm_.find(group_name);
        if (comm_it != group_to_comm_.end()) {
          // Destroy each device communicator in this group
          for (auto& [_, devcomm] : group_map) {
            // Destroy the device communicator using the host communicator
            ncclDevCommDestroy(comm_it->second, &devcomm);
          }
        }
      }
    } catch (...) {
      // Ignore the error - we're in a destructor and can't throw
      // Log a warning for debugging purposes
      LOG(WARNING)
          << "Failed to destroy the NCCL device communicator, skipping";
    }
  }

 private:
  // Device where the NCCL device communicator manager is created.
  // The manager is device-specific and cannot be used across multiple devices.
  const c10::Device device_;

  // Mutex to protect the registry maps.
  std::mutex mutex_;

  // A map from process group name to the host-side NCCL communicator.
  // The host communicator is required for creating and destroying device
  // communicators. It should be registered before any device communicators
  // for the same group.
  std::unordered_map<std::string, ncclComm_t> group_to_comm_;

  // A two-level map for device communicators:
  // - First level: keyed by process group name
  // - Second level: keyed by an optional key (defaults to caller function name
  //   via __builtin_FUNCTION())
  //
  // This structure allows multiple device communicators per process group,
  // which is useful when different functions need separate device communicators
  // for concurrent operations. The key defaults to the caller's function name,
  // but can be customized for cases where a single function needs multiple
  // device communicators.
  std::unordered_map<std::string, std::unordered_map<std::string, ncclDevComm>>
      devcomm_registry_;
};

} // namespace c10d::symmetric_memory
#endif // NCCL_HAS_SYMMEM_DEVICE_SUPPORT
