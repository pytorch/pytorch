#pragma once

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#ifdef USE_ROCM
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/util/env.h>
#endif
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp>
#ifdef USE_ROCM
#include <cstdint>
#endif
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

#ifdef NCCL_HAS_SYMMEM_SUPPORT

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
class TORCH_API NCCLDevCommManager {
 public:
  // Constructor
  // @param device The CUDA device this manager is associated with
  explicit NCCLDevCommManager(const c10::Device device) : device_(device) {}

  // Per-device singleton: Defined out-of-line in nccl_devcomm_manager.cpp
  // (libtorch_cuda) so the function-local registry is process-wide. An inline
  // definition is hidden by
  // `-fvisibility-inlines-hidden`, so a separately linked DSO (e.g.
  // torch._nccl_ep) gets its own empty map and cannot see ProcessGroup / hook
  // registration.
  static NCCLDevCommManager& get(const c10::Device device);

#ifdef NCCL_HAS_SYMMEM_DEVICE_SUPPORT
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
#endif // NCCL_HAS_SYMMEM_DEVICE_SUPPORT

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

#ifdef USE_ROCM
  // Non-throwing lookup for ROCm teardown paths, where the owning process group
  // may already have removed or replaced its communicator.
  std::optional<ncclComm_t> find_comm(const std::string& group_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = group_to_comm_.find(group_name);
    if (it == group_to_comm_.end()) {
      return std::nullopt;
    }
    return it->second;
  }

  uint64_t get_comm_generation(const std::string& group_name, ncclComm_t comm) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto comm_it = group_to_comm_.find(group_name);
    auto generation_it = group_to_comm_generation_.find(group_name);
    TORCH_CHECK(
        comm_it != group_to_comm_.end() && comm_it->second == comm &&
            generation_it != group_to_comm_generation_.end(),
        "ROCm NCCL communicator registration changed while symmetric memory "
        "was rendezvousing group '",
        group_name,
        "'");
    return generation_it->second;
  }

  bool comm_registration_is_live(
      const std::string& group_name,
      ncclComm_t comm,
      uint64_t generation) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto comm_it = group_to_comm_.find(group_name);
    auto generation_it = group_to_comm_generation_.find(group_name);
    return comm_it != group_to_comm_.end() && comm_it->second == comm &&
        generation_it != group_to_comm_generation_.end() &&
        generation_it->second == generation;
  }

  bool comm_has_device_api_support(
      const std::string& group_name,
      ncclComm_t comm) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto comm_it = group_to_comm_.find(group_name);
    auto support_it = group_to_device_api_support_.find(group_name);
    return comm_it != group_to_comm_.end() && comm_it->second == comm &&
        support_it != group_to_device_api_support_.end() && support_it->second;
  }

  bool capture_allocation_supported() {
    if (c10::utils::check_env("TORCH_NCCL_SYMM_MEM_DISABLE_CAPTURE_ALLOC") ==
        true) {
      return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (capture_setup_stream_ == nullptr) {
      return false;
    }
    for (const auto& [_, supported] : group_to_capture_allocation_support_) {
      if (supported) {
        return true;
      }
    }
    return false;
  }

  std::mutex& capture_setup_mutex() {
    return capture_setup_mutex_;
  }

  cudaStream_t capture_setup_stream() {
    std::lock_guard<std::mutex> lock(mutex_);
    TORCH_CHECK(
        capture_setup_stream_ != nullptr,
        "ROCm NCCL symmetric-memory capture setup stream is unavailable");
    return capture_setup_stream_;
  }
#endif

#ifdef NCCL_HAS_SYMMEM_DEVICE_SUPPORT
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
#endif // NCCL_HAS_SYMMEM_DEVICE_SUPPORT

  // Register the host-side NCCL communicator for `group_name` on this
  // manager's device. Last-write-wins so a successor PG can replace the
  // entry before the prior PG's destructor runs (e.g. restart-after-error).
  // Producers must retire their entry before invalidating the communicator;
  // destructor cleanup remains a fallback.
  void register_comm(const std::string& group_name, ncclComm_t comm) {
    std::lock_guard<std::mutex> lock(mutex_);
#ifdef USE_ROCM
    auto registered_comm = group_to_comm_.find(group_name);
    const bool is_new_registration = registered_comm == group_to_comm_.end() ||
        registered_comm->second != comm;
    ncclCommProperties_t comm_props = NCCL_COMM_PROPERTIES_INITIALIZER;
    const bool device_api_support =
        ncclCommQueryProperties(comm, &comm_props) == ncclSuccess &&
        comm_props.deviceApiSupport;
    group_to_device_api_support_[group_name] = device_api_support;

    bool capture_allocation_support = false;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 7)
    int runtime_version = 0;
    capture_allocation_support =
        ncclGetVersion(&runtime_version) == ncclSuccess &&
        runtime_version >= NCCL_VERSION(2, 30, 7) && device_api_support &&
        c10::utils::check_env("NCCL_CUMEM_ENABLE") == true &&
        c10::utils::check_env("NCCL_WIN_ENABLE") == true;
    if (capture_allocation_support && capture_setup_stream_ == nullptr) {
      c10::cuda::CUDAGuard device_guard(device_);
      const cudaError_t status = cudaStreamCreateWithFlags(
          &capture_setup_stream_, cudaStreamNonBlocking);
      if (status != cudaSuccess) {
        capture_setup_stream_ = nullptr;
        capture_allocation_support = false;
        TORCH_WARN(
            "Failed to create the ROCm NCCL symmetric-memory capture setup "
            "stream: ",
            cudaGetErrorString(status));
      }
    }
#endif
    group_to_capture_allocation_support_[group_name] =
        capture_allocation_support;
#endif
    group_to_comm_[group_name] = comm;
#ifdef USE_ROCM
    if (is_new_registration ||
        group_to_comm_generation_.find(group_name) ==
            group_to_comm_generation_.end()) {
      group_to_comm_generation_[group_name] = next_comm_generation_++;
    }
#endif
  }

  // Unregister `group_name` on this manager's device. Safe to call when
  // nothing is registered. Does not destroy the host comm; lifetime stays
  // with the producer.
  //
  // This key-only form is retained for CUDA callers. ROCm teardown uses the
  // identity-safe overload below.
  void unregister_comm(const std::string& group_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    group_to_comm_.erase(group_name);
#ifdef USE_ROCM
    group_to_device_api_support_.erase(group_name);
    group_to_capture_allocation_support_.erase(group_name);
    group_to_comm_generation_.erase(group_name);
#endif
#ifdef NCCL_HAS_SYMMEM_DEVICE_SUPPORT
    devcomm_registry_.erase(group_name);
#endif
  }

  // Drop the entry only if the currently registered comm is still `comm`.
  // Delayed cleanup for an old process group is therefore a no-op after a
  // same-name successor has registered.
  void unregister_comm(const std::string& group_name, ncclComm_t comm) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = group_to_comm_.find(group_name);
    if (it != group_to_comm_.end() && it->second == comm) {
      group_to_comm_.erase(it);
#ifdef USE_ROCM
      group_to_device_api_support_.erase(group_name);
      group_to_capture_allocation_support_.erase(group_name);
      group_to_comm_generation_.erase(group_name);
#endif
#ifdef NCCL_HAS_SYMMEM_DEVICE_SUPPORT
      devcomm_registry_.erase(group_name);
#endif
    }
  }

  // Destructor: Clean up all registered device communicators.
  // This is a best-effort cleanup. If the CUDA context has already been
  // destroyed, the cleanup will be skipped. All errors are caught and ignored
  // to prevent exceptions from propagating during destruction.
  ~NCCLDevCommManager() noexcept {
#ifdef NCCL_HAS_SYMMEM_DEVICE_SUPPORT
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
#ifdef USE_ROCM
            c10::cuda::CUDAStreamCaptureModeGuard capture_mode_guard{
                cudaStreamCaptureModeRelaxed};
#endif
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
#endif // NCCL_HAS_SYMMEM_DEVICE_SUPPORT
#ifdef USE_ROCM
    if (capture_setup_stream_ != nullptr) {
      try {
        c10::cuda::CUDAGuard guard(device_);
        C10_CUDA_CHECK_WARN(cudaStreamDestroy(capture_setup_stream_));
      } catch (...) {
        LOG(WARNING) << "Failed to destroy the ROCm NCCL symmetric-memory "
                        "capture setup stream, skipping";
      }
    }
#endif
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

#ifdef USE_ROCM
  std::unordered_map<std::string, bool> group_to_device_api_support_;
  std::unordered_map<std::string, bool> group_to_capture_allocation_support_;
  std::unordered_map<std::string, uint64_t> group_to_comm_generation_;
  uint64_t next_comm_generation_{1};
  cudaStream_t capture_setup_stream_{nullptr};
  std::mutex capture_setup_mutex_;
#endif

#ifdef NCCL_HAS_SYMMEM_DEVICE_SUPPORT
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
#endif // NCCL_HAS_SYMMEM_DEVICE_SUPPORT
};

} // namespace c10d::symmetric_memory
#endif // NCCL_HAS_SYMMEM_SUPPORT
