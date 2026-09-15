#pragma once

#include <ATen/ATen.h>
#include <c10/util/CallOnce.h>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryTypes.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#include <algorithm>
#include <cstring>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace c10d {
namespace symmetric_memory {

bool device_has_multicast_support(int device_idx);

bool allow_overlapping_devices();

// Query environment variable to get the backend used for CUDA Symmetric Memory.
std::string getSymmMemBackendCUDA();

// All-gather a fixed-size byte payload through the given ProcessGroup.
// Uses ProcessGroup::all_gather_single (NCCL allgather for a NCCL-backed PG).
// The payload is staged through a uint8 tensor on `device_idx`. Returns a
// contiguous CPU tensor of world_size * nbytes uint8 elements.
at::Tensor pg_all_gather_bytes(
    const c10::intrusive_ptr<c10d::ProcessGroup>& pg,
    const void* data,
    size_t nbytes,
    int device_idx);

// Broadcast a fixed-size byte payload from rank `root` through the given
// ProcessGroup. Same staging scheme as `pg_all_gather_bytes`. Returns a
// contiguous CPU tensor of `nbytes` uint8 elements.
at::Tensor pg_broadcast_bytes(
    const c10::intrusive_ptr<c10d::ProcessGroup>& pg,
    const void* data,
    size_t nbytes,
    int device_idx,
    int root);

// Blocking barrier over the given ProcessGroup, pinned to `device_idx` so the
// backend does not have to guess which device to barrier on.
void pg_barrier(
    const c10::intrusive_ptr<c10d::ProcessGroup>& pg,
    int device_idx);

// Templated wrapper around `pg_all_gather_bytes` matching the shape of
// `StoreExchange::all_gather` so rendezvous code can swap transports without
// caring about serialization.
template <typename T>
std::vector<T> pg_all_gather(
    const c10::intrusive_ptr<c10d::ProcessGroup>& pg,
    int device_idx,
    const T& val) {
  static_assert(
      std::is_trivially_copyable_v<T>,
      "pg_all_gather requires a trivially copyable type");
  at::Tensor flat = pg_all_gather_bytes(pg, &val, sizeof(T), device_idx);
  const auto world_size = pg->getSize();
  const size_t expected = static_cast<size_t>(world_size) * sizeof(T);
  TORCH_CHECK(
      static_cast<size_t>(flat.numel()) == expected,
      "pg_all_gather: expected ",
      expected,
      " bytes but got ",
      flat.numel());
  std::vector<T> out(world_size);
  std::memcpy(out.data(), flat.data_ptr(), expected);
  return out;
}

// Templated wrapper around `pg_broadcast_bytes`. On non-source ranks `val` is
// ignored and only used to size the payload.
template <typename T>
T pg_broadcast(
    const c10::intrusive_ptr<c10d::ProcessGroup>& pg,
    int device_idx,
    int root,
    const T& val) {
  static_assert(
      std::is_trivially_copyable_v<T>,
      "pg_broadcast requires a trivially copyable type");
  at::Tensor flat = pg_broadcast_bytes(pg, &val, sizeof(T), device_idx, root);
  TORCH_CHECK(
      static_cast<size_t>(flat.numel()) == sizeof(T),
      "pg_broadcast: expected ",
      sizeof(T),
      " bytes but got ",
      flat.numel());
  T out{};
  std::memcpy(&out, flat.data_ptr(), sizeof(T));
  return out;
}

class IpcChannel {
 public:
  IpcChannel();
  ~IpcChannel();

  void send_fd(int dst_pid, int fd);
  int recv_fd();

  std::vector<int> all_gather_fds(
      int rank,
      const std::vector<int>& pids,
      int fd);

  int broadcast_fds(
      int rank,
      int src_rank,
      const std::vector<int>& pids,
      int fd);

 private:
  static std::string get_socket_name(int pid);

  std::string socket_name_;
  int socket_;
};

// A set of store-based exchange methods with a preset prefix typically type of
// the SymmetricMemory.  Most used as static instances at respective
// SymmetricMemory implementation files.
//
// Two things the signatures cannot express. `store` and `group_name` must
// belong to the same group; passing one group's store with another's name
// gives mismatched counters and a hang. And the ranks of a group must issue
// exchanges one at a time and in the same order: the mutex below makes the
// counter increment atomic, not concurrent rendezvous safe, since nothing
// makes two ranks hand out a sequence number to the same exchange.
class StoreExchange {
 public:
  StoreExchange(std::string store_prefix)
      : store_prefix_(std::move(store_prefix)) {
    // Registering here rather than from a translation unit's static
    // initializer: that only runs if the linker keeps the object file, and if
    // it is ever dropped the counters silently stop being cleared.
    static c10::once_flag hook_once;
    c10::call_once(hook_once, [] {
      c10d::register_group_unregister_hook([](const std::string& group_name) {
        forget_group_everywhere(group_name);
      });
    });
    std::lock_guard<std::mutex> lock(instances_mutex());
    instances().push_back(this);
  }

  ~StoreExchange() {
    std::lock_guard<std::mutex> lock(instances_mutex());
    auto& v = instances();
    v.erase(std::remove(v.begin(), v.end(), this), v.end());
  }

  StoreExchange(const StoreExchange&) = delete;
  StoreExchange& operator=(const StoreExchange&) = delete;
  StoreExchange(StoreExchange&&) = delete;
  StoreExchange& operator=(StoreExchange&&) = delete;

  // Drop a group's counter. A numeric group name is only unique among live
  // groups: destroy_process_group() resets the name counter so the next
  // new_group() reuses it, and a counter surviving that would start the new
  // group's members at different values.
  void forget_group(const std::string& group_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    seq_ids_.erase(group_name);
  }

  // The list holds raw pointers, kept valid by each instance removing itself
  // in its destructor.
  //
  // Not safe under thread isolation, where one process hosts several ranks
  // and a group name resolves to a different group per rank: seq_ids_ is
  // process-global, so the ranks already share a counter they should not, and
  // clearing it on one rank's unregister takes it from the others. That mode
  // does not work with StoreExchange either way.
  static void forget_group_everywhere(const std::string& group_name) {
    std::vector<StoreExchange*> snapshot;
    {
      std::lock_guard<std::mutex> lock(instances_mutex());
      snapshot = instances();
    }
    for (auto* instance : snapshot) {
      instance->forget_group(group_name);
    }
  }

  // Put template function in header file so that compiler can easily access it.
  template <typename T>
  std::vector<T> all_gather(
      const c10::intrusive_ptr<c10d::Store>& store,
      int rank,
      int world_size,
      T val,
      const std::string& group_name) {
    static_assert(std::is_trivially_copyable_v<T>);

    const size_t seq_id = next_seq_id(group_name);
    std::vector<std::string> peer_keys;
    peer_keys.reserve(world_size);
    for (int r = 0; r < world_size; ++r) {
      std::ostringstream oss;
      oss << store_prefix_ << '/' << seq_id << '/' << r;
      peer_keys.push_back(std::move(oss).str());
    }

    {
      std::vector<uint8_t> payload(
          reinterpret_cast<uint8_t*>(&val),
          reinterpret_cast<uint8_t*>(&val) + sizeof(T));
      store->set(peer_keys[rank], payload);
    }

    auto payloads = store->multiGet(peer_keys);

    std::vector<T> peer_vals;
    peer_vals.reserve(world_size);
    for (int r = 0; r < world_size; ++r) {
      TORCH_CHECK(payloads[r].size() == sizeof(T));
      T peer_val{};
      std::memcpy(&peer_val, payloads[r].data(), sizeof(T));
      peer_vals.push_back(peer_val);
    }
    return peer_vals;
  }

  void barrier(
      const c10::intrusive_ptr<c10d::Store>& store,
      int rank,
      int world_size,
      const std::string& group_name) {
    (void)rank;
    std::ostringstream oss;
    oss << store_prefix_ << '/' << next_seq_id(group_name);
    store->barrier(std::move(oss).str(), world_size);
  }

 private:
  // One counter per group, not one per process. The counter is part of the
  // key every rank computes, so it has to be a function of the group: a
  // process-global counter advances on ranks that take part in a rendezvous
  // and not on those that do not, after which the members of a larger group
  // disagree about which key to read and the rendezvous hangs with no
  // diagnostic. See pytorch/pytorch#196082.
  //
  // The key itself needs no group component: a process group's store is
  // already a PrefixStore over the group name, so distinct groups cannot
  // collide on keys. Only the counter was shared.
  //
  // The entry is dropped when the group is unregistered, so a recycled name
  // does not inherit it.
  size_t next_seq_id(const std::string& group_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    return seq_ids_[group_name]++;
  }

  static std::vector<StoreExchange*>& instances() {
    static std::vector<StoreExchange*> v;
    return v;
  }

  static std::mutex& instances_mutex() {
    static std::mutex m;
    return m;
  }

  const std::string store_prefix_;
  std::mutex mutex_;
  std::unordered_map<std::string, size_t> seq_ids_;
};

// Returns a pointer of virtual address that is mapped to the physical memory
// held by the handle.
void map_block(
    void** ptr,
    c10d::symmetric_memory::HandleType handle,
    size_t size,
    int device_idx);

} // namespace symmetric_memory
} // namespace c10d
