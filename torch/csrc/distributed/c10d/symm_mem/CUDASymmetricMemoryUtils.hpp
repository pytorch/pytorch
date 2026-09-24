#pragma once

#include <ATen/ATen.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryTypes.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#include <cstdint>
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
// The ranks of a group must issue exchanges one at a time and in the same
// order. The mutex makes the counter increment atomic; it does not make
// concurrent rendezvous safe.
class StoreExchange {
 public:
  StoreExchange(std::string store_prefix)
      : store_prefix_(std::move(store_prefix)) {}

  // Put template function in header file so that compiler can easily access it.
  template <typename T>
  std::vector<T> all_gather(
      const c10::intrusive_ptr<c10d::Store>& store,
      int rank,
      int world_size,
      T val) {
    static_assert(std::is_trivially_copyable_v<T>);

    const size_t seq_id = next_seq_id(store);
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
      int world_size) {
    (void)rank;
    std::ostringstream oss;
    oss << store_prefix_ << '/' << next_seq_id(store);
    store->barrier(std::move(oss).str(), world_size);
  }

 private:
  // One counter per group: the sequence number is part of the key every rank
  // computes, so a process-global counter desyncs ranks that sit out a
  // rendezvous (pytorch/pytorch#196082).
  //
  // Keyed by the group's store, not its name: destroy_process_group() resets
  // the name counter so names are reused, while new_group() builds a fresh
  // PrefixStore per group. The weak reference keeps the address allocated
  // while the entry lives, so a live store can never reuse it.
  size_t next_seq_id(const c10::intrusive_ptr<c10d::Store>& store) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto key = reinterpret_cast<std::uintptr_t>(store.get());
    auto it = seq_ids_.find(key);
    if (it == seq_ids_.end()) {
      // Prune on a miss only: the map grows just when a new group appears,
      // so the scan tracks group creation, not rendezvous frequency.
      std::erase_if(
          seq_ids_, [](const auto& kv) { return kv.second.ref.expired(); });
      it = seq_ids_
               .emplace(
                   key, Entry{c10::weak_intrusive_ptr<c10d::Store>(store), 0})
               .first;
    }
    return it->second.seq_id++;
  }

  struct Entry {
    c10::weak_intrusive_ptr<c10d::Store> ref;
    size_t seq_id;
  };

  const std::string store_prefix_;
  std::mutex mutex_;
  // The key is the store's address as an integer: an identity token, never
  // dereferenced. Entry::ref pins the allocation so it cannot be reissued.
  std::unordered_map<std::uintptr_t, Entry> seq_ids_;
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
