#pragma once

#include <ATen/ATen.h>
#include <fmt/format.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryTypes.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace c10d {
namespace symmetric_memory {

bool device_has_multicast_support(int device_idx);

bool allow_overlapping_devices();

// Query environment variable to get the backend used for CUDA Symmetric Memory.
std::string getSymmMemBackendCUDA();

// All-gather a fixed-size byte payload through the given ProcessGroup.
// Uses ProcessGroup::_allgather_base (NCCL allgather for a NCCL-backed PG).
// The payload is staged through a uint8 CUDA tensor on `device_idx`; the H2D
// and D2H copies are negligible at the sizes exchanged during rendezvous (a
// few hundred bytes per rank). Returns a contiguous CPU tensor of
// world_size * nbytes uint8 elements.
at::Tensor pg_all_gather_bytes(
    const c10::intrusive_ptr<c10d::ProcessGroup>& pg,
    const void* data,
    size_t nbytes,
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

    std::vector<std::string> peer_keys;
    peer_keys.reserve(world_size);
    for (int r = 0; r < world_size; ++r) {
      peer_keys.push_back(fmt::format("{}/{}/{}", store_prefix_, seq_id_, r));
    }
    ++seq_id_;

    {
      store->set(peer_keys[rank], to_payload(val));
    }

    auto payloads = store->multiGet(peer_keys);

    std::vector<T> peer_vals;
    peer_vals.reserve(world_size);
    for (int r = 0; r < world_size; ++r) {
      peer_vals.push_back(from_payload<T>(payloads[r]));
    }
    return peer_vals;
  }

  template <typename T>
  T broadcast(
      const c10::intrusive_ptr<c10d::Store>& store,
      int rank,
      int world_size,
      int src_rank,
      T val) {
    static_assert(std::is_trivially_copyable_v<T>);
    TORCH_CHECK(
        rank >= 0 && rank < world_size,
        "StoreExchange::broadcast: rank ",
        rank,
        " is out of range for world_size ",
        world_size,
        ".");
    TORCH_CHECK(
        src_rank >= 0 && src_rank < world_size,
        "StoreExchange::broadcast: source rank ",
        src_rank,
        " is out of range for world_size ",
        world_size,
        ".");

    auto key = fmt::format("{}/{}/{}", store_prefix_, seq_id_++, src_rank);

    if (rank == src_rank) {
      store->set(key, to_payload(val));
      return val;
    }
    return from_payload<T>(store->get(key));
  }

  void barrier(
      const c10::intrusive_ptr<c10d::Store>& store,
      int rank,
      int world_size) {
    (void)rank;
    store->barrier(fmt::format("{}/{}", store_prefix_, seq_id_++), world_size);
  }

 private:
  template <typename T>
  static std::vector<uint8_t> to_payload(const T& val) {
    static_assert(std::is_trivially_copyable_v<T>);
    auto begin = reinterpret_cast<const uint8_t*>(&val);
    return std::vector<uint8_t>(begin, begin + sizeof(T));
  }

  template <typename T>
  static T from_payload(const std::vector<uint8_t>& payload) {
    static_assert(std::is_trivially_copyable_v<T>);
    TORCH_CHECK(payload.size() == sizeof(T));
    T val{};
    std::memcpy(&val, payload.data(), sizeof(T));
    return val;
  }

  const std::string store_prefix_;
  size_t seq_id_ = 0;
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
