#pragma once

#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryTypes.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

namespace c10d {
namespace symmetric_memory {

bool device_has_multicast_support(int device_idx);

bool allow_overlapping_devices();

// Query environment variable to get the backend used for CUDA Symmetric Memory.
std::string getSymmMemBackendCUDA();

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
  StoreExchange(const std::string& store_prefix)
      : store_prefix_(store_prefix) {}

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
      std::ostringstream oss;
      oss << store_prefix_ << "/" << seq_id_ << "/" << r;
      peer_keys.push_back(oss.str());
    }
    ++seq_id_;

    {
      std::vector<uint8_t> payload(
          reinterpret_cast<uint8_t*>(&val),
          reinterpret_cast<uint8_t*>(&val) + sizeof(T));
      store->set(peer_keys[rank], payload);
    }

    std::vector<T> peer_vals;
    peer_vals.reserve(world_size);
    for (int r = 0; r < world_size; ++r) {
      if (r == rank) {
        peer_vals.push_back(val);
        continue;
      }
      store->wait({peer_keys[r]});
      auto payload = store->get(peer_keys[r]);
      TORCH_CHECK(payload.size() == sizeof(T));
      T peer_val{};
      std::memcpy(&peer_val, payload.data(), sizeof(T));
      peer_vals.push_back(peer_val);
    }
    return peer_vals;
  }

  void barrier(
      const c10::intrusive_ptr<c10d::Store>& store,
      int rank,
      int world_size) {
    // TODO: implement an efficient one?
    all_gather(store, rank, world_size, 0);
  }

 private:
  const std::string store_prefix_;
  size_t seq_id_ = 0;
};

// Teturns a pointer of virtual address that is mapped to the physical memory
// held by the handle.
void map_block(
    void** ptr,
    c10d::symmetric_memory::HandleType handle,
    size_t size,
    int device_idx);

} // namespace symmetric_memory
} // namespace c10d
