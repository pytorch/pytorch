#include <torch/csrc/distributed/c10d/nvshmem_extension.cuh>

#include <c10/cuda/CUDAGuard.h>

#include <nvshmem.h>

namespace c10d::nvshmem_extension {

const std::string store_comm_prefix = "nvshmem_extension";
static size_t store_comm_seq_id = 0;

template <typename T>
std::vector<T> store_all_gather(
    const c10::intrusive_ptr<c10d::Store>& store,
    int rank,
    int world_size,
    T val) {
  static_assert(std::is_trivially_copyable_v<T>);

  std::vector<std::string> peer_keys;
  for (int r = 0; r < world_size; ++r) {
    std::ostringstream oss;
    oss << store_comm_prefix << "/" << store_comm_seq_id << "/" << r;
    peer_keys.push_back(oss.str());
  }
  ++store_comm_seq_id;

  {
    std::vector<uint8_t> payload(
        reinterpret_cast<uint8_t*>(&val),
        reinterpret_cast<uint8_t*>(&val) + sizeof(T));
    store->set(peer_keys[rank], payload);
  }

  std::vector<T> peer_vals;
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

void initialize_nvshmem_with_store(
    c10::intrusive_ptr<c10d::Store> store,
    int rank,
    int world_size) {
  static bool is_initialized = false;
  if (is_initialized) {
    return;
  }

  nvshmemx_uniqueid_t unique_id;
  TORCH_CHECK(
      nvshmemx_get_uniqueid(&unique_id) == 0, "nvshmemx_get_uniqueid failed");

  // Using an existing store_all_gather due to laziness.
  // TODO(yifu): should use broadcast
  auto unique_ids = store_all_gather(store, rank, world_size, unique_id);

  nvshmemx_init_attr_t attr;
  nvshmemx_set_attr_uniqueid_args(rank, world_size, &unique_ids[0], &attr);

  TORCH_CHECK(
      nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr) == 0,
      "nvshmemx_init_attr failed");

  is_initialized = true;
}

void* nvshmem_malloc(size_t size) {
  return ::nvshmem_malloc(size);
}

void* nvshmem_ptr(const void *dest, int pe) {
  return ::nvshmem_ptr(dest, pe);
}

}
