#include <torch/csrc/distributed/c10d/nvshmem_extension.cuh>

#include <c10/cuda/CUDAGuard.h>

#include <torch/csrc/distributed/c10d/SymmetricMemory.hpp>

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

void* nvshmem_ptr(const void* dest, int pe) {
  return ::nvshmem_ptr(dest, pe);
}

__global__ void ring_bcast(int* data, size_t nelem, int root, uint64_t* psync) {
  int mype = nvshmem_my_pe();
  int npes = nvshmem_n_pes();
  int peer = (mype + 1) % npes;

  if (mype == root)
    *psync = 1;

  nvshmem_signal_wait_until(psync, NVSHMEM_CMP_NE, 0);

  if (mype == npes - 1)
    return;

  nvshmem_int_put(data, data, nelem, peer);
  nvshmem_fence();
  nvshmemx_signal_op(psync, 1, NVSHMEM_SIGNAL_SET, peer);

  *psync = 0;
}

at::Tensor nvshmem_hello(at::Tensor& input) {
  auto symm_mem = c10d::symmetric_memory::rendezvous(input, "0");
  int rank = symm_mem->get_rank();
  int world_size = symm_mem->get_world_size();

  void* buffer_ptr = symm_mem->get_buffer_ptrs()[rank];
  void* signal_pad_ptr = symm_mem->get_signal_pad_ptrs()[rank];
  size_t buffer_size = symm_mem->get_buffer_size();
  int root = 0;
  void* args[] = {&buffer_ptr, &buffer_size, &root, &signal_pad_ptr};

  dim3 grid_dim(1), block_dim(1);
  auto stream = at::cuda::getCurrentCUDAStream();
  nvshmemx_barrier_all_on_stream(stream);
  nvshmemx_collective_launch(
      (const void*)ring_bcast, grid_dim, block_dim, args, 0, stream);
  nvshmemx_barrier_all_on_stream(stream);

  return input;
}

} // namespace c10d::nvshmem_extension
