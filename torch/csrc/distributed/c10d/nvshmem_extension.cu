#include <torch/csrc/distributed/c10d/nvshmem_extension.cuh>

#include <c10/cuda/CUDAGuard.h>

#include <torch/csrc/distributed/c10d/CUDASymmetricMemory-inl.h>
#include <torch/csrc/distributed/c10d/SymmetricMemory.hpp>

#include <cuda_awbarrier_primitives.h>
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

void maybe_initialize_env_vars() {
  auto nccl_socket_if_name = c10::utils::get_env("NCCL_SOCKET_IFNAME");
  auto nccl_hca_list = c10::utils::get_env("NCCL_IB_HCA");
  auto nccl_ib_gid_index = c10::utils::get_env("NCCL_IB_GID_INDEX");
  auto nvshmem_socket_if_name =
      c10::utils::get_env("NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME");
  auto nvshmem_hca_list = c10::utils::get_env("NCCL_IB_HCA");
  auto nvshmem_ib_gid_index = c10::utils::get_env("NVSHMEM_IB_GID_INDEX");

  if (!nvshmem_socket_if_name.has_value() && nccl_socket_if_name.has_value()) {
    c10::utils::set_env(
        "NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME", nccl_socket_if_name->c_str());
  }
  if (!nvshmem_hca_list.has_value() && nccl_hca_list.has_value()) {
    c10::utils::set_env("NVSHMEM_ENABLE_NIC_PE_MAPPING", "1");
    c10::utils::set_env("NVSHMEM_HCA_LIST", nccl_hca_list->c_str());
  }
  if (!nvshmem_ib_gid_index.has_value() && nccl_ib_gid_index.has_value()) {
    c10::utils::set_env("NVSHMEM_IB_GID_INDEX", nccl_ib_gid_index->c_str());
  }
}

void initialize_nvshmem_with_store(
    c10::intrusive_ptr<c10d::Store> store,
    int rank,
    int world_size) {
  static bool is_initialized = false;
  if (is_initialized) {
    return;
  }

  maybe_initialize_env_vars();

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

std::unordered_map<std::string, nvshmem_team_t> group_name_to_team_;

nvshmem_team_t group_to_team(
    const std::string& group_name,
    const std::vector<int>& global_ranks) {
  auto it = group_name_to_team_.find(group_name);
  if (it != group_name_to_team_.end()) {
    return it->second;
  }
  TORCH_CHECK(global_ranks.size() > 1);
  int stride = global_ranks[1] - global_ranks[0];
  for (size_t r = 1; r < global_ranks.size(); ++r) {
    TORCH_CHECK(global_ranks[r] - global_ranks[r - 1] == stride);
  }

  nvshmem_team_t team;
  TORCH_CHECK(
      nvshmem_team_split_strided(
          NVSHMEM_TEAM_WORLD,
          global_ranks[0],
          stride,
          global_ranks.size(),
          nullptr,
          0,
          &team) == 0);
  group_name_to_team_[group_name] = team;
  TORCH_CHECK(team != NVSHMEM_TEAM_INVALID);
  return team;
}

at::Tensor nvshmem_broadcast(at::Tensor& input, const std::string& group_name) {
  auto input_hdl = c10d::symmetric_memory::rendezvous(input, group_name);
  int rank = input_hdl->get_rank();
  int world_size = input_hdl->get_world_size();
  auto team = group_to_team(group_name, input_hdl->get_rank_to_global_rank());
  void* buffer_ptr = input_hdl->get_buffer_ptrs()[rank];

  auto stream = at::cuda::getCurrentCUDAStream();
  nvshmemx_broadcastmem_on_stream(team, buffer_ptr, buffer_ptr, input_hdl->get_buffer_size(), 0, stream);
  return input;
}

// TODO: lock-free ring buf
template <typename T, int N>
struct ring_buf {
  T data[N];
  uint32_t head;
  uint32_t tail;
  uint32_t lock;
};

template <typename T, int N>
__device__ void ring_buf_init(ring_buf<T, N>* buf) {
  buf->head = 0;
  buf->tail = 0;
  buf->lock = 0;
}

template <typename T, int N>
__device__ void ring_buf_lock(ring_buf<T, N>* buf) {
  while (atomicCAS(&buf->lock, 0, 1) != 0) {
    __nanosleep(100);
  }
}

template <typename T, int N>
__device__ void ring_buf_unlock(ring_buf<T, N>* buf) {
  while (atomicCAS(&buf->lock, 1, 0) != 1) {
    __nanosleep(100);
  }
}

template <typename T, int N>
__device__ bool ring_buf_try_push(ring_buf<T, N>* buf, const T& value) {
  ring_buf_lock(buf);

  int new_head = (buf->head + 1) % N;
  if (new_head == buf->tail) {
    // Buffer is full
    ring_buf_unlock(buf);
    return false;
  }

  // Store the data and advance the head
  buf->data[buf->head] = value;
  buf->head = new_head;
  ring_buf_unlock(buf);
  return true;
}

template <typename T, int N>
__device__ bool ring_buf_try_pop(ring_buf<T, N>* buf, T& out) {
  ring_buf_lock(buf);

  if (buf->tail == buf->head) {
    // Buffer is empty
    ring_buf_unlock(buf);
    return false;
  }

  // Retrieve the data and advance the tail
  out = buf->data[buf->tail];
  buf->tail = (buf->tail + 1) % N;
  ring_buf_unlock(buf);
  return true;
}

template <typename T, int N>
__device__ void ring_buf_push(ring_buf<T, N>* buf, const T& value) {
  while (!ring_buf_try_push(buf, value)) {
    __nanosleep(100);
  }
}

template <typename T, int N>
__device__ T ring_buf_pop(ring_buf<T, N>* buf) {
  T value;
  while (!ring_buf_try_pop(buf, value)) {
    __nanosleep(100);
  }
  return value;
}

//         |  |  |
//         v  v  v
//        +--+--+--+--+
// rank 0 |  |  |  |  |
//        +--+--+--+--+
//            |  |  |
//            v  v  v
//        +--+--+--+--+
// rank 1 |  |  |  |  |
//        +--+--+--+--+
//         |     |  |
//         v     v  v
//        +--+--+--+--+
// rank 2 |  |  |  |  |
//        +--+--+--+--+
//         |  |     |
//         v  v     v
//        +--+--+--+--+
// rank 3 |  |  |  |  |
//        +--+--+--+--+
//         |  |  |
//         v  v  v
template <typename T, bool debug>
__global__ void nvshmem_all_reduce_kernel(
    T* input_ptr,
    T* output_ptr,
    size_t numel,
    uint64_t* signal_pad_ptr,
    int rank,
    int world_size,
    int* rank_to_global_rank,
    nvshmem_team_t team) {
  __shared__ ring_buf<int, 128> acc_queue;
  __shared__ int acc_split_idx;

  if (threadIdx.x == 0) {
    ring_buf_init(&acc_queue);
  }
  __syncthreads();

  constexpr int warp_size = 32;
  const int warp_idx = threadIdx.x / warp_size;
  const size_t split_size = numel / world_size;  // TODO: handle unaligned
  const size_t chunk_size = split_size / gridDim.x; // TODO: handle unaligned

  // Split the signal pad among blocks
  uint64_t* split_signals = &signal_pad_ptr[blockIdx.x * world_size];

  if (warp_idx == 0) {
    // ==================
    // Communication warp
    // ==================
    const int thread_idx = threadIdx.x;
    const int next_global_rank = rank_to_global_rank[(rank + 1) % world_size];
    if (thread_idx != 0) {
      return;
    }

    {
      const int split_idx = (rank + world_size - 1) % world_size;
      const size_t split_begin = split_idx * split_size;
      const size_t chunk_begin = split_begin + blockIdx.x * chunk_size;
      nvshmem_int_put_signal_nbi(
          output_ptr + chunk_begin,
          input_ptr + chunk_begin,
          chunk_size,
          &split_signals[split_idx],
          1,
          NVSHMEM_SIGNAL_SET,
          next_global_rank);
    }

    int received = 0, forwarded = 1;
    while (true) {
      for (int split_idx = 0;
           split_idx < world_size && received != world_size - 1;
           ++split_idx) {
        if (nvshmem_uint64_test(&split_signals[split_idx], NVSHMEM_CMP_EQ, 1)) {
          ring_buf_push(&acc_queue, split_idx);
          split_signals[split_idx] = 0;
          received += 1;
        }
      }

      for (int split_idx = 0;
           split_idx < world_size && forwarded != world_size - 1;
           ++split_idx) {
        if (nvshmem_uint64_test(&split_signals[split_idx], NVSHMEM_CMP_EQ, 2)) {
          const int split_begin = split_idx * split_size;
          const int chunk_begin = split_begin + blockIdx.x * chunk_size;
          nvshmem_int_put_signal_nbi(
              output_ptr + chunk_begin,
              output_ptr + chunk_begin,
              chunk_size,
              &split_signals[split_idx],
              1,
              NVSHMEM_SIGNAL_SET,
              next_global_rank);
          split_signals[split_idx] = 0;
          forwarded += 1;
        }
      }

      if (forwarded == world_size - 1 && received == world_size - 1) {
        nvshmem_quiet();
        break;
      }
    }
  } else {
    // ===============
    // Reduction wraps
    // ===============
    const int thread_idx = threadIdx.x - warp_size;
    const int num_threads = blockDim.x - warp_size;

    for (int step = 0; step < world_size - 1; ++step) {
      if (thread_idx == 0) {
        int split_idx = ring_buf_pop(&acc_queue);
        acc_split_idx = split_idx;
      }
      asm volatile("bar.sync 0, 512;" : : : "memory");

      const int split_idx = acc_split_idx;
      const size_t split_begin = split_idx * split_size;
      const size_t chunk_begin = split_begin + blockIdx.x * chunk_size;

      for (size_t offset = chunk_begin;
           offset < std::min(chunk_begin + chunk_size, numel);
           offset += num_threads) {
        if (offset + thread_idx < numel) {
          output_ptr[offset + thread_idx] =
              output_ptr[offset + thread_idx] + input_ptr[offset + thread_idx];
        }
      }
      asm volatile("bar.sync 0, 512;" : : : "memory");

      if (thread_idx == 0) {
        if (split_idx != rank) {
          split_signals[split_idx] = 2;
        }
      }
    }
  }
}

at::Tensor nvshmem_reduce_scatter_out(
    at::Tensor& input,
    std::string group_name,
    at::Tensor& out) {
  auto input_hdl = c10d::symmetric_memory::rendezvous(input, group_name);
  auto out_hdl = c10d::symmetric_memory::rendezvous(out, group_name);
  int rank = input_hdl->get_rank();
  int world_size = input_hdl->get_world_size();
  auto team = group_to_team(group_name, input_hdl->get_rank_to_global_rank());

  void* input_ptr = input_hdl->get_buffer_ptrs()[rank];
  void* output_ptr = out_hdl->get_buffer_ptrs()[rank];
  size_t numel = input.numel();
  void* signal_pad_ptr = input_hdl->get_signal_pad_ptrs()[rank];
  void* rank_to_global_rank = input_hdl->get_rank_to_global_rank_dev();
  void* args[] = {
      &input_ptr,
      &output_ptr,
      &numel,
      &signal_pad_ptr,
      &rank,
      &world_size,
      &rank_to_global_rank,
      &team};

  dim3 grid_dim(32), block_dim(544);
  auto stream = at::cuda::getCurrentCUDAStream();
  nvshmemx_barrier_on_stream(team, stream);
  nvshmemx_collective_launch(
      (const void*)nvshmem_all_reduce_kernel<int, true>,
      grid_dim,
      block_dim,
      args,
      0,
      stream);
  nvshmemx_barrier_on_stream(team, stream);
  return out;
}

#define THREADS_PER_BLOCK 512

__global__ void sendrecv(float *send_data, float *recv_data, int num_elems, int mype,
                                     int npes) {
    int peer = (mype + 1) % npes;
    int block_offset = blockIdx.x * blockDim.x;
    // All threads in a block call the API with the same arguments
    nvshmemx_float_put_block(recv_data + block_offset, send_data + block_offset,
                             min(blockDim.x, num_elems - block_offset),
                             peer);
}

at::Tensor nvshmem_sendrecv(
    at::Tensor& input,
    at::Tensor& out,
    std::string group_name) {
  auto input_hdl = c10d::symmetric_memory::rendezvous(input, group_name);
  auto out_hdl = c10d::symmetric_memory::rendezvous(out, group_name);
  int rank = input_hdl->get_rank();
  int world_size = input_hdl->get_world_size();

  float* input_ptr = (float*)(input_hdl->get_buffer_ptrs()[rank]);
  float* output_ptr = (float*)(out_hdl->get_buffer_ptrs()[rank]);
  size_t numel = input.numel();

  assert(numel % THREADS_PER_BLOCK == 0); /* for simplicity */
  int num_blocks = numel / THREADS_PER_BLOCK;

  sendrecv<<<num_blocks, THREADS_PER_BLOCK>>>(input_ptr, output_ptr, numel, rank, world_size);
  return out;
}

at::Tensor nvshmem_all_to_all(
    at::Tensor& input,
    at::Tensor& out,
    std::string group_name) {
  auto input_hdl = c10d::symmetric_memory::rendezvous(input, group_name);
  auto out_hdl = c10d::symmetric_memory::rendezvous(out, group_name);
  int rank = input_hdl->get_rank();
  int world_size = input_hdl->get_world_size();
  auto team = group_to_team(group_name, input_hdl->get_rank_to_global_rank());

  void* input_ptr = (float*)(input_hdl->get_buffer_ptrs()[rank]);
  void* output_ptr = (float*)(out_hdl->get_buffer_ptrs()[rank]);
  assert input_hdl->get_buffer_size() % world_size == 0;
  size_t bytes_per_rank = input_hdl->get_buffer_size() / world_size;

  auto stream = at::cuda::getCurrentCUDAStream(input.device().index());
  nvshmemx_alltoallmem_on_stream(team, output_ptr, input_ptr, bytes_per_rank, stream);
  return out;
}

} // namespace c10d::nvshmem_extension
