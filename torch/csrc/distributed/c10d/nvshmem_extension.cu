#include <torch/csrc/distributed/c10d/nvshmem_extension.cuh>

#include <c10/cuda/CUDAGuard.h>

#include <torch/csrc/distributed/c10d/CUDASymmetricMemory-inl.h>
#include <torch/csrc/distributed/c10d/CUDASymmetricMemoryUtils.hpp>
#include <torch/csrc/distributed/c10d/SymmetricMemory.hpp>

#include <cuda_awbarrier_primitives.h>
#include <nvshmem.h>

namespace c10d::nvshmem_extension {

using c10d::symmetric_memory::StoreExchange;
static StoreExchange storeExchange = StoreExchange("nvshmem_ext");

#define THREADS_PER_BLOCK 512

// Bootstrap based on user's setting for NCCL
// Long term, this may be a bit unclean; short term, it improves UX
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
  auto unique_ids = storeExchange.all_gather(store, rank, world_size, unique_id);

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

at::Tensor nvshmem_all_to_all(
    at::Tensor& input,
    at::Tensor& out,
    std::string group_name) {
  auto input_hdl = c10d::symmetric_memory::rendezvous(input, group_name);
  auto out_hdl = c10d::symmetric_memory::rendezvous(out, group_name);
  int rank = input_hdl->get_rank();
  int world_size = input_hdl->get_world_size();
  auto team = group_to_team(group_name, input_hdl->get_rank_to_global_rank());

  void* input_ptr = input_hdl->get_buffer_ptrs()[rank];
  void* output_ptr = out_hdl->get_buffer_ptrs()[rank];
  size_t bytes_per_rank = input_hdl->get_buffer_size() / world_size;

  auto stream = at::cuda::getCurrentCUDAStream(input.device().index());
  nvshmemx_alltoallmem_on_stream(team, output_ptr, input_ptr, bytes_per_rank, stream);
  return out;
}

// This is a prefix sum function that calculates read (or write) offsets for each peer
// TODO: currently it is assumed that the number of PE's is smaller than `THREADS_PER_BLOCK` (512)
__device__ void scan(int64_t *odata, int64_t *idata, int n) {
  constexpr int N = THREADS_PER_BLOCK;
  assert (n <= N);
  __shared__ int64_t temp[N * 2];
  int thid = threadIdx.x;
  int pout = 1, pin = 0;
  // Load input into shared memory scratchpad.
  // This is exclusive scan, so shift right by one
  // and set first element to 0
  if (thid < n) {
    temp[pout * N + thid] = temp[pin * N + thid] = (thid > 0) ? idata[thid - 1] : 0;
  }
  __syncthreads();
  for (int offset = 1; offset < n; offset *= 2) {
    pout = 1 - pout; // swap double buffer indices
    pin = 1 - pout;
    if (thid >= offset && thid < n)
      temp[pout * N + thid] = temp[pin * N + thid] + temp[pin * N + thid - offset];
    else if (thid < offset)
      temp[pout * N + thid] = temp[pin * N + thid];
    __syncthreads();
  }
  odata[thid] = temp[pout * N + thid]; // write output
}

// This kernel is used to exchange output splits and source offsets between peers.
// `in_out_splits` is of size (3, npes) and contains:
// - input splits (IN)
// - output splits (OUT) and
// - source offsets (OUT).
__global__ void exchangeSplitAndOffset(int64_t* in_out_splits, int mype, int npes) {
  auto input_splits = in_out_splits;
  auto output_splits = in_out_splits + npes;
  auto source_offsets = in_out_splits + npes * 2;
  int tid = threadIdx.x;

  __shared__ int64_t peer_offsets[THREADS_PER_BLOCK];

  // Scan input splits to get the source offsets
  scan(peer_offsets, input_splits, npes);
  __syncthreads();;

  // Use 1 block to do the exchange
  if (tid < npes) {
    int peer = tid;
    nvshmem_int64_p(source_offsets + mype, peer_offsets[peer], peer);
    nvshmem_int64_p(output_splits + mype, input_splits[peer], peer);
  }
  // This barrier ensures that all remote PEs see the updated values
  nvshmemx_barrier_all_block();
}

// This kernel is used to do the actual data exchange.
// `in_out_splits` has the same definition as in `exchangeSplitAndOffset`.
__global__ void allToAllV(void *send_data, void *recv_data, int64_t* in_out_splits, size_t stride, int mype, int npes) {
  auto output_splits = in_out_splits + npes;
  auto source_offsets = in_out_splits + npes * 2;
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  // Calculate the output offsets
  __shared__ int64_t peer_offsets[THREADS_PER_BLOCK];
  scan(peer_offsets, output_splits, npes);
  __syncthreads();

  // Each block targets a different peer
  size_t row_size = stride * sizeof(float);  // Assuming float (TODO)
  for (int i = bid; i < npes; i += gridDim.x) {
    int peer = (mype + i) % npes;
    auto size = output_splits[peer] * row_size;
    auto source_offset = source_offsets[peer] * row_size;
    auto write_offset = peer_offsets[peer] * row_size;
    nvshmemx_getmem_block(
      (char*)recv_data + write_offset,
      (char*)send_data + source_offset,
      size,
      peer);
  }
  // Write out the output offsets (to the scratchpad line)
  if (bid == 0 && tid < npes) {
    source_offsets[tid] = peer_offsets[tid];
  }
}

at::Tensor nvshmem_all_to_all_vdev(
    at::Tensor& input,
    at::Tensor& out,
    at::Tensor& in_out_splits,
    std::string group_name) {
  /* Perform AllToAllv operation using NVSHMEM, with split information provided on device.
   * Arguments:
   *  - `input` is the input tensor
   *  - `out` is the output tensor
   *  - `in_out_splits` is a 2D tensor of size (3, npes). The rows are (in order):
        input splits (IN)
        output splits (OUT) and
        output offsets (OUT).
  */
  auto input_hdl = c10d::symmetric_memory::rendezvous(input, group_name);
  auto out_hdl = c10d::symmetric_memory::rendezvous(out, group_name);
  auto splits_hdl = c10d::symmetric_memory::rendezvous(in_out_splits, group_name);
  int rank = input_hdl->get_rank();
  int world_size = input_hdl->get_world_size();

  void* input_ptr = input_hdl->get_buffer_ptrs()[rank];
  void* output_ptr = out_hdl->get_buffer_ptrs()[rank];
  int64_t* splits_ptr = (int64_t*)(splits_hdl->get_buffer_ptrs()[rank]);

  auto stream = at::cuda::getCurrentCUDAStream(input.device().index());

  // Exchange output splits and source offsets
  // Use collective launch because kernel involves nvshmem barrier
  void* args0[] = {
      &splits_ptr,
      &rank,
      &world_size};
  nvshmemx_collective_launch(
      (const void*)exchangeSplitAndOffset,
      dim3(1),
      dim3(THREADS_PER_BLOCK),
      args0,
      0,
      stream);

  // All to all data exchange
  // Limit the number of blocks to 16
  int num_blocks = std::min(world_size, 16);
  // Stride at dim 0 (assuming input is contiguous, TODO)
  size_t stride = input.stride(0);
  void* args1[] = {
      &input_ptr,
      &output_ptr,
      &splits_ptr,
      &stride,
      &rank,
      &world_size};
  nvshmemx_collective_launch(
      (const void*)allToAllV,
      dim3(num_blocks),
      dim3(THREADS_PER_BLOCK),
      args1,
      0,
      stream);
  return out;
}

} // namespace c10d::nvshmem_extension


TORCH_LIBRARY_IMPL(symm_mem, CUDA, m) {
  m.impl("nvshmem_broadcast", c10d::nvshmem_extension::nvshmem_broadcast);
  m.impl("nvshmem_all_to_all", c10d::nvshmem_extension::nvshmem_all_to_all);
  m.impl("nvshmem_all_to_all_vdev", c10d::nvshmem_extension::nvshmem_all_to_all_vdev);
}
