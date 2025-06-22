#include <dlfcn.h>
#include <c10/cuda/CUDAGuard.h>

#include <torch/csrc/distributed/c10d/symm_mem/nvshmem_extension.cuh>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory-inl.h>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryUtils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

#include <cuda_awbarrier_primitives.h>
// Use torch's cub wrapper instead of CUDA's <cub/cub.cuh>, see #55292
#include <ATen/cuda/cub.cuh>
#include <nvshmem.h>

namespace c10d::nvshmem_extension {

using c10d::symmetric_memory::StoreExchange;
static StoreExchange storeExchange = StoreExchange("nvshmem_ext");

#define THREADS_PER_BLOCK 512
#define WARP_SIZE 32

constexpr int MiB = 1024 * 1024;

// Check if NVSHMEM is available
bool is_nvshmem_available() {
  // Runtime check
  static std::mutex mutex;
  static int is_available = -2;
  std::lock_guard<std::mutex> lock(mutex);
  if (is_available == -2) {
    void* handle{};
    // Open the shared library, RTLD_LAZY defers symbol resolution until needed
    handle = dlopen("libnvshmem_host.so.3", RTLD_LAZY);
    if (!handle) {
      std::cerr << dlerror() << "\n";
      is_available = 0;
    } else {
      is_available = 1;
      // Close the shared library
      dlclose(handle);
    }
  }
  return is_available == 1;
}

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

  // Print version
  int major, minor;
  ::nvshmem_info_get_version(&major, &minor);
  LOG(INFO) << "NVSHMEM is available, version: " << major << "." << minor;
}

// Initializes the device state in CUmodule so that it’s able to perform NVSHMEM
// operations.
void nvshmemx_cumodule_init(uintptr_t module) {
  auto cumodule = reinterpret_cast<CUmodule>(module);
  TORCH_CHECK(
    ::nvshmemx_cumodule_init(cumodule) == 0,
    "nvshmemx_cumodule_init failed");
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

void nvshmem_put(at::Tensor& tensor, int64_t peer) {
  // TODO: support non-contiguous tensors
  TORCH_CHECK(tensor.is_contiguous(),
      "put op currently supports contiguous tensors only");
  // TODO: rendezvous should remember the group name
  auto hdl = c10d::symmetric_memory::rendezvous(tensor, "0");
  auto rank = hdl->get_rank();
  void* buffer_ptr = hdl->get_buffer_ptrs()[rank];
  auto buffer_size = tensor.numel() * tensor.element_size();

  c10::cuda::CUDAGuard guard(tensor.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  nvshmemx_putmem_on_stream(buffer_ptr, tensor.data_ptr(), buffer_size, peer, stream);
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

// This is an exclusive prefix sum function that calculates read (or write) offsets for each peer.
__device__ int64_t prefixSum(int64_t *odata, int64_t *idata, int n) {
  // Specialize BlockScan for a 1D block of threads, of type int64_t.
  // - `BLOCK_SCAN_WARP_SCANS` is a low-latency scan algorithm (instead of high
  // throughput which we don't need here).
  // - `at_cuda_detail::cub` is torch's cub wrapper, see #55292.
  using BlockScanT = at_cuda_detail::cub::BlockScan<int64_t, THREADS_PER_BLOCK, at_cuda_detail::cub::BLOCK_SCAN_WARP_SCANS>;
  // Allocate shared memory for BlockScan
  __shared__ typename BlockScanT::TempStorage temp_storage;

  // TODO: currently it is assumed that the number of PE's is smaller than
  // `THREADS_PER_BLOCK`
  CUDA_KERNEL_ASSERT(n <= THREADS_PER_BLOCK);

  // Obtain input item for each thread
  int tid = threadIdx.x;
  int64_t thread_data = (tid < n) ? idata[tid] : 0;

  // Collectively compute the block-wide exclusive prefix sum
  int64_t block_aggregate;
  BlockScanT(temp_storage).ExclusiveSum(thread_data, thread_data, block_aggregate);

  // Store the result
  odata[tid] = thread_data;
  return block_aggregate;
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
  prefixSum(peer_offsets, input_splits, npes);
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
// `stride` is the stride at dim 0, unit in byte.
__global__ void allToAllV(void *send_data, void *recv_data, int64_t* in_out_splits, size_t stride, int mype, int npes) {
  auto output_splits = in_out_splits + npes;
  auto source_offsets = in_out_splits + npes * 2;
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int blocks_per_peer = max(gridDim.x / npes, 1);

  // Calculate the output offsets
  __shared__ int64_t peer_offsets[THREADS_PER_BLOCK];
  prefixSum(peer_offsets, output_splits, npes);
  __syncthreads();

  // Target a different peer based on bid
  for (int i = bid / blocks_per_peer; i < npes; i += gridDim.x / blocks_per_peer) {
    int peer = (mype + i) % npes;
    // Total amount from `peer`
    auto peer_size = output_splits[peer] * stride;
    // Amount to get from `peer` in this block
    auto block_size = peer_size / blocks_per_peer;
    // Being lazy here, we should handle the residual if the division is not exact
    CUDA_KERNEL_ASSERT(block_size * blocks_per_peer == peer_size);
    // This block's offset in the data from `peer`
    auto block_offset = block_size * (bid % blocks_per_peer);
    auto source_offset = source_offsets[peer] * stride + block_offset;
    auto write_offset = peer_offsets[peer] * stride + block_offset;
    nvshmemx_getmem_block(
      (char*)recv_data + write_offset,
      (char*)send_data + source_offset,
      block_size,
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

  // CTA Tuning
  // Intra-node: use multiple blocks per peer to increase data parallelism, up to 8.
  // Up to 1 MB -> 1 block
  // Up to 2 MB -> 2 blocks
  // Up to 4 MB -> 4 blocks
  // More -> 8 blocks
  // The tuning for `num_blocks` below multiplies these numbers by world_size
  // (e.g. 8 -> 8 * 8). If world_size is smaller, we simply shift the blocks
  // towards data parallelism. (There may be room for improvement here)
  auto input_size = input.numel() * input.element_size();
  int num_blocks = input_size < MiB ? 8 :
      (input_size < 2 * MiB ? 16 :
      (input_size < 4 * MiB ? 32 : 64));

  // Inter-node: limit the total the number of blocks to 8 which is able to
  // drive 57 GB/s bandwidth in test, enough to drive a 400 Gb/s NIC.
  // TODO: better intra vs inter detection, currently it is based on world_size
  if (world_size > 8) {
    num_blocks = std::min(num_blocks, 8);
  }

  // Stride at dim 0 (assuming input is contiguous, TODO)
  size_t stride_bytes = input.stride(0) * input.element_size();

  // All to all data exchange
  void* args1[] = {
      &input_ptr,
      &output_ptr,
      &splits_ptr,
      &stride_bytes,
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

// Start of `nvshmem_all_to_all_vdev_2d`
// This kernel is used to exchange output splits and source offsets between peers.
// For meaning of `mype` and `npes`, see the docstring of `nvshmem_all_to_all_vdev_2d`.
// `in_out_splits` is of size (3, npes * ne) and contains:
// - input splits (IN)
// - output splits (OUT) and
// - source offsets (OUT).
__global__ void exchangeSplitAndOffset_2d(int64_t* in_out_splits, int mype, int npes, int ne, size_t input_dim0) {
  int nsplits = npes * ne;
  auto input_splits = in_out_splits;
  auto output_splits = in_out_splits + nsplits;
  auto source_offsets = in_out_splits + nsplits * 2;
  int tid = threadIdx.x;

  __shared__ int64_t peer_offsets[THREADS_PER_BLOCK];

  // Scan input splits to get the source offsets
  auto sum_of_splits = prefixSum(peer_offsets, input_splits, nsplits);
  __syncthreads();;
  CUDA_KERNEL_ASSERT(sum_of_splits <= input_dim0);

  // Use 1 block to do the exchange
  if (tid < nsplits) {
    int peer = tid / ne;
    int e = tid % ne;
    // This does a transpose from rank-major order to expert-major order
    int dst_offset = e * npes + mype;
    auto split_val = input_splits[tid];
    CUDA_KERNEL_ASSERT(split_val >= 0);
    nvshmem_int64_p(source_offsets + dst_offset, peer_offsets[tid], peer);
    nvshmem_int64_p(output_splits + dst_offset, split_val, peer);
  }
  // This barrier ensures that all remote PEs see the updated values
  nvshmemx_barrier_all_block();
}

// This is an warp-scope, exclusive prefix sum. When called by a block of
// threads, each warp will perform an independent prefix sum, concurrently.
// Returns the sum of all elements in the warp.
// `NUM_WARPS` is the number of warps participating the concurrent prefix sum.
template <int NUM_WARPS>
__device__ int64_t prefixSum_warp(int64_t *odata, int64_t *idata, int n) {
  CUDA_KERNEL_ASSERT(n <= WARP_SIZE);

  // Specialize WarpScan for type int
  using WarpScan = at_cuda_detail::cub::WarpScan<int64_t>;
  // Allocate WarpScan shared memory for N warps
  __shared__ typename WarpScan::TempStorage temp_storage[NUM_WARPS];

  int warp_id = threadIdx.x / WARP_SIZE;
  if (warp_id >= NUM_WARPS) {
    return 0;
  }

  // Obtain input item for each thread
  int tid = threadIdx.x % WARP_SIZE;
  int64_t thread_data = (tid < n) ? idata[tid] : 0;

  // Total sum of all elements in the warp
  int64_t warp_aggregate;
  // Compute the warp-wide exclusive prefix sum
  WarpScan(temp_storage[warp_id]).ExclusiveSum(thread_data, thread_data, warp_aggregate);

  // Store the result
  odata[tid] = thread_data;
  return warp_aggregate;
}

// This is for abstracting a thread-group-scope, exclusive prefix sum.
// Since we use warp-scope prefix sum, the thread group size is limited to warp size.
#define A2AV_TILE_SIZE WARP_SIZE

// This kernel is used to do the actual data exchange.
// `in_out_splits` has the same definition as in `exchangeSplitAndOffset`.
// `stride` is the stride at dim 0, unit in byte.
// For meaning of `mype` and `npes`, see the docstring of `nvshmem_all_to_all_vdev_2d`.
__global__ void allToAllV_2d(void *send_data, void *recv_data, int64_t* in_out_splits, size_t stride, int mype, int npes, int ne, int64_t major_align) {
  int nsplits = npes * ne;
  auto output_splits = in_out_splits + nsplits;
  auto source_offsets = in_out_splits + nsplits * 2;
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  // Split the thread block into tiles
  constexpr int NUM_TILES = THREADS_PER_BLOCK / A2AV_TILE_SIZE;
  int tileId = tid / A2AV_TILE_SIZE;
  int laneId = tid % A2AV_TILE_SIZE;
  // Each tile calculates its own prefix sum
  __shared__ int64_t tile_prefix_sums[NUM_TILES][A2AV_TILE_SIZE];
  // A tile takes care of npes worth of splits
  int nsplits_per_tile = min(npes, nsplits - tileId * npes);
  // TODO: currently it is assumed that the number of PE's is smaller than
  // `A2AV_TILE_SIZE` bc the warp-scope prefix sum can only handle up to
  // WARP_SIZE elements
  CUDA_KERNEL_ASSERT(npes <= A2AV_TILE_SIZE);
  // Similarly, the number of experts per rank is also assumed to be smaller
  // than `NUM_TILES`
  CUDA_KERNEL_ASSERT(ne <= NUM_TILES);

  // Total length of each tile
  __shared__ int64_t len_per_tile[NUM_TILES];
  // When `nsplits` is small, not every tile gets data to sum. They can skip
  // this local prefix sum.
  if (nsplits_per_tile > 0) {
    // Each tile calculates its own prefix sum, return value is the sum of all elements in the tile.
    int64_t my_tile_len = prefixSum_warp<NUM_TILES>(tile_prefix_sums[tileId], output_splits + tileId * npes, nsplits_per_tile);
    // Last thread in each tile does the up aligning.
    if (laneId == A2AV_TILE_SIZE - 1) {
      auto aligned_len = (my_tile_len + major_align - 1) / major_align * major_align;
      // In case `aligned_len` is 0, we set it to `major_align` to avoid an
      // empty bin, bc cutlass currently does not support it. See
      // https://github.com/pytorch/pytorch/issues/152668.
      len_per_tile[tileId] = max(aligned_len, major_align);
    }
  }
  __syncthreads();

  // Starting offset of each tile
  __shared__ int64_t start_offset_per_tile[NUM_TILES];
  // Prefix sum again to get the tiles' start offsets.
  // `NUM_TILES` is typically not greater than 32, because 32 tiles * 32 threads
  // = 1024 threads, and this kernel is launched within 1024 threads. Thus, we
  // can use warp-scope prefix sum.
  static_assert(NUM_TILES <= WARP_SIZE);
  // Only 1 warp is needed
  prefixSum_warp<1>(start_offset_per_tile, len_per_tile, NUM_TILES);
  __syncthreads();

  // Add tile offset to every element in the tile
  tile_prefix_sums[tileId][laneId] += start_offset_per_tile[tileId];
  __syncthreads();

  // Target a different e based on bid
  for (int eid = bid; eid < nsplits; eid += gridDim.x) {
    int peer = eid % npes;
    // Amount from `peer` for `e`
    auto peer_size = output_splits[eid] * stride;
    auto source_offset = source_offsets[eid] * stride;
    auto e_offset = tile_prefix_sums[eid / npes][peer];
    auto write_offset = e_offset * stride;
    nvshmemx_getmem_block(
      (char*)recv_data + write_offset,
      (char*)send_data + source_offset,
      peer_size,
      peer);
  }
  // Write out the output offsets (to the scratchpad line)
  if (bid == 0 && tid < nsplits) {
    source_offsets[tid] = tile_prefix_sums[tid / npes][tid % npes];
  }
}

at::Tensor nvshmem_all_to_all_vdev_2d(
    at::Tensor& input,
    at::Tensor& out,
    at::Tensor& in_out_splits,
    std::string group_name,
    std::optional<int64_t> major_align) {
  /* Perform a 2D AllToAllv shuffle operation using NVSHMEM, with split information provided on device.
   * Arguments:
   *  - `input` is the input tensor
   *  - `out` is the output tensor
   *  - `in_out_splits` is a 2D tensor of size (3, `world_size` * `ne`). In the
        scenario of Mixture-of-Experts models, `ne` is the number of experts per
        rank. The rows of `in_out_splits` are (in order):
        input splits (IN)
        output splits (OUT) and
        output offsets (OUT).
   *  - `group_name` is the name of the group to use for the collective operation.
   *  - `major_align` is the alignment of the "major dimension" of the output
        sequence. See below for details.

   *  A 2D AllToAllv shuffle is illustrated below:
        (world_size = 2, ne = 2, total number of experts = 4)
        Source: |       Rank 0      |       Rank 1      |
                | c0 | c1 | c2 | c3 | d0 | d1 | d2 | d3 |

        Dest  : |       Rank 0      |       Rank 1      |
                | c0 | d0 | c1 | d1 | c2 | d2 | c3 | d3 |
        where each `c_i` / `d_i` are slices of the `input` tensor, targeting
        expert `i`, with length indicated by input splits (in
        `in_out_splits[0]`).  That is, the 2D AllToAllv shuffle achieves a
        transpose from rank-major order at input to expert-major order at
        output.

   *  If `major_align` is not 1, the output offsets of c1, c2, c3 will be
      up-aligned to this value. For example, if c0 has length 5 and d0 has
      length 7 (making a total of 12), and if the `major_align` is set to 16,
      the output offset of c1 will be 16. Similar for c2 and c3. This value has
      no effect on the offset of the minor dimension, i.e.  d0, d1, d2 and d3.
      Note: since cutlass does not support empty bins, we set the aligned length
      to `major_align` if it is 0. See
      https://github.com/pytorch/pytorch/issues/152668.
  */
  auto input_hdl = c10d::symmetric_memory::rendezvous(input, group_name);
  auto out_hdl = c10d::symmetric_memory::rendezvous(out, group_name);
  auto splits_hdl = c10d::symmetric_memory::rendezvous(in_out_splits, group_name);
  int rank = input_hdl->get_rank();
  int world_size = input_hdl->get_world_size();
  // TODO: world_size is currently limited by the number of elements in a WarpScan.
  TORCH_CHECK(world_size <= A2AV_TILE_SIZE, "world_size must be smaller than A2AV_TILE_SIZE", A2AV_TILE_SIZE);

  // If `major_align` is not provided, use 1 as the default value.
  int64_t major_align_val = major_align.value_or(1);
  TORCH_CHECK(major_align_val > 0, "major_align must be positive");

  void* input_ptr = input_hdl->get_buffer_ptrs()[rank];
  void* output_ptr = out_hdl->get_buffer_ptrs()[rank];
  int64_t* splits_ptr = (int64_t*)(splits_hdl->get_buffer_ptrs()[rank]);

  // Shape checks
  auto split_shape = in_out_splits.sizes();
  TORCH_CHECK(in_out_splits.is_contiguous()
      && input.is_contiguous()
      && out.is_contiguous(),
      "input, out and in_out_splits must be contiguous");
  TORCH_CHECK(split_shape.size() == 2
      && split_shape[0] == 3
      && split_shape[1] % world_size == 0,
      "in_out_splits must be 2D with 3 rows, "
      "each row must be a multiple of world_size");

  // Consistency checks
  TORCH_CHECK(input.dtype() == out.dtype()
      && input.stride(0) == out.stride(0),
      "input and out must have the same dtype and same stride at dim 0");
  TORCH_CHECK(in_out_splits.scalar_type() == at::kLong, "in_out_splits must be int64");

  // Number of experts per rank
  int ne = split_shape[1] / world_size;
  constexpr int NUM_TILES = THREADS_PER_BLOCK / A2AV_TILE_SIZE;
  TORCH_CHECK(ne <= NUM_TILES, "Number of experts must be smaller than NUM_TILES", NUM_TILES);

  // Set device context for getting the stream and launching kernels below
  c10::cuda::CUDAGuard guard(input.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  // Exchange output splits and source offsets
  auto input_dim0 = input.size(0);
  // Use collective launch because kernel involves nvshmem barrier
  void* args0[] = {
      &splits_ptr,
      &rank,
      &world_size,
      &ne,
      &input_dim0};
  nvshmemx_collective_launch(
      (const void*)exchangeSplitAndOffset_2d,
      dim3(1),
      dim3(THREADS_PER_BLOCK),
      args0,
      0,
      stream);

  // CTA Tuning
  // Naive for now, use 1 block per expert.
  // Total number of blocks is limited to 64 (intra-node) or 8 (inter-node).
  int num_blocks = std::min(world_size * ne, world_size > 8 ? 8 : 64);

  // Stride at dim 0
  size_t stride_bytes = input.stride(0) * input.element_size();

  // All to all data exchange
  void* args1[] = {
      &input_ptr,
      &output_ptr,
      &splits_ptr,
      &stride_bytes,
      &rank,
      &world_size,
      &ne,
      &major_align_val};
  nvshmemx_collective_launch(
      (const void*)allToAllV_2d,
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
  m.impl("nvshmem_put", c10d::nvshmem_extension::nvshmem_put);
  m.impl("nvshmem_all_to_all", c10d::nvshmem_extension::nvshmem_all_to_all);
  m.impl("nvshmem_all_to_all_vdev", c10d::nvshmem_extension::nvshmem_all_to_all_vdev);
  m.impl("nvshmem_all_to_all_vdev_2d", c10d::nvshmem_extension::nvshmem_all_to_all_vdev_2d);
}
