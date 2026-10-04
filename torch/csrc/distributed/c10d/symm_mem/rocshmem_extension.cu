// ROCm implementation of the NVSHMEM symmetric memory extension ops.
//
// This is a separate file from nvshmem_extension.cu (rather than a hipified
// copy) for the following reasons:
//
// 1. API differences: NVSHMEM and rocSHMEM device APIs diverge enough that
//    #ifdef'ing would be more noise than signal. Key differences include:
//    - nvshmemx_collective_launch (grid-wide sync) has no rocSHMEM equivalent.
//      The 1D all_to_all_vdev works around this entirely on-device with per-peer
//      signals (see the kernels below); the 2D ops instead use host-side barriers
//      between kernel launches.
//    - nvshmemx_getmem_nbi_block -> rocshmem_getmem_nbi_wg / rocshmem_putmem_nbi_wg
//      (workgroup scope).
//
// 2. Missing features: rocSHMEM does not yet support tiled communication
//    (nvshmemx::Tensor, nvshmemx::tile_sum_reduce_block, etc.), so the
//    tile_reduce and multi_root_tile_reduce ops are not included here.
//
// 3. Offset writeback (2D ops): without grid-wide sync a multi-block kernel
//    cannot safely write output offsets in-kernel (race with blocks still
//    reading source_offsets), so a separate writeOutputOffsets_2d kernel runs
//    after the data exchange. The 1D op sidesteps this by computing its output
//    offsets in the metadata-exchange kernel, before any data moves.

#include <hip/hip_runtime.h>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <vector>
#include <ATen/ceil_div.h>
#include <c10/hip/HIPGuard.h>

#include <torch/csrc/distributed/c10d/symm_mem/env.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nvshmem_extension.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nvshmem_team_manager.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory-inl.cuh>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryUtils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

#include <ATen/hip/cub.cuh>

#include <c10/hip/HIPException.h>
#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;
namespace c10d::nvshmem_extension {

#define THREADS_PER_BLOCK 512
#define WARP_SIZE 64

namespace {

bool parse_rocshmem_version_ge(
    const char* version,
    unsigned min_major,
    unsigned min_minor,
    unsigned min_patch) {
  if (version == nullptr) {
    return false;
  }
  char* end = nullptr;
  unsigned long major = std::strtoul(version, &end, 10);
  if (end == version || *end != '.') {
    return false;
  }
  version = end + 1;
  unsigned long minor = std::strtoul(version, &end, 10);
  if (end == version || *end != '.') {
    return false;
  }
  version = end + 1;
  unsigned long patch = std::strtoul(version, &end, 10);
  if (end == version) {
    return false;
  }
  if (major > min_major) {
    return true;
  }
  if (major < min_major) {
    return false;
  }
  if (minor > min_minor) {
    return true;
  }
  if (minor < min_minor) {
    return false;
  }
  return patch >= min_patch;
}

} // namespace

extern "C" void rocshmem_init() __attribute__((weak));

bool is_nvshmem_available() {
  static const bool ok =
      parse_rocshmem_version_ge(rocshmem::VERSION, 3, 3, 0);
  return ok;
}

void nvshmemx_cumodule_init(uintptr_t module) {
  auto hipmodule = reinterpret_cast<hipModule_t>(module);
  NVSHMEM_CHECK(
    rocshmem_hipmodule_init(hipmodule),
    "rocshmem_hipmodule_init failed");
}

at::Tensor nvshmem_broadcast(at::Tensor& input, const int64_t root, const std::string& group_name) {
  auto input_hdl = c10d::symmetric_memory::rendezvous(input, group_name);
  int rank = input_hdl->get_rank();
  void* buffer_ptr = input.mutable_data_ptr();
  auto buffer_size = input.numel() * input.element_size();
  auto& team_manager = TeamManager::get(input.device());
  auto team = team_manager.get_team(group_name, input_hdl->get_rank_to_global_rank());
  int team_size = rocshmem_team_n_pes(team);
  TORCH_CHECK(root < team_size, "root must be smaller than group size");

  auto stream = at::cuda::getCurrentCUDAStream();
  rocshmem_broadcastmem_on_stream(team, buffer_ptr, buffer_ptr, buffer_size, root, stream);
  return input;
}

void nvshmem_put(at::Tensor& tensor, const int64_t peer) {
  // TODO: support non-contiguous tensors
  TORCH_CHECK(tensor.is_contiguous(),
      "put op currently supports contiguous tensors only");
  // TODO: rendezvous should remember the group name
  auto hdl = c10d::symmetric_memory::rendezvous(tensor, "0");
  auto rank = hdl->get_rank();
  void* buffer_ptr = hdl->get_buffer_ptrs()[rank];
  auto buffer_size = tensor.numel() * tensor.element_size();
  TORCH_CHECK(peer < hdl->get_world_size(), "peer must be smaller than world size");

  c10::cuda::CUDAGuard guard(tensor.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  rocshmem_putmem_on_stream(buffer_ptr, tensor.data_ptr(), buffer_size, peer, stream);
}

void nvshmem_wait_for_signal(at::Tensor& sigpad, int64_t signal, int64_t peer) {
  c10::cuda::CUDAGuard guard(sigpad.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  rocshmem_signal_wait_until_on_stream(static_cast<uint64_t*>(sigpad.data_ptr()), ROCSHMEM_CMP_EQ, signal, stream);
}

void nvshmem_put_with_signal(at::Tensor& tensor, at::Tensor& sigpad, int64_t signal, int64_t peer) {
  auto buffer_size = tensor.numel() * tensor.element_size();

  c10::cuda::CUDAGuard guard(tensor.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  rocshmem_putmem_signal_on_stream(
    tensor.mutable_data_ptr(),
    tensor.mutable_data_ptr(),
    buffer_size,
    static_cast<uint64_t*>(sigpad.mutable_data_ptr()),
    signal,
    ROCSHMEM_SIGNAL_SET,
    peer,
    stream);
}

void nvshmem_get(at::Tensor& tensor, const int64_t peer) {
  // TODO: support non-contiguous tensors
  TORCH_CHECK(tensor.is_contiguous(),
      "get op currently supports contiguous tensors only");
  // TODO: rendezvous should remember the group name
  auto hdl = c10d::symmetric_memory::rendezvous(tensor, "0");
  auto rank = hdl->get_rank();
  void* buffer_ptr = hdl->get_buffer_ptrs()[rank];
  auto buffer_size = tensor.numel() * tensor.element_size();
  TORCH_CHECK(peer < hdl->get_world_size(), "peer must be smaller than world size");

  c10::cuda::CUDAGuard guard(tensor.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  rocshmem_getmem_on_stream(tensor.mutable_data_ptr(), buffer_ptr, buffer_size, peer, stream);
}

at::Tensor nvshmem_all_to_all(
    at::Tensor& input,
    at::Tensor& out,
    std::string group_name) {
  auto input_hdl = c10d::symmetric_memory::rendezvous(input, group_name);
  auto out_hdl = c10d::symmetric_memory::rendezvous(out, group_name);
  int rank = input_hdl->get_rank();
  int world_size = input_hdl->get_world_size();
  auto& team_manager = TeamManager::get(input.device());
  auto team = team_manager.get_team(group_name, input_hdl->get_rank_to_global_rank());

  void* input_ptr = input.data_ptr();
  void* output_ptr = out.mutable_data_ptr();
  TORCH_CHECK(input.is_contiguous() && out.is_contiguous());
  TORCH_CHECK_EQ(input.numel(), out.numel());
  TORCH_CHECK_EQ(input.dtype(), out.dtype());
  TORCH_CHECK_EQ(input.numel() % world_size, 0);
  auto buffer_size = input.numel() * input.element_size();
  size_t bytes_per_rank = buffer_size / world_size;

  auto stream = at::cuda::getCurrentCUDAStream(input.device().index());
  rocshmem_alltoallmem_on_stream(team, output_ptr, input_ptr, bytes_per_rank, stream);
  return out;
}

// This is an exclusive prefix sum function that calculates read (or write) offsets for each peer.
__device__ int64_t prefixSum(int64_t *odata, int64_t *idata, int n) {
  // Specialize BlockScan for a 1D block of threads, of type int64_t.
  // - `BLOCK_SCAN_WARP_SCANS` is a low-latency scan algorithm (instead of high
  // throughput which we don't need here).
  // - `at_cuda_detail::cub` is torch's cub wrapper, see #55292.
  using BlockScanT = ROCM_HIPCUB(at_cuda_detail::cub)::BlockScan<int64_t,
        THREADS_PER_BLOCK, ROCM_HIPCUB(at_cuda_detail::cub)::BLOCK_SCAN_WARP_SCANS>;
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

static int get_a2a_nblocks(size_t size, int world_size, bool intra_node) {
  // Check user setting first
  int num_blocks = c10d::symmetric_memory::getenv_nblocks();
  if (num_blocks > 0) {  // set by user
    return num_blocks;
  }
  // 16B per thread, 8 loops
  constexpr size_t chunk_size = 16 * THREADS_PER_BLOCK * 8;
  num_blocks = at::ceil_div(size, chunk_size);
  // Allow kernel to target even number of blocks per peer
  num_blocks = at::round_up(num_blocks, world_size);
  const int max_blocks = intra_node ? 64 : 16;
  return ::min(num_blocks, max_blocks);
}

// 1D all_to_all_vdev: variable-length all-to-all with split info on the device.
//
// Fully device-driven, no host barrier. Each rank PUSHES its output into peers'
// receive buffers with putmem (device-workgroup put sustains markedly higher
// bandwidth than get on this fabric); cross-rank ordering is done in-kernel with
// per-peer signals on the symmetric-memory signal pad, and a monotonically
// increasing epoch makes each call self-contained (allToAllVComplete is a per-op
// barrier, so back-to-back calls that reuse the buffers are safe). Signal-pad
// layout (int64): [0,npes)=counts-ready, [npes,2npes)=offset-ready,
// [2npes,3npes)=data-done, [3npes,4npes)=my write offset into each peer.

// Exchange metadata for the push: send each peer our per-peer count, compute our
// receive layout, and hand each source the offset at which it should write into us.
//   input_splits       : int64[npes] IN  - #elements this rank sends to each peer
//   out_splits_offsets : int64[2,npes] OUT - row 0 = output splits (recv count from
//                        each source), row 1 = output offsets (exclusive prefix sum)
//   sig                : signal pad, int64[>=4*npes]
//   epoch              : per-call token; peers signal with CMP_GE(epoch)
//   team               : rocSHMEM team for this group
__global__ void exchangeSplitAndOffset(int64_t* input_splits, int64_t* out_splits_offsets, int64_t* sig, int64_t epoch, rocshmem_team_t team) {
  int mype = rocshmem_team_my_pe(team);
  int npes = rocshmem_team_n_pes(team);
  auto output_splits = out_splits_offsets;
  auto output_offsets = out_splits_offsets + npes;
  int64_t* sigA = sig;
  int64_t* sigB = sig + npes;
  int64_t* dest = sig + 3 * npes;
  int tid = threadIdx.x;
  CUDA_KERNEL_ASSERT(npes <= THREADS_PER_BLOCK);
  // Round A: tell each peer how many elements I send it (peer P, slot mype).
  if (tid < npes) {
    int pg = rocshmem_team_translate_pe(team, tid, ROCSHMEM_TEAM_WORLD);
    rocshmem_int64_p(output_splits + mype, input_splits[tid], pg);
    rocshmem_fence();
    rocshmem_longlong_atomic_set(reinterpret_cast<long long*>(sigA + mype), epoch, pg);
  }
  if (tid < npes)
    rocshmem_longlong_wait_until(reinterpret_cast<long long*>(sigA + tid), ROCSHMEM_CMP_GE, epoch);
  __syncthreads();
  __shared__ int64_t roff[THREADS_PER_BLOCK];
  prefixSum(roff, output_splits, npes);
  __syncthreads();
  if (tid < npes) {
    output_offsets[tid] = roff[tid];  // caller contract
    // Round B: tell source `tid` the offset at which it should write into me.
    int pg = rocshmem_team_translate_pe(team, tid, ROCSHMEM_TEAM_WORLD);
    rocshmem_int64_p(dest + mype, roff[tid], pg);
    rocshmem_fence();
    rocshmem_longlong_atomic_set(reinterpret_cast<long long*>(sigB + mype), epoch, pg);
  }
}

// Push this rank's per-peer chunks straight into each peer's receive buffer with
// putmem. Multi-block: blocks_per_peer blocks split each destination's chunk.
//   send_data    : local send buffer (symmetric)
//   recv_data    : peer receive buffer (symmetric-addressed on the target PE)
//   input_splits : int64[npes] IN - #elements this rank sends to each peer
//   sig          : signal pad; waits on offset-ready ([npes,2npes)) from exchange
//   epoch        : per-call token
//   stride       : bytes per row (dim-0 element stride)
//   team         : rocSHMEM team
__global__ void allToAllVPush(void* send_data, void* recv_data, int64_t* input_splits, int64_t* sig, int64_t epoch, size_t stride, rocshmem_team_t team) {
  int mype = rocshmem_team_my_pe(team);
  int npes = rocshmem_team_n_pes(team);
  int64_t* sigB = sig + npes;
  int64_t* dest = sig + 3 * npes;
  int bid = blockIdx.x, tid = threadIdx.x;
  int blocks_per_peer = max(gridDim.x / npes, 1);
  CUDA_KERNEL_ASSERT(npes <= THREADS_PER_BLOCK);
  if (tid < npes)
    rocshmem_longlong_wait_until(reinterpret_cast<long long*>(sigB + tid), ROCSHMEM_CMP_GE, epoch);
  __syncthreads();
  __shared__ int64_t soff[THREADS_PER_BLOCK];  // my send-buffer offsets
  prefixSum(soff, input_splits, npes);
  __syncthreads();
  for (int i = bid / blocks_per_peer; i < npes; i += gridDim.x / blocks_per_peer) {
    int peer = (mype + i) % npes;
    int pg = rocshmem_team_translate_pe(team, peer, ROCSHMEM_TEAM_WORLD);
    size_t peer_bytes = (size_t)input_splits[peer] * stride;
    int slot = bid % blocks_per_peer;
    size_t base = peer_bytes / blocks_per_peer;
    size_t block_off = base * slot;
    // last block of each peer absorbs the remainder so no tail bytes are dropped
    // when peer_bytes is not a multiple of blocks_per_peer.
    size_t block_size = (slot == blocks_per_peer - 1) ? (peer_bytes - block_off) : base;
    size_t src_off = (size_t)soff[peer] * stride + block_off;
    size_t dst_off = (size_t)dest[peer] * stride + block_off;
    rocshmem_putmem_nbi_wg(
        (char*)recv_data + dst_off, (char*)send_data + src_off, block_size, pg);
  }
  rocshmem_quiet();  // finish this PE's puts (kernel boundary guarantees all blocks done)
}

// Per-op completion barrier: tell every peer our push to it is done, then wait
// until every source has signalled us -> our receive buffer is fully written.
//   sig   : signal pad; uses the data-done array ([2npes,3npes))
//   epoch : per-call token
//   team  : rocSHMEM team
//
// Separate kernel on purpose: data to a peer is split across `blocks_per_peer`
// blocks in allToAllVPush, and a regular multi-block launch has no grid-wide
// barrier -- so we cannot safely signal "done to peer P" from inside that kernel
// (another block might still be writing into P). allToAllVPush *finishing* on the
// stream is the barrier that guarantees all blocks' puts + quiets are complete;
// only then do we raise the per-peer completion signals here.
__global__ void allToAllVComplete(int64_t* sig, int64_t epoch, rocshmem_team_t team) {
  int mype = rocshmem_team_my_pe(team);
  int npes = rocshmem_team_n_pes(team);
  int64_t* sigC = sig + 2 * npes;
  int tid = threadIdx.x;
  if (tid < npes) {
    int pg = rocshmem_team_translate_pe(team, tid, ROCSHMEM_TEAM_WORLD);
    rocshmem_longlong_atomic_set(reinterpret_cast<long long*>(sigC + mype), epoch, pg);
  }
  __syncthreads();
  if (tid < npes)
    rocshmem_longlong_wait_until(reinterpret_cast<long long*>(sigC + tid), ROCSHMEM_CMP_GE, epoch);
}

// Variable-length all-to-all with split info kept on the device (see the kernel
// header above for the device-driven push/signal design).
//   input               : send buffer (symmetric), tokens laid out per-peer
//   out                 : receive buffer (symmetric)
//   in_splits           : int64[npes] - #elements this rank sends to each peer
//   out_splits_offsets  : int64[2, npes], both OUT - row 0 = output splits
//                         (per-peer recv counts), row 1 = output offsets
//                         (exclusive prefix sum of row 0). This tensor's signal
//                         pad is used as per-peer signaling scratch, so it must
//                         not be shared with another concurrent symm-mem op.
void all_to_all_vdev(
    at::Tensor& input,
    at::Tensor& out,
    at::Tensor& in_splits,
    at::Tensor& out_splits_offsets,
    std::string group_name) {
  auto input_hdl = c10d::symmetric_memory::rendezvous(input, group_name);
  auto out_hdl = c10d::symmetric_memory::rendezvous(out, group_name);
  auto in_splits_hdl = c10d::symmetric_memory::rendezvous(in_splits, group_name);
  auto out_splits_offsets_hdl = c10d::symmetric_memory::rendezvous(out_splits_offsets, group_name);
  int world_size = input_hdl->get_world_size();
  void* input_ptr = input.data_ptr();
  void* output_ptr = out.mutable_data_ptr();
  int64_t* in_splits_ptr = (int64_t*)(in_splits.const_data_ptr());
  int64_t* out_splits_offsets_ptr = (int64_t*)(out_splits_offsets.mutable_data_ptr());

  TORCH_CHECK_EQ(input.device(), out.device());
  auto device = input.device();
  c10::cuda::CUDAGuard guard(device);
  auto& team_manager = TeamManager::get(device);
  auto team = team_manager.get_team(group_name, input_hdl->get_rank_to_global_rank());
  auto stream = at::cuda::getCurrentCUDAStream(device.index());

  int rank = out_splits_offsets_hdl->get_rank();
  auto* sig_ptr = reinterpret_cast<int64_t*>(
      out_splits_offsets_hdl->get_signal_pad_ptrs()[rank]);
  TORCH_CHECK(
      out_splits_offsets_hdl->get_signal_pad_size() >=
          (size_t)world_size * 4 * sizeof(int64_t),
      "signal pad too small for all_to_all_vdev signaling");
  static std::atomic<int64_t> g_epoch{0};
  int64_t epoch = ++g_epoch;

  auto input_size = input.numel() * input.element_size();
  int num_blocks = get_a2a_nblocks(
      input_size, world_size, input_hdl->world_within_direct_access());
  size_t stride_bytes = input.stride(0) * input.element_size();

  exchangeSplitAndOffset<<<dim3(1), dim3(THREADS_PER_BLOCK), 0, stream>>>(
      in_splits_ptr, out_splits_offsets_ptr, sig_ptr, epoch, team);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  allToAllVPush<<<dim3(num_blocks), dim3(THREADS_PER_BLOCK), 0, stream>>>(
      input_ptr, output_ptr, in_splits_ptr, sig_ptr, epoch, stride_bytes, team);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  allToAllVComplete<<<dim3(1), dim3(THREADS_PER_BLOCK), 0, stream>>>(
      sig_ptr, epoch, team);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Start of `all_to_all_vdev_2d`

// This is a warp-scope, exclusive prefix sum. When called by a block of
// threads, each warp will perform an independent prefix sum, concurrently.
// Returns the sum of all elements in the warp.
// `NUM_WARPS` is the number of warps participating the concurrent prefix sum.
template <int NUM_WARPS>
__device__ int64_t prefixSum_warp(int64_t *odata, int64_t *idata, int n) {
  CUDA_KERNEL_ASSERT(n <= WARP_SIZE);

  // Specialize WarpScan for type int
  using WarpScan = ROCM_HIPCUB(at_cuda_detail::cub)::WarpScan<int64_t>;
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

  // Store only valid lanes to avoid out-of-bounds writes when n < WARP_SIZE.
  if (tid < n) {
    odata[tid] = thread_data;
  }
  return warp_aggregate;
}

// This is for abstracting a thread-group-scope, exclusive prefix sum.
// Since we use warp-scope prefix sum, the thread group size is limited to warp size.
#define A2AV_TILE_SIZE WARP_SIZE


__global__ void writeOutputOffsets_2d(
    int64_t* out_splits_offsets,
    int minor_size,
    int major_size,
    int64_t major_align) {
  int nsplits = minor_size * major_size;
  auto output_splits = out_splits_offsets;
  auto source_offsets = out_splits_offsets + nsplits;
  int tid = threadIdx.x;

  constexpr int NUM_TILES = THREADS_PER_BLOCK / A2AV_TILE_SIZE;
  int tileId = tid / A2AV_TILE_SIZE;
  int laneId = tid % A2AV_TILE_SIZE;
  __shared__ int64_t tile_prefix_sums[NUM_TILES][A2AV_TILE_SIZE];
  int nsplits_per_tile = min(minor_size, nsplits - tileId * minor_size);

  __shared__ int64_t len_per_tile[NUM_TILES];
  if (nsplits_per_tile > 0) {
    int64_t my_tile_len = prefixSum_warp<NUM_TILES>(tile_prefix_sums[tileId], output_splits + tileId * minor_size, nsplits_per_tile);
    if (laneId == A2AV_TILE_SIZE - 1) {
      if (major_align != 0) {
        auto aligned_len = (my_tile_len + major_align - 1) / major_align * major_align;
        len_per_tile[tileId] = max(aligned_len, major_align);
      } else {
        len_per_tile[tileId] = my_tile_len;
      }
    }
  }
  __syncthreads();

  __shared__ int64_t start_offset_per_tile[WARP_SIZE];
  prefixSum_warp<1>(start_offset_per_tile, len_per_tile, NUM_TILES);
  __syncthreads();

  tile_prefix_sums[tileId][laneId] += start_offset_per_tile[tileId];
  __syncthreads();

  if (tid < nsplits) {
    source_offsets[tid] = tile_prefix_sums[tid / minor_size][tid % minor_size];
  }
}

// `exchangeSplitAndOffset_2d` is used to exchange output splits and source
// offsets between peers.

/* Arguments:
 * `in_splits_offsets`: input splits and offsets (optional), of size (2, nsplits), or (1, nsplits) if no offsets are provided.
 * `out_splits_offsets`: output splits and offsets, of size (2, nsplits).
 * `mype`: the rank of the current PE.
 * `npes`: the number of PEs.
 * `ne`: the number of experts.
 * `input_dim0`: the size of dim 0 of the input tensor.
 * `rank_is_row_in` is a boolean flag indicating whether the input has ranks as row or experts as row.
*/

/* Template parameters:
 * `HAS_IN_OFFSETS` is a boolean flag indicating whether `in_splits_offsets` has offsets (2nd row) or not.
*/

template <bool HAS_IN_OFFSETS>
__global__ void exchangeSplitAndOffset_2d(int64_t* in_splits_offsets, int64_t* out_splits_offsets, rocshmem_team_t team, int ne, size_t input_dim0, bool rank_is_row_in) {
  CUDA_KERNEL_ASSERT(team != ROCSHMEM_TEAM_INVALID);
  int mype = rocshmem_team_my_pe(team);
  int npes = rocshmem_team_n_pes(team);
  int nsplits = npes * ne;
  auto input_splits = in_splits_offsets;
  auto output_splits = out_splits_offsets;
  // Borrowing the space below as a temporary exchange pad.
  auto source_offsets = out_splits_offsets + nsplits;
  int tid = threadIdx.x;

  int64_t* input_offsets = nullptr;
  if (HAS_IN_OFFSETS) {
    // input offset are provided, so we can use them directly
    input_offsets = in_splits_offsets + nsplits;
  } else {
    // input offset are not provided, so we need to calculate them.
    // Scan input splits to get the source offsets
    __shared__ int64_t peer_offsets[THREADS_PER_BLOCK];
    auto sum_of_splits = prefixSum(peer_offsets, input_splits, nsplits);
    __syncthreads();;
    CUDA_KERNEL_ASSERT(sum_of_splits <= input_dim0 && "sum of splits is larger than input dim\n");
    // Redirect the input splits to the calculated result
    input_offsets = peer_offsets;
  }

  // Use 1 block to do the exchange
  if (tid < nsplits) {
    int peer, e, dst_offset;
    if (rank_is_row_in) {
      peer = tid / ne;
      e = tid % ne;
      dst_offset = e * npes + mype;
    } else {  // expert is row in input
      peer = tid % npes;
      e = tid / npes;
      dst_offset = mype * ne + e;
    }
    // This does a transpose from rank-major order to expert-major order
    // (or vice versa).
    auto split_val = input_splits[tid];
    CUDA_KERNEL_ASSERT(split_val >= 0 && "split value is negative\n");
    auto peer_global = rocshmem_team_translate_pe(team, peer, ROCSHMEM_TEAM_WORLD);
    rocshmem_int64_p(source_offsets + dst_offset, input_offsets[tid], peer_global);
    rocshmem_int64_p(output_splits + dst_offset, split_val, peer_global);
  }
  rocshmem_barrier_wg();
}


// This kernel is used to do the actual data exchange.
// `out_splits_offsets` has the same definition as in `exchangeSplitAndOffset_2d`.
// `stride` is the stride at dim 0, unit in byte.
// For meaning of `mype` and `npes`, see the docstring of `all_to_all_vdev_2d`.
// `major_align` is the alignment at dim 0, unit in element. If 0, no alignment is needed.

// `rank_is_row_out` is a boolean flag indicating whether the output has ranks as rows or experts as rows.
// In dispatch case, rank_is_row_out = false, major_size = ne, minor_size = npes.
// In combine case, rank_is_row_out = true, major_size = npes, minor_size = ne.

__global__ void allToAllV_2d(void *send_data, void *recv_data, int64_t* in_splits, int64_t* out_splits_offsets, size_t stride, int minor_size, int major_size, int64_t major_align, bool rank_is_row_out, rocshmem_team_t team) {
  int nsplits = minor_size * major_size;
  auto output_splits = out_splits_offsets;
  auto source_offsets = out_splits_offsets + nsplits;
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  // Split the thread block into tiles
  constexpr int NUM_TILES = THREADS_PER_BLOCK / A2AV_TILE_SIZE;
  int tileId = tid / A2AV_TILE_SIZE;
  int laneId = tid % A2AV_TILE_SIZE;
  // Each tile calculates its own prefix sum
  __shared__ int64_t tile_prefix_sums[NUM_TILES][A2AV_TILE_SIZE];
  // A tile takes care of minor_size worth of splits
  int nsplits_per_tile = min(minor_size, nsplits - tileId * minor_size);
  // TODO: currently it is assumed that the number of PE's is smaller than
  // `A2AV_TILE_SIZE` bc the warp-scope prefix sum can only handle up to
  // WARP_SIZE elements
  CUDA_KERNEL_ASSERT(minor_size <= A2AV_TILE_SIZE && "minor_size is too large\n");
  // Similarly, the number of experts per rank is also assumed to be smaller
  // than `NUM_TILES`
  CUDA_KERNEL_ASSERT(major_size <= NUM_TILES && "major_size is too large\n");

  // Total length of each tile
  __shared__ int64_t len_per_tile[NUM_TILES];
  // When `nsplits` is small, not every tile gets data to sum. They can skip
  // this local prefix sum.
  if (nsplits_per_tile > 0) {
    // Each tile calculates its own prefix sum, return value is the sum of all elements in the tile.
    int64_t my_tile_len = prefixSum_warp<NUM_TILES>(tile_prefix_sums[tileId], output_splits + tileId * minor_size, nsplits_per_tile);
    // Last thread in each tile does the up aligning.
    if (laneId == A2AV_TILE_SIZE - 1) {
      if (major_align != 0) {  // Needs alignment
        auto aligned_len = (my_tile_len + major_align - 1) / major_align * major_align;
        // In case `aligned_len` is 0, we set it to `major_align` to avoid an
        // empty bin, bc cutlass currently does not support it. See
        // https://github.com/pytorch/pytorch/issues/152668.
        len_per_tile[tileId] = max(aligned_len, major_align);
      } else {  // 0 means alignment not needed
        len_per_tile[tileId] = my_tile_len;
      }
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
    int row = eid / minor_size;
    int col = eid % minor_size;
    // Amount from `peer` for `e`
    auto peer_size = output_splits[eid] * stride;
    auto source_offset = source_offsets[eid] * stride;
    auto e_offset = tile_prefix_sums[row][col];
    auto write_offset = e_offset * stride;
    auto peer_global = rocshmem_team_translate_pe(team, rank_is_row_out ? row : col, ROCSHMEM_TEAM_WORLD);
    rocshmem_getmem_nbi_wg(
      (char*)recv_data + write_offset,
      (char*)send_data + source_offset,
      peer_size,
      peer_global);  // peer's global index
  }
  rocshmem_quiet();
}

void all_to_all_vdev_2d(
    at::Tensor& input,
    at::Tensor& out,
    at::Tensor& in_splits,
    at::Tensor& out_splits_offsets,
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
  auto in_splits_hdl = c10d::symmetric_memory::rendezvous(in_splits, group_name);
  auto out_splits_offsets_hdl = c10d::symmetric_memory::rendezvous(out_splits_offsets, group_name);
  int world_size = input_hdl->get_world_size();
  // TODO: world_size is currently limited by the number of elements in a WarpScan.
  TORCH_CHECK(world_size <= A2AV_TILE_SIZE, "world_size must be smaller than A2AV_TILE_SIZE", A2AV_TILE_SIZE);

  // If `major_align` is not provided, use 1 as the default value.
  int64_t major_align_val = major_align.value_or(1);
  TORCH_CHECK(major_align_val > 0, "major_align must be positive");

  void* input_ptr = input.data_ptr();
  void* output_ptr = out.mutable_data_ptr();
  int64_t* in_splits_ptr = (int64_t*)(in_splits.data_ptr());
  int64_t* out_splits_offsets_ptr = (int64_t*)(out_splits_offsets.mutable_data_ptr());

  // Shape checks
  TORCH_CHECK(in_splits.is_contiguous()
      && out_splits_offsets.is_contiguous()
      && input.is_contiguous()
      && out.is_contiguous(),
      "input, out, in_splits and out_splits_offsets must be contiguous");
  auto in_split_shape = in_splits.sizes();
  auto out_split_shape = out_splits_offsets.sizes();
  TORCH_CHECK(out_split_shape.size() == 2
      && out_split_shape[0] == 2
      && out_split_shape[1] == in_split_shape[0]
      && in_split_shape[0] % world_size == 0,
      "out_splits_offsets must be 2D with 2 rows, "
      "each row must be a multiple of world_size");

  // Consistency checks
  TORCH_CHECK(input.dtype() == out.dtype()
      && input.stride(0) == out.stride(0),
      "input and out must have the same dtype and same stride at dim 0");
  TORCH_CHECK(in_splits.scalar_type() == at::kLong
      && out_splits_offsets.scalar_type() == at::kLong,
      "splits and offsets must be int64");

  // Number of experts per rank
  int ne = in_split_shape[0] / world_size;
  constexpr int NUM_TILES = THREADS_PER_BLOCK / A2AV_TILE_SIZE;
  TORCH_CHECK(ne <= NUM_TILES, "Number of experts must be smaller than NUM_TILES", NUM_TILES);

  // Set device context for getting the stream and launching kernels below
  auto device = input.device();
  TORCH_CHECK(device.type() == at::DeviceType::CUDA &&
      out.device() == device &&
      in_splits.device() == device &&
      out_splits_offsets.device() == device,
      "all tensor arguments must be on the same CUDA device");
  c10::cuda::CUDAGuard guard(device);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto& team_manager = TeamManager::get(device);
  auto team = team_manager.get_team(group_name, input_hdl->get_rank_to_global_rank());

  // Exchange output splits and source offsets
  auto input_dim0 = input.size(0);
  bool rank_is_row_in = true;
  exchangeSplitAndOffset_2d<false><<<dim3(1), dim3(THREADS_PER_BLOCK), 0, stream>>>(
      in_splits_ptr, out_splits_offsets_ptr, team,
      ne, input_dim0, rank_is_row_in);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  C10_CUDA_CHECK(hipStreamSynchronize(stream));
  rocshmem::rocshmem_barrier_all();
  // CTA Tuning
  // Naive for now, use 1 block per expert.
  // Total number of blocks is limited to 64 (intra-node) or 8 (inter-node).
  int num_blocks = ::min(world_size * ne, world_size > 8 ? 8 : 64);

  // Stride at dim 0
  size_t stride_bytes = input.stride(0) * input.element_size();
  bool rank_is_row_out = !rank_is_row_in;

  allToAllV_2d<<<dim3(num_blocks), dim3(THREADS_PER_BLOCK), 0, stream>>>(
      input_ptr, output_ptr,
      in_splits_ptr, out_splits_offsets_ptr,
      stride_bytes, world_size,
      ne, major_align_val, rank_is_row_out, team);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // allToAllV_2d uses a regular multi-block launch with no grid-wide sync, so
  // in-kernel writeback can race with other blocks still reading source_offsets.
  writeOutputOffsets_2d<<<dim3(1), dim3(THREADS_PER_BLOCK), 0, stream>>>(
      out_splits_offsets_ptr, world_size, ne, major_align_val);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  C10_CUDA_CHECK(hipStreamSynchronize(stream));
}

void all_to_all_vdev_2d_offset(
    at::Tensor& input,
    at::Tensor& out,
    at::Tensor& in_splits_offsets,
    at::Tensor& out_splits_offsets,
    std::string group_name) {
  /* Perform a 2D AllToAllv shuffle operation, with input split and offset
   * information provided on device. The input offsets are not required to be
   * exact prefix sum of the input splits, i.e. paddings are allowed between the
   * split chunks. The paddings, however, will not be transferred to peer
   * ranks.

   * In Mixture of Experts models, this operation can be used to combine tokens
   * processed by experts on parallel ranks. This operation can be viewed as an
   * "reverse" operation to the `all_to_all_vdev_2d` operation (which shuffles
   * tokens to experts).

   * Arguments:
   *  - `input` is the input tensor
   *  - `out` is the output tensor
   *  - `in_splits_offsets` is a 2D tensor of size (2, `ne` * `world_size`). In the
        scenario of Mixture-of-Experts models, `ne` is the number of experts per
        rank. The rows of `in_splits_offsets` are (in order):
        input splits (IN) and
        input offsets (IN)
   *  - `out_splits_offsets` is a 2D tensor of size (2, `world_size` * `ne`). The
        rows are (in order):
        output splits (OUT) and
        output offsets (OUT).
   *  - `group_name` is the name of the group to use for the collective operation.
  */
  auto input_hdl = c10d::symmetric_memory::rendezvous(input, group_name);
  auto out_hdl = c10d::symmetric_memory::rendezvous(out, group_name);
  auto out_splits_offsets_hdl = c10d::symmetric_memory::rendezvous(out_splits_offsets, group_name);
  auto in_splits_offsets_hdl = c10d::symmetric_memory::rendezvous(in_splits_offsets, group_name);
  int rank = input_hdl->get_rank();
  int world_size = input_hdl->get_world_size();
  constexpr int NUM_TILES = THREADS_PER_BLOCK / A2AV_TILE_SIZE;
  TORCH_CHECK(world_size <= NUM_TILES, "world_size must be smaller than NUM_TILES", NUM_TILES);

  int64_t major_align_val = 0;

  void* input_ptr = input.data_ptr();
  void* output_ptr = out.mutable_data_ptr();
  int64_t* out_splits_offsets_ptr = (int64_t*)(out_splits_offsets.mutable_data_ptr());
  int64_t* in_splits_offsets_ptr = (int64_t*)(in_splits_offsets.data_ptr());

  // Shape checks
  TORCH_CHECK(out_splits_offsets.is_contiguous()
      && in_splits_offsets.is_contiguous()
      && input.is_contiguous()
      && out.is_contiguous(),
      "input, out, in_splits_offsets and out_splits_offsets must be contiguous");
  auto out_split_shape = out_splits_offsets.sizes();
  auto in_split_shape = in_splits_offsets.sizes();
  TORCH_CHECK(in_split_shape.size() == 2
      && in_split_shape[0] == 2
      && in_split_shape[1] % world_size == 0,
      "in_splits_offsets must be 2D with 2 rows, "
      "each row must be a multiple of world_size");

  // Consistency checks
  TORCH_CHECK(input.dtype() == out.dtype()
      && input.stride(0) == out.stride(0),
      "input and out must have the same dtype and same stride at dim 0");
  TORCH_CHECK(out_splits_offsets.scalar_type() == at::kLong
      && in_splits_offsets.scalar_type() == at::kLong,
      "splits and offsets must be int64");

  // Number of experts per rank
  int ne = in_split_shape[1] / world_size;
  // TODO: number of experts is currently limited by the number of elements in a WarpScan.
  TORCH_CHECK(ne <= A2AV_TILE_SIZE, "Number of experts must be smaller than A2AV_TILE_SIZE", A2AV_TILE_SIZE);

  // Set device context for getting the stream and launching kernels below
  auto device = input.device();
  TORCH_CHECK(device.type() == at::DeviceType::CUDA &&
      out.device() == device &&
      in_splits_offsets.device() == device &&
      out_splits_offsets.device() == device,
      "all tensor arguments must be on the same CUDA device");
  c10::cuda::CUDAGuard guard(device);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto& team_manager = TeamManager::get(device);
  auto team = team_manager.get_team(group_name, input_hdl->get_rank_to_global_rank());

  // Exchange output splits and source offsets
  auto input_dim0 = input.size(0);
  bool rank_is_row_in = false;
  exchangeSplitAndOffset_2d<true><<<dim3(1), dim3(THREADS_PER_BLOCK), 0, stream>>>(
      in_splits_offsets_ptr,
      out_splits_offsets_ptr,
      team,
      ne, input_dim0, rank_is_row_in);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  C10_CUDA_CHECK(hipStreamSynchronize(stream));
  rocshmem::rocshmem_barrier_all();
  // CTA Tuning
  // Naive for now, use 1 block per expert.
  // Total number of blocks is limited to 64 (intra-node) or 8 (inter-node).
  int num_blocks = ::min(world_size * ne, world_size > 8 ? 8 : 64);

  // Stride at dim 0
  size_t stride_bytes = input.stride(0) * input.element_size();
  bool rank_is_row_out = !rank_is_row_in;

  allToAllV_2d<<<dim3(num_blocks), dim3(THREADS_PER_BLOCK), 0, stream>>>(
      input_ptr,
      output_ptr,
      in_splits_offsets_ptr,
      out_splits_offsets_ptr,
      stride_bytes,
      ne,
      world_size,
      major_align_val,
      rank_is_row_out,
      team);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  writeOutputOffsets_2d<<<dim3(1), dim3(THREADS_PER_BLOCK), 0, stream>>>(
      out_splits_offsets_ptr, ne, world_size, major_align_val);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  C10_CUDA_CHECK(hipStreamSynchronize(stream));
}

} // namespace c10d::nvshmem_extension

TORCH_LIBRARY_IMPL(symm_mem, CUDA, m) {
  m.impl("nvshmem_broadcast", c10d::nvshmem_extension::nvshmem_broadcast);
  m.impl("nvshmem_put", c10d::nvshmem_extension::nvshmem_put);
  m.impl("nvshmem_get", c10d::nvshmem_extension::nvshmem_get);
  m.impl("nvshmem_wait_for_signal", c10d::nvshmem_extension::nvshmem_wait_for_signal);
  m.impl("nvshmem_put_with_signal", c10d::nvshmem_extension::nvshmem_put_with_signal);
  m.impl("nvshmem_all_to_all", c10d::nvshmem_extension::nvshmem_all_to_all);
  m.impl("all_to_all_vdev", c10d::nvshmem_extension::all_to_all_vdev);
  m.impl("all_to_all_vdev_2d", c10d::nvshmem_extension::all_to_all_vdev_2d);
  m.impl("all_to_all_vdev_2d_offset", c10d::nvshmem_extension::all_to_all_vdev_2d_offset);
}
