#include <c10/cuda/CUDAGuard.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/cub.cuh>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/macros.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_extension.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_devcomm_manager.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/NCCLSymmetricMemory.hpp>

// AllToAllV with split sizes that live on device (drop-in for the existing
// nvshmem_extension.cu `all_to_all_vdev`). All four tensors must be allocated
// from the NCCL symmetric memory backend (e.g. via empty_strided_p2p).
//
// API matches the NVSHMEM version verbatim:
//   - `input`              : send buffer (contiguous along dim 0)
//   - `out`                : recv buffer (contiguous along dim 0)
//   - `in_splits`          : 1-D int64 tensor of size (npes,)
//                            `in_splits[p]` = number of rows to send to peer p
//   - `out_splits_offsets` : 2-D int64 tensor of shape (2, npes); on return
//                            row 0 holds output splits, row 1 holds output
//                            offsets (the position of each peer's chunk inside
//                            `out`).  During execution row 1 transiently holds
//                            *source* offsets (where to read from on the peer)
//                            -- the kernel rewrites it to write offsets at the
//                            end so the user observes the documented contract.
//
// Algorithm (mirrors the NVSHMEM implementation):
//   Phase 1 (1 CTA, `nccl_a2av_exchange_kernel`):
//     - Each rank computes the exclusive prefix sum of its own `in_splits` to
//       get per-peer source offsets in its send buffer.
//     - Each rank writes to peer p's `out_splits_offsets`:
//         peer.out[      mype] = in_splits[p]                (output split)
//         peer.out[npes+ mype] = source_offset_for(p)        (source offset)
//       via one-sided LSA pointer writes (NVLink load/store).
//     - LSA barrier so all peers' metadata is visible.
//
//   Phase 2 (N CTAs, `nccl_a2av_data_kernel`):
//     - Each rank reads the now-populated `output_splits` and `source_offsets`,
//       computes the exclusive prefix sum of `output_splits` to get write
//       offsets in its own recv buffer.
//     - CTAs are partitioned across peers; each block-cooperatively pulls a
//       slice of `peer.input[source_offset .. source_offset + split)` into
//       `out[write_offset .. write_offset + split)` using `ncclGetLsaPointer`
//       (direct NVLink loads, no in-flight queue / quiet needed -- on
//       completion of the kernel the loads have all retired locally).
//     - CTA 0 overwrites row 1 of `out_splits_offsets` with write offsets so
//       the user-visible contract (row 1 = output offsets) is honoured.
//
//   No exit barrier is added (matches NVSHMEM's behaviour); the user is
//   responsible for synchronising before reusing send buffers, exactly as with
//   the NVSHMEM path. The next collective's entry barrier (or a manual
//   `barrier()` on the symm_mem handle) closes the read-after-write hazard
//   between successive ops.

namespace c10d::nccl_extension {

using namespace c10d::symmetric_memory;

#ifdef NCCL_HAS_SYMMEM_DEVICE_SUPPORT

namespace {

// THREADS_PER_BLOCK also caps the maximum world size we can serve -- one
// thread per peer is used for the metadata exchange and prefix-sum stages.
// For NVLink-domain symmetric memory (NVL8 / NVL72) this is plenty.
constexpr int A2A_THREADS_PER_BLOCK = 128;
constexpr int A2A_MAX_PEERS = A2A_THREADS_PER_BLOCK;
// CTA caps for the data kernel; mirrors the NVSHMEM heuristic.
constexpr int A2A_MAX_BLOCKS_INTRA = 64;
constexpr int A2A_MAX_BLOCKS_INTER = 16;

// Block-cooperative byte copy. Vectorises to 16-B loads when src/dst are both
// 16-B aligned (always true for empty_strided_p2p allocations, but we still
// fall back gracefully on the tail).
__device__ __forceinline__ void block_copy_bytes(
    char* __restrict__ dst,
    const char* __restrict__ src,
    size_t n_bytes) {
  constexpr size_t kVec = 16;
  size_t bulk = 0;
  if ((reinterpret_cast<uintptr_t>(src) % kVec == 0) &&
      (reinterpret_cast<uintptr_t>(dst) % kVec == 0)) {
    bulk = (n_bytes / kVec) * kVec;
    const int4* src4 = reinterpret_cast<const int4*>(src);
    int4* dst4 = reinterpret_cast<int4*>(dst);
    const size_t n_vec = bulk / kVec;
    for (size_t i = threadIdx.x; i < n_vec; i += blockDim.x) {
      dst4[i] = src4[i];
    }
  }
  for (size_t i = bulk + threadIdx.x; i < n_bytes; i += blockDim.x) {
    dst[i] = src[i];
  }
}

// ---------------------------------------------------------------------------
// Phase 1: exchange output splits and source offsets.
// Launched as a single CTA so we can use a CUB block-scan and a single LSA
// barrier slot. All ranks launch one CTA so the per-CTA LSA barrier slot 0
// pairs up cleanly across ranks.
// ---------------------------------------------------------------------------
__global__ void nccl_a2av_exchange_kernel(
    ncclWindow_t in_splits_win,
    size_t in_splits_off,
    ncclWindow_t scratch_win,
    size_t scratch_off,
    ncclDevComm devcomm) {
  using BlockScanT = at_cuda_detail::cub::
      BlockScan<int64_t, A2A_THREADS_PER_BLOCK, at_cuda_detail::cub::BLOCK_SCAN_WARP_SCANS>;
  __shared__ typename BlockScanT::TempStorage temp_storage;
  __shared__ int64_t s_my_offsets[A2A_MAX_PEERS];

  const ncclTeam lsa = ncclTeamLsa(devcomm);
  const int mype = lsa.rank;
  const int npes = lsa.nRanks;
  const int tid = threadIdx.x;
  CUDA_KERNEL_ASSERT(npes <= A2A_MAX_PEERS);

  // Read MY input_splits[tid] from local window.
  int64_t* in_splits = reinterpret_cast<int64_t*>(
      ncclGetLocalPointer(in_splits_win, in_splits_off));
  int64_t my_split = (tid < npes) ? in_splits[tid] : int64_t{0};

  // Exclusive prefix sum -> per-peer source offset in MY send buffer.
  int64_t my_src_off;
  int64_t total;
  BlockScanT(temp_storage).ExclusiveSum(my_split, my_src_off, total);
  if (tid < npes) {
    s_my_offsets[tid] = my_src_off;
  }
  __syncthreads();

  // Each thread tid (tid < npes) writes 2 int64s into peer tid's scratchpad:
  //   peer.scratch[mype]        = my in_splits[tid]      (peer's output split from me)
  //   peer.scratch[npes + mype] = my src_off for peer    (peer's source offset on me)
  if (tid < npes) {
    int64_t* peer_scratch = reinterpret_cast<int64_t*>(
        ncclGetLsaPointer(scratch_win, scratch_off, tid));
    peer_scratch[mype] = my_split;
    peer_scratch[npes + mype] = s_my_offsets[tid];
  }

  // Cross-rank fence: after this returns, all peers have completed their
  // metadata writes into MY scratchpad and the writes are visible to me.
  ncclCoopCta coop{};
  ncclLsaBarrierSession<ncclCoopCta> bar{
      coop,
      devcomm,
      ncclTeamLsa(devcomm),
      devcomm.lsaBarrier,
      /*index=*/0};
  bar.sync(coop, cuda::memory_order_seq_cst);
}

// ---------------------------------------------------------------------------
// Phase 2: pull data from peers using one-sided NVLink loads.
// Each peer is assigned `blocks_per_peer = max(gridDim.x / npes, 1)` CTAs;
// CTAs are rotated by `mype` so different ranks hammer different peers first
// (matches the NVSHMEM heuristic to spread NVLink load).
// ---------------------------------------------------------------------------
__global__ void nccl_a2av_data_kernel(
    ncclWindow_t input_win,
    size_t input_win_off,
    void* output_ptr,
    ncclWindow_t scratch_win,
    size_t scratch_off,
    size_t stride_bytes,
    ncclDevComm devcomm) {
  using BlockScanT = at_cuda_detail::cub::
      BlockScan<int64_t, A2A_THREADS_PER_BLOCK, at_cuda_detail::cub::BLOCK_SCAN_WARP_SCANS>;
  __shared__ typename BlockScanT::TempStorage temp_storage;
  __shared__ int64_t s_write_offsets[A2A_MAX_PEERS];

  const ncclTeam lsa = ncclTeamLsa(devcomm);
  const int mype = lsa.rank;
  const int npes = lsa.nRanks;
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  const int blocks_per_peer = max(gridDim.x / npes, 1);
  CUDA_KERNEL_ASSERT(npes <= A2A_MAX_PEERS);

  // Read the metadata that peers wrote in phase 1.
  int64_t* scratch = reinterpret_cast<int64_t*>(
      ncclGetLocalPointer(scratch_win, scratch_off));
  int64_t* output_splits = scratch;
  int64_t* source_offsets = scratch + npes;

  // Exclusive prefix sum of output_splits -> per-peer write offsets in MY
  // recv buffer.
  int64_t my_split = (tid < npes) ? output_splits[tid] : int64_t{0};
  int64_t my_write_off;
  int64_t total;
  BlockScanT(temp_storage).ExclusiveSum(my_split, my_write_off, total);
  if (tid < npes) {
    s_write_offsets[tid] = my_write_off;
  }
  __syncthreads();

  // Walk peers in rotated order so different ranks start on different peers.
  for (int i = bid / blocks_per_peer; i < npes;
       i += gridDim.x / blocks_per_peer) {
    const int peer = (mype + i) % npes;
    const int64_t peer_split = output_splits[peer];
    if (peer_split == 0) {
      continue;
    }
    const size_t peer_bytes = static_cast<size_t>(peer_split) * stride_bytes;
    const size_t block_bytes = peer_bytes / blocks_per_peer;
    // We assume each peer's chunk divides evenly across its CTAs (matches
    // NVSHMEM's allToAllV; the CTA budget is rounded up to a multiple of npes
    // on the host side). If the peer's chunk is smaller than blocks_per_peer
    // bytes, only block 0 will do useful work for that peer.
    CUDA_KERNEL_ASSERT(block_bytes * blocks_per_peer == peer_bytes);
    const size_t block_off = block_bytes * (bid % blocks_per_peer);
    const size_t src_off_bytes =
        static_cast<size_t>(source_offsets[peer]) * stride_bytes + block_off;
    const size_t dst_off_bytes =
        static_cast<size_t>(s_write_offsets[peer]) * stride_bytes + block_off;

    char* peer_src = reinterpret_cast<char*>(ncclGetLsaPointer(
        input_win, input_win_off + src_off_bytes, peer));
    char* dst = reinterpret_cast<char*>(output_ptr) + dst_off_bytes;
    block_copy_bytes(dst, peer_src, block_bytes);
  }

  // Honour the user-visible contract: row 1 of out_splits_offsets must be the
  // *output* offsets (write offsets in `out`). It currently holds the source
  // offsets we used internally; CTA 0 overwrites it once with the write
  // offsets we just computed.
  if (bid == 0 && tid < npes) {
    source_offsets[tid] = s_write_offsets[tid];
  }
}

// CTA budget for the data kernel. We pad up to a multiple of npes so each
// peer has the same number of CTAs, then cap by an intra-/inter-node ceiling.
int get_a2a_nblocks(size_t input_bytes, int npes, bool intra_node) {
  // 16 B per thread, 8 unrolled iterations per CTA -- the same heuristic the
  // NVSHMEM implementation uses for the inter-rank case.
  constexpr size_t kChunk = 16ull * A2A_THREADS_PER_BLOCK * 8;
  int n = static_cast<int>((input_bytes + kChunk - 1) / kChunk);
  // Round up to a multiple of npes so each peer gets the same number of CTAs.
  n = ((n + npes - 1) / npes) * npes;
  if (n < npes) n = npes;
  const int max_n = intra_node ? A2A_MAX_BLOCKS_INTRA : A2A_MAX_BLOCKS_INTER;
  return std::min(n, max_n);
}

} // namespace

// Host entry point. See file-level comment for semantics.
void nccl_all_to_all_vdev(
    at::Tensor& input,
    at::Tensor& out,
    at::Tensor& in_splits,
    at::Tensor& out_splits_offsets,
    const std::string& group_name) {
  // Rendezvous all four tensors as symmetric memory. We require all four to
  // be NCCL-symmetric; this matches the NVSHMEM contract (all participating
  // tensors live on the symmetric heap).
  auto in_hdl = c10d::symmetric_memory::rendezvous(input, group_name);
  auto out_hdl = c10d::symmetric_memory::rendezvous(out, group_name);
  auto in_splits_hdl =
      c10d::symmetric_memory::rendezvous(in_splits, group_name);
  auto scratch_hdl =
      c10d::symmetric_memory::rendezvous(out_splits_offsets, group_name);

  TORCH_CHECK(
      in_hdl != nullptr && out_hdl != nullptr && in_splits_hdl != nullptr &&
          scratch_hdl != nullptr,
      "nccl_all_to_all_vdev: all of input / out / in_splits / out_splits_offsets "
      "must be allocated via NCCL symmetric memory (use empty_strided_p2p with "
      "the NCCL backend)");

  auto* nccl_in = dynamic_cast<NCCLSymmetricMemory*>(in_hdl.get());
  auto* nccl_in_splits = dynamic_cast<NCCLSymmetricMemory*>(in_splits_hdl.get());
  auto* nccl_scratch = dynamic_cast<NCCLSymmetricMemory*>(scratch_hdl.get());
  TORCH_CHECK(
      nccl_in != nullptr && nccl_in_splits != nullptr &&
          nccl_scratch != nullptr,
      "nccl_all_to_all_vdev: requires the NCCL symmetric memory backend");

  // One-sided LSA loads require every peer in the NCCL team to be in the same
  // direct-access (LSA) domain. Cross-node groups are not supported yet.
  TORCH_CHECK(
      in_hdl->world_within_direct_access(),
      "nccl_all_to_all_vdev: requires intra-node direct peer access for all ranks "
      "(LSA domain); cross-node execution is not supported yet");

  TORCH_CHECK(input.is_contiguous(), "nccl_all_to_all_vdev: input must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "nccl_all_to_all_vdev: out must be contiguous");
  TORCH_CHECK(input.dim() >= 1, "nccl_all_to_all_vdev: input must be at least 1-D");
  TORCH_CHECK(input.dtype() == out.dtype(),
      "nccl_all_to_all_vdev: input and out must have the same dtype");
  TORCH_CHECK(in_splits.scalar_type() == at::kLong,
      "nccl_all_to_all_vdev: in_splits must be int64");
  TORCH_CHECK(out_splits_offsets.scalar_type() == at::kLong,
      "nccl_all_to_all_vdev: out_splits_offsets must be int64");
  TORCH_CHECK(out_splits_offsets.dim() == 2 && out_splits_offsets.size(0) == 2,
      "nccl_all_to_all_vdev: out_splits_offsets must have shape (2, npes), got ",
      out_splits_offsets.sizes());
  TORCH_CHECK(in_splits.is_contiguous() && out_splits_offsets.is_contiguous(),
      "nccl_all_to_all_vdev: in_splits and out_splits_offsets must be contiguous");

  c10::cuda::CUDAGuard guard(input.device());
  auto stream = at::cuda::getCurrentCUDAStream(input.device().index());
  auto device = input.device();

  auto& manager = NCCLDevCommManager::get(device);
  ncclComm_t comm = manager.get_comm(group_name);

  // Cache a devcomm under our own key so we don't trample whatever
  // reduce_scatter_offset (or any other op) has already created on this group.
  static constexpr char const kDevcommKey[] = "nccl_all_to_all_vdev";
  auto devcomm_opt = manager.get_devcomm(group_name, kDevcommKey);
  if (!devcomm_opt) {
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    // We only need a single LSA barrier slot: phase 1's single CTA enters
    // and exits it before phase 2 starts (stream order on each rank, LSA
    // sync across ranks). Phase 2 deliberately has no inter-rank barrier
    // -- callers must synchronise before reusing the send buffer, exactly
    // as with the NVSHMEM version.
    reqs.lsaBarrierCount = 1;
    ncclDevComm devcomm;
    C10D_NCCL_CHECK(
        ncclDevCommCreate(comm, &reqs, &devcomm),
        "ncclDevCommCreate failed in nccl_all_to_all_vdev");
    devcomm_opt = manager.register_devcomm(group_name, devcomm, kDevcommKey);
  }
  ncclDevComm& devcomm = devcomm_opt->get();

  const int npes = devcomm.nRanks;
  TORCH_CHECK(npes <= A2A_MAX_PEERS,
      "nccl_all_to_all_vdev: world size ", npes,
      " exceeds compile-time max ", A2A_MAX_PEERS,
      " (= A2A_THREADS_PER_BLOCK)");
  TORCH_CHECK(in_splits.numel() == npes,
      "nccl_all_to_all_vdev: in_splits.numel() must equal world size, got ",
      in_splits.numel(), " vs ", npes);
  TORCH_CHECK(out_splits_offsets.size(1) == npes,
      "nccl_all_to_all_vdev: out_splits_offsets.size(1) must equal world size, got ",
      out_splits_offsets.size(1), " vs ", npes);

  // Resolve the per-tensor (window, byte-offset) pairs. get_offset() is the
  // tensor's allocation offset within the NCCL window; we add the tensor's
  // own storage_offset so views work correctly.
  ncclWindow_t in_win = nccl_in->get_window();
  size_t in_off = nccl_in->get_offset() +
      static_cast<size_t>(input.storage_offset()) * input.element_size();

  ncclWindow_t in_splits_win = nccl_in_splits->get_window();
  size_t in_splits_off = nccl_in_splits->get_offset() +
      static_cast<size_t>(in_splits.storage_offset()) * in_splits.element_size();

  ncclWindow_t scratch_win = nccl_scratch->get_window();
  size_t scratch_off = nccl_scratch->get_offset() +
      static_cast<size_t>(out_splits_offsets.storage_offset()) *
          out_splits_offsets.element_size();

  TORCH_CHECK(in_win != nullptr && in_splits_win != nullptr && scratch_win != nullptr,
      "nccl_all_to_all_vdev: NCCL window is null");

  // Bytes per row in the input tensor along dim 0; matches NVSHMEM's `stride`
  // argument and is used to scale split counts (which are in row units, not
  // bytes) into byte offsets.
  const size_t stride_bytes =
      static_cast<size_t>(input.stride(0)) * input.element_size();

  // ---- Phase 1: metadata exchange (1 CTA) ----
  nccl_a2av_exchange_kernel<<<1, A2A_THREADS_PER_BLOCK, 0, stream>>>(
      in_splits_win, in_splits_off,
      scratch_win, scratch_off,
      devcomm);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // ---- Phase 2: data exchange (N CTAs) ----
  // input.numel() * element_size is the size of MY send buffer; we choose the
  // CTA budget from that (NVSHMEM does the same -- using the recv size would
  // require the device-side metadata which isn't available host-side until
  // after the alltoallv finishes).
  const size_t input_bytes =
      static_cast<size_t>(input.numel()) * input.element_size();
  const int num_blocks =
      get_a2a_nblocks(input_bytes, npes, /*intra_node=*/true);
  nccl_a2av_data_kernel<<<num_blocks, A2A_THREADS_PER_BLOCK, 0, stream>>>(
      in_win, in_off,
      out.mutable_data_ptr(),
      scratch_win, scratch_off,
      stride_bytes,
      devcomm);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#else // !NCCL_HAS_SYMMEM_DEVICE_SUPPORT

void nccl_all_to_all_vdev(
    at::Tensor& /*input*/,
    at::Tensor& /*out*/,
    at::Tensor& /*in_splits*/,
    at::Tensor& /*out_splits_offsets*/,
    const std::string& /*group_name*/) {
  TORCH_CHECK(false,
      "nccl_all_to_all_vdev requires NCCL >= 2.28 with symmetric memory device API support");
}

#endif // NCCL_HAS_SYMMEM_DEVICE_SUPPORT

} // namespace c10d::nccl_extension
