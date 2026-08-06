#include <c10/cuda/CUDAGuard.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/macros.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_extension.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_devcomm_manager.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/NCCLSymmetricMemory.hpp>

#if defined(NCCL_DEVICE_HAS_REDUCE_COPY) && defined(NCCL_HAS_LSA_PEER_PTR)
#include <mutex>
#include <unordered_map>
// HIP shims required to compile RCCL's device-side reduce/copy API. Must precede
// <nccl_device.h>.
//   1. RCCL selects the real (non-static_assert) reduce_copy implementation on
//      __CUDACC_EXTENDED_LAMBDA__. hipcc does not define it, but HIP supports
//      extended device lambdas natively, so defining it takes the real branch.
//   2. RCCL's int8 sum-reduce specialization uses CUDA's __vadd4 per-byte SIMD
//      intrinsic, which HIP lacks. It is an explicit specialization (compiled
//      eagerly even though these float/half/bf16 tests never use int8), so a
//      byte-wise wrapping-add equivalent must be visible.
#ifndef __CUDACC_EXTENDED_LAMBDA__
#define __CUDACC_EXTENDED_LAMBDA__ 1
#endif
__device__ __forceinline__ unsigned int __vadd4(unsigned int a, unsigned int b) {
  unsigned int res;
  auto* r = reinterpret_cast<unsigned char*>(&res);
  auto* pa = reinterpret_cast<unsigned char*>(&a);
  auto* pb = reinterpret_cast<unsigned char*>(&b);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    r[i] = static_cast<unsigned char>(pa[i] + pb[i]);
  }
  return res;
}
//   3. ATen compiles this TU with __HIP_NO_HALF_OPERATORS__, which strips
//      __half's native operator+. RCCL's OpSum<__half> needs it: although the
//      reduction accumulates in float (AccumulateType<OpSum<half>>::Type ==
//      float), AT_DISPATCH_NV_FLOATS eagerly instantiates the __half kernel.
//      Provide a convert-to-float add so the instantiation compiles; the runtime
//      reduction precision is unaffected. bf16 keeps its own operator+.
#if defined(__HIP_NO_HALF_OPERATORS__)
__device__ __forceinline__ __half operator+(const __half& a, const __half& b) {
  return __float2half(__half2float(a) + __half2float(b));
}
#endif
#include <nccl_device.h>
#endif // NCCL_DEVICE_HAS_REDUCE_COPY && NCCL_HAS_LSA_PEER_PTR

// Simultaneously reduce N blocks of a 2-D input tensor from a symmetric memory
// buffer, routing each block to a specific destination rank (dst_ranks[i]).
// Only the destination rank writes the reduced value to a contiguous output
// tensor, with the same shape as the owned block.
//
// The `dim` argument controls which dimension is sharded (0 or 1):
//   dim=1 (column sharding): each block spans input[:, offsets[i-1]:offsets[i]]
//   dim=0 (row sharding):    each block spans input[offsets[i-1]:offsets[i], :]
//
// Blocks are described by inclusive-prefix-sum offsets along `dim`.
// For each j, out[j] must have the same shape across all ranks (i.e. the j-th
// owned block on every rank must have equal size); different j's may differ.
//
// If offsets is nullopt, input.size(dim) is divided equally into group_size blocks.
// If dst_ranks is nullopt, blocks are distributed round-robin across ranks.
//
// Ownership must be balanced: every rank must own the same number of blocks
// (N % group_size == 0 and dst_ranks distributes evenly).

namespace c10d::nccl_extension {

using namespace c10d::symmetric_memory;

// Kernel requires device-side API: ncclLsaReduceSum.
#ifdef NCCL_DEVICE_HAS_REDUCE_COPY

// Naming conventions in this file:
// "BLOCK" means tensor block (as opposed to CUDA block);
// "CTA" means CUDA block;
// "RS" means Reduce Scatter;
// "slot" means which tensor block a CTA is assigned to.

constexpr int RS_MAX_BLOCKS = 64;           // max total blocks being scattered (N)
constexpr int RS_MAX_BLOCKS_PER_RANK = 16;  // max blocks owned by a single rank
constexpr int RS_MAX_CTAS_PER_BLOCK = 16;   // max CTAs assigned to one block
// Threads per CTA; defaults to a medium value to fit medium-width blocks.
constexpr int RS_THREADS_PER_CTA = 128;
// Upper bound on LSA barrier slots to request for the devcomm.  Per-slot launches
// use at most RS_MAX_CTAS_PER_BLOCK barrier indices each; this over-allocation is
// harmless and keeps a single devcomm requirement across all launch shapes.
constexpr int RS_MAX_CTA_COUNT = (RS_MAX_BLOCKS_PER_RANK * RS_MAX_CTAS_PER_BLOCK);

// Reduces ONE owned block ("slot") per launch.  Grid: 1D, gridDim.x == number of
// CTAs (row tiles) cooperating on this slot; blockIdx.x is both the row-tile
// index and the LSA barrier index.  All ranks launch the per-slot kernels in the
// same order with the same per-slot CTA count (owned_sizes[j] is consistent
// across ranks), so every rank agrees on the barrier index for each row tile.
//
// Slots are launched sequentially on the stream (one launch per owned block) so
// that at most one destination allocation is being written at a time.  On ROCm,
// running the LSA device reduce concurrently across CTAs that write to DIFFERENT
// destination allocations triggers a memory-aperture fault in RCCL's device
// reduce; serializing per destination avoids it.  Row tiles within a single slot
// all write the same destination allocation, which is safe.
//
// UseMultimem=true: uses ncclMultimemReduceSum for hardware reduction via
// multicast; requires devcomm created with lsaMultimem=true.
// UseMultimem=false: uses ncclLsaReduceSum (software reduce via LSA reads).
template <typename T, bool UseMultimem>
__global__ void reduce_scatter_offset_kernel(
    ncclWindow_t window,
    size_t base_byte_offset, // window byte offset of this slot's block start
    T* dst_base,             // this slot's output pointer (contiguous)
    int rows,                // number of rows to reduce for this slot
    int cols,                // elements per row
    int64_t outer_stride,    // row stride of the input buffer (in elements)
    ncclDevComm devComm) {
  const int ctas_for_slot = gridDim.x;
  const int local_block = static_cast<int>(blockIdx.x);
  const ncclCoopCta coop{};

  // One LSA barrier per row tile; all ranks must call both syncs unconditionally.
  ncclLsaBarrierSession<ncclCoopCta> bar{
      coop,
      devComm,
      ncclTeamLsa(devComm),
      devComm.lsaBarrier,
      blockIdx.x};
  // Acquire: wait until all peers have written their data into the window.
  bar.sync(coop, cuda::memory_order_acquire);

  // Each CTA handles a strided subset of rows; the reduce reads from all peers
  // and writes cols elements starting at dst_row.
  for (int row = local_block; row < rows; row += ctas_for_slot) {
    const size_t row_offset =
        base_byte_offset + static_cast<size_t>(row * outer_stride) * sizeof(T);
    T* dst_row = dst_base + row * cols;
    if constexpr (UseMultimem) {
      ncclMultimemReduceSum(
          coop, window, row_offset, dst_row, cols, devComm.lsaMultimem);
    } else {
      ncclLsaReduceSum(coop, window, row_offset, dst_row, cols, devComm);
    }
  }

  // Release: signal peers that we are done reading window memory.
  bar.sync(coop, cuda::memory_order_release);
}

#if defined(NCCL_HAS_LSA_PEER_PTR)
// File-local ncclDevComm cache for the ROCm path. The shared NCCLDevCommManager
// stores ncclDevComm only under NCCL_HAS_SYMMEM_DEVICE_SUPPORT (CUDA): its
// header is included by host-only translation units (init.cpp,
// ProcessGroupNCCL.cpp) compiled with the plain host compiler, which cannot
// parse RCCL's <nccl_device.h> (it needs HIP device builtins). Caching the
// devcomm here keeps the ncclDevComm type confined to hipcc-compiled TUs.
// Keyed by group_name; devcomm lifetime spans the process (not destroyed), same
// as the CUDA registry which only tears down in the manager destructor.
static std::mutex g_rs_devcomm_mutex;
static std::unordered_map<std::string, ncclDevComm> g_rs_devcomm_cache;

static ncclDevComm& get_or_create_rs_devcomm(
    ncclComm_t comm,
    const std::string& group_name,
    bool use_multimem) {
  std::lock_guard<std::mutex> lock(g_rs_devcomm_mutex);
  auto it = g_rs_devcomm_cache.find(group_name);
  if (it == g_rs_devcomm_cache.end()) {
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.lsaBarrierCount = RS_MAX_CTA_COUNT;
    reqs.lsaMultimem = use_multimem;
    ncclDevComm devcomm;
    C10D_NCCL_CHECK(
        ncclDevCommCreate(comm, &reqs, &devcomm),
        "ncclDevCommCreate failed in nccl_reduce_scatter_offset");
    it = g_rs_devcomm_cache.emplace(group_name, devcomm).first;
  }
  return it->second;
}
#endif // NCCL_HAS_LSA_PEER_PTR

#endif // NCCL_DEVICE_HAS_REDUCE_COPY

// Host entry point.  Validates arguments, resolves defaults, and launches one
// reduce kernel per owned slot (sequentially on the stream).
// See file-level comment for semantics.
void nccl_reduce_scatter_offset(
    const at::Tensor& input,
    at::TensorList out,
    const std::string& group_name,
    int64_t dim,
    std::optional<at::IntArrayRef> offsets,
    std::optional<at::IntArrayRef> dst_ranks,
    const std::string& red_op) {
#ifdef NCCL_DEVICE_HAS_REDUCE_COPY
  TORCH_CHECK(
      red_op == "sum",
      "nccl_reduce_scatter_offset: only red_op='sum' is supported, got '", red_op, "'");

  TORCH_CHECK(
      input.dim() == 2,
      "nccl_reduce_scatter_offset: input must be 2-D");
  TORCH_CHECK(
      dim == 0 || dim == 1,
      "nccl_reduce_scatter_offset: dim must be 0 or 1, got ", dim);
  TORCH_CHECK(
      input.stride(-1) == 1,
      "nccl_reduce_scatter_offset: innermost dimension must be contiguous "
      "(stride[-1] == 1)");

  // rendezvous retrieves the symmetric memory handle; the tensor must have
  // been allocated via empty_strided_p2p with the NCCL backend.
  auto symm_mem = c10d::symmetric_memory::rendezvous(input, group_name);
  TORCH_CHECK(
      symm_mem != nullptr,
      "nccl_reduce_scatter_offset: input must be allocated via NCCL symmetric "
      "memory (use empty_strided_p2p with NCCL backend)");

  auto* nccl_hdl = dynamic_cast<NCCLSymmetricMemory*>(symm_mem.get());
  TORCH_CHECK(
      nccl_hdl != nullptr,
      "nccl_reduce_scatter_offset: requires NCCL symmetric memory backend");

  c10::cuda::CUDAGuard guard(input.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto device = input.device();

  auto& manager = c10d::symmetric_memory::NCCLDevCommManager::get(device);
  // Get the host-side communicator.
  ncclComm_t comm = manager.get_comm(group_name);

  const bool use_multimem = nccl_hdl->has_multicast_support();

  // The devcomm is cached per group and created on first use.
  // lsaBarrierCount must cover the maximum number of concurrent CTAs.
  // lsaMultimem is set when the allocation has multicast support, so that
  // devComm.lsaMultimem is valid for ncclMultimemReduceSum in the kernel.
#if defined(NCCL_HAS_LSA_PEER_PTR)
  // ROCm: NCCLDevCommManager cannot hold ncclDevComm (see the file-local cache
  // comment above), so cache it here instead.
  ncclDevComm& devcomm = get_or_create_rs_devcomm(comm, group_name, use_multimem);
#else
  static constexpr char const kDevcommKey[] = "nccl_reduce_scatter_offset";
  auto devcomm_opt = manager.get_devcomm(group_name, kDevcommKey);
  if (!devcomm_opt) {
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.lsaBarrierCount = RS_MAX_CTA_COUNT;
    reqs.lsaMultimem = use_multimem;
    ncclDevComm devcomm;
    C10D_NCCL_CHECK(
        ncclDevCommCreate(comm, &reqs, &devcomm),
        "ncclDevCommCreate failed in nccl_reduce_scatter_offset");
    // Cache the device communicator.
    devcomm_opt = manager.register_devcomm(group_name, devcomm, kDevcommKey);
  }
  ncclDevComm& devcomm = devcomm_opt->get();
#endif

  const int my_rank = devcomm.rank;
  const int group_size = devcomm.nRanks;

  // Determine n_blocks: from offsets if given, else group_size (equal-size default).
  const int n_blocks = offsets.has_value()
      ? static_cast<int>(offsets->size())
      : group_size;
  TORCH_CHECK(
      n_blocks > 0,
      "nccl_reduce_scatter_offset: must have at least one block");

  // Fill dst_ranks default: round-robin across ranks.
  std::vector<int64_t> dst_ranks_vec;
  at::IntArrayRef effective_dst_ranks;
  if (dst_ranks.has_value()) {
    effective_dst_ranks = *dst_ranks;
  } else {
    dst_ranks_vec.resize(n_blocks);
    for (int i = 0; i < n_blocks; i++) {
      dst_ranks_vec[i] = i % group_size;
    }
    effective_dst_ranks = at::IntArrayRef(dst_ranks_vec);
  }

  // Fill offsets default: divide input.size(dim) equally among group_size blocks.
  std::vector<int64_t> offsets_vec;
  at::IntArrayRef effective_offsets;
  if (offsets.has_value()) {
    effective_offsets = *offsets;
    TORCH_CHECK(
        effective_offsets[n_blocks - 1] <= input.size(dim),
        "nccl_reduce_scatter_offset: offsets exceed input size along dim ", dim);
  } else {
    const int64_t total = input.size(dim);
    TORCH_CHECK(
        total % group_size == 0,
        "nccl_reduce_scatter_offset: input.size(", dim, ")=", total,
        " must be divisible by group size (", group_size, ")");
    const int64_t block_size = total / group_size;
    offsets_vec.resize(n_blocks);
    for (int i = 0; i < n_blocks; i++) {
      offsets_vec[i] = (i + 1) * block_size;
    }
    effective_offsets = at::IntArrayRef(offsets_vec);
  }

  TORCH_CHECK(
      n_blocks <= RS_MAX_BLOCKS,
      "nccl_reduce_scatter_offset: too many blocks: ", n_blocks,
      " (max ", RS_MAX_BLOCKS, ")");
  TORCH_CHECK(
      static_cast<int>(effective_dst_ranks.size()) == n_blocks,
      "nccl_reduce_scatter_offset: dst_ranks.size() must match offsets.size()");

  const int64_t outer_stride = input.stride(0);

  // Collect owned blocks (in order).
  std::vector<int> owned_indices;
  for (int i = 0; i < n_blocks; i++) {
    if (static_cast<int>(effective_dst_ranks[i]) == my_rank) {
      owned_indices.push_back(i);
    }
  }
  const int n_owned = static_cast<int>(owned_indices.size());
  TORCH_CHECK(
      n_owned * group_size == n_blocks,
      "nccl_reduce_scatter_offset: dst_ranks must distribute blocks evenly "
      "(rank owns ", n_owned, "/", n_blocks, ", group_size=", group_size, ")");
  TORCH_CHECK(
      n_owned <= RS_MAX_BLOCKS_PER_RANK,
      "nccl_reduce_scatter_offset: too many owned blocks: ", n_owned,
      " (max ", RS_MAX_BLOCKS_PER_RANK, ")");
  // Balance is guaranteed above (n_owned * group_size == n_blocks), so
  // rank_counter[r] never exceeds n_owned during the owned_sizes loop.

  // For each j, out[j] must have the same shape across all ranks.  That means
  // all blocks that are the j-th owned block on their respective rank must have
  // equal size.  Different j's may differ in size.
  //
  // Compute the size for each j by iterating all blocks in order, tracking
  // how many blocks each rank has seen so far (= the j-index for that block).
  std::vector<int64_t> owned_sizes(n_owned, -1);
  {
    std::vector<int> rank_counter(group_size, 0);
    for (int i = 0; i < n_blocks; i++) {
      const int r = static_cast<int>(effective_dst_ranks[i]);
      const int j = rank_counter[r]++;
      const int64_t sz =
          effective_offsets[i] - (i > 0 ? effective_offsets[i - 1] : 0);
      if (owned_sizes[j] < 0) {
        owned_sizes[j] = sz;
      } else {
        TORCH_CHECK(
            sz == owned_sizes[j],
            "nccl_reduce_scatter_offset: all output at position j=", j,
            " must have equal size across all ranks");
      }
    }
  }

  TORCH_CHECK(
      static_cast<int>(out.size()) == n_owned,
      "nccl_reduce_scatter_offset: out.size() must be ", n_owned);
  for (int j = 0; j < n_owned; j++) {
    // dim=1: out[j] shape is (input.size(0), owned_sizes[j])
    // dim=0: out[j] shape is (owned_sizes[j], input.size(1))
    const int64_t exp0 = dim == 1 ? input.size(0) : owned_sizes[j];
    const int64_t exp1 = dim == 1 ? owned_sizes[j] : input.size(1);
    TORCH_CHECK(
        out[j].size(0) == exp0 && out[j].size(1) == exp1,
        "nccl_reduce_scatter_offset: out[", j, "] must have shape (",
        exp0, ", ", exp1, ")");
    TORCH_CHECK(
        out[j].is_contiguous(),
        "nccl_reduce_scatter_offset: out[", j, "] must be contiguous");
    TORCH_CHECK(
        out[j].scalar_type() == input.scalar_type(),
        "nccl_reduce_scatter_offset: out[", j, "] must have the same dtype as input");
  }

  // Per-slot geometry.  owned_sizes[j] is consistent across ranks, so every rank
  // computes the same per-slot CTA count and launches the same per-slot kernels
  // in the same order, which keeps LSA barrier participation consistent.
  const bool col_sharded = (dim == 1);
  const int fixed_dim_size = static_cast<int>(col_sharded ? input.size(0) : input.size(1));
  const int unroll = 4 * 16 / static_cast<int>(input.element_size());
  const int elems_per_cta = RS_THREADS_PER_CTA * unroll;
  const size_t window_base_offset = nccl_hdl->get_window_offset();

  auto window = nccl_hdl->get_window();
  TORCH_CHECK(window != nullptr, "nccl_reduce_scatter_offset: NCCL window is null");

  // Launch one kernel per owned slot, sequentially on the stream.  Reducing to
  // different destination allocations concurrently faults in RCCL's device
  // reduce on ROCm (see the kernel comment); stream-ordered per-slot launches
  // keep at most one destination in flight while still tiling rows across CTAs.
  AT_DISPATCH_NV_FLOATS(
      input.scalar_type(),
      "nccl_reduce_scatter_offset",
      [&]() {
        for (int j = 0; j < n_owned; j++) {
          const int i = owned_indices[j];
          const int64_t block_start = (i > 0 ? effective_offsets[i - 1] : 0);
          const size_t elem_offset = col_sharded
              ? static_cast<size_t>(input.storage_offset() + block_start)
              : static_cast<size_t>(input.storage_offset()) +
                    static_cast<size_t>(block_start) * outer_stride;
          const size_t base_byte_offset =
              window_base_offset + elem_offset * input.element_size();
          auto* dst_base = static_cast<scalar_t*>(out[j].data_ptr());
          const int rows = col_sharded ? fixed_dim_size : static_cast<int>(owned_sizes[j]);
          const int cols = col_sharded ? static_cast<int>(owned_sizes[j]) : fixed_dim_size;
          const int numel_j = static_cast<int>(owned_sizes[j]) * fixed_dim_size;
          const int ctas_j = ::max(1, ::min(
              (numel_j + elems_per_cta - 1) / elems_per_cta, RS_MAX_CTAS_PER_BLOCK));
          if (use_multimem) {
            reduce_scatter_offset_kernel<scalar_t, true>
                <<<ctas_j, RS_THREADS_PER_CTA, 0, stream>>>(
                    window, base_byte_offset, dst_base, rows, cols, outer_stride, devcomm);
          } else {
            reduce_scatter_offset_kernel<scalar_t, false>
                <<<ctas_j, RS_THREADS_PER_CTA, 0, stream>>>(
                    window, base_byte_offset, dst_base, rows, cols, outer_stride, devcomm);
          }
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
      });
#else
  TORCH_CHECK(
      false,
      "nccl_reduce_scatter_offset requires NCCL >= 2.29.7 with reduce copy support");
#endif // NCCL_DEVICE_HAS_REDUCE_COPY
}

} // namespace c10d::nccl_extension
