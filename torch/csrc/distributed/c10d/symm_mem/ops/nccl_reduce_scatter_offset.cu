#include <c10/cuda/CUDAGuard.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/macros.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_extension.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_devcomm_manager.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/NCCLSymmetricMemory.hpp>

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
// Total LSA barrier slots needed: one per CTA across all owned blocks.
constexpr int RS_MAX_CTA_COUNT = (RS_MAX_BLOCKS_PER_RANK * RS_MAX_CTAS_PER_BLOCK);

// Per-slot data passed to the kernel in a single struct to avoid multiple
// kernel arguments.  Indexed by owned slot (0..n_owned-1).
struct ReduceScatterOffsetsInfo {
  size_t byte_offsets[RS_MAX_BLOCKS_PER_RANK]; // byte offset into the NCCL window
  void* dst_ptrs[RS_MAX_BLOCKS_PER_RANK];      // output pointer (contiguous)
  uint16_t dst_block_size[RS_MAX_BLOCKS_PER_RANK]; // per-slot size along the sharding dim
  uint16_t ctas_offset[RS_MAX_BLOCKS_PER_RANK]; // inclusive prefix sum of per-slot CTA counts
  uint8_t cta_slot[RS_MAX_CTA_COUNT];          // slot index for each flat CTA
  int n_owned;
};

// Grid: 1D, total_ctas = sum of per-slot CTA counts (info.ctas_offset[n_owned]).
// Each CTA belongs to one slot; blockIdx.x is the flat CTA index used as the
// LSA barrier index, ensuring all ranks assign the same index to each logical
// (slot, local_block) pair (because owned_sizes[j] is consistent across ranks).
//
// UseMultimem=true: uses ncclMultimemReduceSum for hardware reduction via
// NVLink multicast; requires devcomm created with lsaMultimem=true.
// UseMultimem=false: uses ncclLsaReduceSum (software reduce via LSA reads).
template <typename T, bool UseMultimem>
__global__ void reduce_scatter_offset_kernel(
    ncclWindow_t window,
    ReduceScatterOffsetsInfo info,
    int fixed_dim_size,   // input.size(1-dim): constant across all slots
    bool col_sharded,     // true when dim==1
    int64_t outer_stride, // row stride of the input buffer (in elements)
    ncclDevComm devComm) {
  // cta_slot maps the flat CTA index to its owned slot.
  const int slot = info.cta_slot[blockIdx.x];
  // ctas_offset is an inclusive prefix sum, so slot_start is the flat index
  // of the first CTA assigned to this slot.
  const int slot_start = slot > 0 ? info.ctas_offset[slot - 1] : 0;
  // local_block is this CTA's position within its slot (0-based row tile index).
  const int local_block = static_cast<int>(blockIdx.x) - slot_start;
  // Number of CTAs sharing this slot; used as the row-loop stride.
  const int ctas_for_slot = info.ctas_offset[slot] - slot_start;
  const ncclCoopCta coop{};

  // One LSA barrier per CTA; all ranks must call both syncs unconditionally.
  ncclLsaBarrierSession<ncclCoopCta> bar{
      coop,
      devComm,
      ncclTeamLsa(devComm),
      devComm.lsaBarrier,
      blockIdx.x};
  // Acquire: wait until all peers have written their data into the window.
  bar.sync(coop, cuda::memory_order_acquire);

  const size_t base_byte_offset = info.byte_offsets[slot]; // start of this block in the window
  T* dst_base = reinterpret_cast<T*>(info.dst_ptrs[slot]); // start of out[slot]
  const int block_size = info.dst_block_size[slot]; // size along the sharding dim
  const int rows = col_sharded ? fixed_dim_size : block_size;
  const int cols = col_sharded ? block_size : fixed_dim_size;

  // Each CTA handles a strided subset of rows; the reduce reads from all peers
  // and writes cols elements starting at dst_row.
  for (int row = local_block; row < rows; row += ctas_for_slot) {
    const size_t row_offset =
        base_byte_offset +
        static_cast<size_t>(row * outer_stride) * sizeof(T);
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

#endif // NCCL_DEVICE_HAS_REDUCE_COPY

// Host entry point.  Validates arguments, resolves defaults, builds the
// per-slot ReduceScatterOffsetsInfo, and launches the kernel.
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

  // The devcomm is cached per (group, key); create it on first use.
  // lsaBarrierCount must cover the maximum number of concurrent CTAs.
  // lsaMultimem is set when the allocation has multicast support, so that
  // devComm.lsaMultimem is valid for ncclMultimemReduceSum in the kernel.
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

  // Per-slot CTA count: sized for each slot independently.  owned_sizes[j] is
  // consistent across ranks, so ctas_offset is identical on every rank, which
  // guarantees all ranks launch the same total CTA count and agree on the
  // flat barrier index for each (slot, local_block) pair.
  const bool col_sharded = (dim == 1);
  const int fixed_dim_size = static_cast<int>(col_sharded ? input.size(0) : input.size(1));
  const int unroll = 4 * 16 / static_cast<int>(input.element_size());
  const int elems_per_cta = RS_THREADS_PER_CTA * unroll;
  const size_t window_base_offset = nccl_hdl->get_offset();

  // Build the per-slot info struct.
  // For dim=1: byte_offsets encodes the column-block start within the window.
  // For dim=0: byte_offsets encodes the row-block start within the window.
  ReduceScatterOffsetsInfo info;
  info.n_owned = n_owned;
  for (int j = 0; j < n_owned; j++) {
    const int i = owned_indices[j];
    const int64_t block_start = (i > 0 ? effective_offsets[i - 1] : 0);
    const size_t elem_offset = col_sharded
        ? static_cast<size_t>(input.storage_offset() + block_start)
        : static_cast<size_t>(input.storage_offset()) +
              static_cast<size_t>(block_start) * outer_stride;
    info.byte_offsets[j] = window_base_offset + elem_offset * input.element_size();
    info.dst_ptrs[j] = out[j].data_ptr();
    info.dst_block_size[j] = static_cast<uint16_t>(owned_sizes[j]);
    const int numel_j = static_cast<int>(owned_sizes[j]) * fixed_dim_size;
    const int ctas_j = std::max(1, std::min(
        (numel_j + elems_per_cta - 1) / elems_per_cta, RS_MAX_CTAS_PER_BLOCK));
    info.ctas_offset[j] = static_cast<uint16_t>((j > 0 ? info.ctas_offset[j - 1] : 0) + ctas_j);
    const int slot_start = j > 0 ? info.ctas_offset[j - 1] : 0;
    for (int k = slot_start; k < info.ctas_offset[j]; ++k) {
      info.cta_slot[k] = static_cast<uint8_t>(j);
    }
  }
  const int total_ctas = info.ctas_offset[n_owned - 1];

  auto window = nccl_hdl->get_window();
  TORCH_CHECK(window != nullptr, "nccl_reduce_scatter_offset: NCCL window is null");

  // Each owned (slot, local_block) pair gets one CTA; the flat CTA index is
  // the LSA barrier index.  All ranks launch the same total_ctas because
  // owned_sizes[j] is consistent, so every rank's ctas_offset is identical.
  AT_DISPATCH_NV_FLOATS(
      input.scalar_type(),
      "nccl_reduce_scatter_offset",
      [&]() {
        if (use_multimem) {
          reduce_scatter_offset_kernel<scalar_t, true>
              <<<total_ctas, RS_THREADS_PER_CTA, 0, stream>>>(
                  window, info, fixed_dim_size, col_sharded, outer_stride, devcomm);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
          reduce_scatter_offset_kernel<scalar_t, false>
              <<<total_ctas, RS_THREADS_PER_CTA, 0, stream>>>(
                  window, info, fixed_dim_size, col_sharded, outer_stride, devcomm);
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
