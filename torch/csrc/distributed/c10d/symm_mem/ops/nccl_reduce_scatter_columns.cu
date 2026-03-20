#include <c10/cuda/CUDAGuard.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_extension.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_devcomm_manager.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/NCCLSymmetricMemory.hpp>

// Simultaneously reduce N column-block 2-D tensors from a shared input buffer,
// routing each to a specific destination rank (dst_ranks[i]).  Only the
// destination rank writes the reduced value; all other ranks participate in the
// LSA barrier but do no writes.
//
// Column blocks are described by inclusive-prefix-sum offsets: block i spans
// input[:, offsets[i-1] : offsets[i]] (offsets[-1] == 0 by convention).
// All blocks must have equal width (uniform kernel requirement).
//
// If offsets is nullopt, columns are divided into group_size equal-width blocks.
// If dst_ranks is nullopt, blocks are distributed round-robin across ranks.
//
// Ownership must be balanced: every rank must own the same number of blocks
// (N % group_size == 0 and dst_ranks distributes evenly).

namespace c10d::nccl_extension {

using namespace c10d::symmetric_memory;

// Kernel requires device-side API: ncclLsaReduceSum.
#ifdef NCCL_DEVICE_HAS_REDUCE_COPY

#define RS_MAX_COL_BLOCKS 64          // max number of column blocks (N)
#define RS_MAX_CTAS_PER_COL_BLOCK 16  // max CTAs assigned to one column block's rows
// Threads per CTA; defaults to a medium value to fit medium-width blocks.
#define RS_THREADS_PER_CTA 128
// Total LSA barrier slots needed: one per CTA across all col blocks.
#define RS_MAX_CTA_COUNT (RS_MAX_COL_BLOCKS * RS_MAX_CTAS_PER_COL_BLOCK)

// Per-slot data passed to the kernel in a single struct to avoid multiple
// kernel arguments.  Indexed by owned slot (0..n_owned-1).
struct ReduceScatterColumnsInfo {
  size_t byte_offsets[RS_MAX_COL_BLOCKS]; // byte offset into the NCCL window
  void* dst_ptrs[RS_MAX_COL_BLOCKS];      // output pointer (contiguous)
  int64_t dst_cols[RS_MAX_COL_BLOCKS];    // number of columns in the output
};

// Grid: dim3(n_owned, ctas_per_col_block).
// blockIdx.x selects the owned slot; blockIdx.y tiles rows within that slot.
// Each CTA holds a dedicated LSA barrier (index = blockIdx.x * gridDim.y + blockIdx.y)
// so all ranks synchronize per-CTA rather than globally.
//
// UseMultimem=true: uses ncclMultimemReduceSum for hardware reduction via
// NVLink multicast; requires devcomm created with lsaMultimem=true.
// UseMultimem=false: uses ncclLsaReduceSum (software reduce via LSA reads).
template <typename T, bool UseMultimem>
__global__ void reduce_scatter_columns_kernel(
    ncclWindow_t window,
    ReduceScatterColumnsInfo info,
    int rows,
    int64_t outer_stride, // row stride of the input buffer (in elements)
    ncclDevComm devComm) {
  const int slot = blockIdx.x;
  const int local_block = blockIdx.y;
  const ncclCoopCta coop{};

  // One LSA barrier per CTA; all ranks must call both syncs unconditionally.
  ncclLsaBarrierSession<ncclCoopCta> bar{
      coop,
      devComm,
      ncclTeamLsa(devComm),
      devComm.lsaBarrier,
      blockIdx.x * gridDim.y + blockIdx.y};
  // Acquire: wait until all peers have written their data into the window.
  bar.sync(coop, cuda::memory_order_acquire);

  const size_t base_byte_offset = info.byte_offsets[slot];
  T* dst_base = reinterpret_cast<T*>(info.dst_ptrs[slot]);
  const int cols = static_cast<int>(info.dst_cols[slot]);

  // Each CTA handles a strided subset of rows; the reduce reads from all peers
  // and writes cols elements starting at dst_row.
  for (int row = local_block; row < rows; row += gridDim.y) {
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
// per-slot ReduceScatterColumnsInfo, and launches the kernel.
// See file-level comment for semantics.
void nccl_reduce_scatter_columns(
    const at::Tensor& input,
    at::TensorList out,
    const std::string& group_name,
    std::optional<at::IntArrayRef> offsets,
    std::optional<at::IntArrayRef> dst_ranks,
    const std::string& red_op) {
#ifdef NCCL_DEVICE_HAS_REDUCE_COPY
  TORCH_CHECK(
      red_op == "sum",
      "nccl_reduce_scatter_columns: only red_op='sum' is supported, got '",
      red_op,
      "'");

  TORCH_CHECK(
      input.dim() == 2,
      "nccl_reduce_scatter_columns: input must be 2-D");
  TORCH_CHECK(
      input.stride(-1) == 1,
      "nccl_reduce_scatter_columns: innermost dimension must be contiguous "
      "(stride[-1] == 1)");

  // rendezvous retrieves the symmetric memory handle; the tensor must have
  // been allocated via empty_strided_p2p with the NCCL backend.
  auto symm_mem = c10d::symmetric_memory::rendezvous(input, group_name);
  TORCH_CHECK(
      symm_mem != nullptr,
      "nccl_reduce_scatter_columns: input must be allocated via NCCL symmetric "
      "memory (use empty_strided_p2p with NCCL backend)");

  auto* nccl_hdl = dynamic_cast<NCCLSymmetricMemory*>(symm_mem.get());
  TORCH_CHECK(
      nccl_hdl != nullptr,
      "nccl_reduce_scatter_columns: requires NCCL symmetric memory backend");

  c10::cuda::CUDAGuard guard(input.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto device = input.device();

  auto& manager = c10d::symmetric_memory::NCCLDevCommManager::get(device);
  ncclComm_t comm = manager.get_comm(group_name);

  const bool use_multimem = nccl_hdl->has_multicast_support();

  // The devcomm is cached per (group, key); create it on first use.
  // lsaBarrierCount must cover the maximum number of concurrent CTAs.
  // lsaMultimem is set when the allocation has multicast support, so that
  // devComm.lsaMultimem is valid for ncclMultimemReduceSum in the kernel.
  static constexpr char const kDevcommKey[] = "nccl_reduce_scatter_columns";
  auto devcomm_opt = manager.get_devcomm(group_name, kDevcommKey);
  if (!devcomm_opt) {
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.lsaBarrierCount = RS_MAX_CTA_COUNT;
    reqs.lsaMultimem = use_multimem;
    ncclDevComm devcomm;
    C10D_NCCL_CHECK(
        ncclDevCommCreate(comm, &reqs, &devcomm),
        "ncclDevCommCreate failed in nccl_reduce_scatter_columns");
    devcomm_opt = manager.register_devcomm(group_name, devcomm, kDevcommKey);
  }
  ncclDevComm& devcomm = devcomm_opt->get();

  const int my_rank = devcomm.rank;
  const int group_size = devcomm.nRanks;

  // Determine n_col_blocks: from offsets if given, else group_size (equal-width default).
  const int n_col_blocks = offsets.has_value()
      ? static_cast<int>(offsets->size())
      : group_size;

  // Fill dst_ranks default: round-robin across ranks.
  std::vector<int64_t> dst_ranks_vec;
  at::IntArrayRef effective_dst_ranks;
  if (dst_ranks.has_value()) {
    effective_dst_ranks = *dst_ranks;
  } else {
    dst_ranks_vec.resize(n_col_blocks);
    for (int i = 0; i < n_col_blocks; i++) {
      dst_ranks_vec[i] = i % group_size;
    }
    effective_dst_ranks = at::IntArrayRef(dst_ranks_vec);
  }

  // Fill offsets default: divide input columns equally among group_size blocks.
  std::vector<int64_t> offsets_vec;
  at::IntArrayRef effective_offsets;
  int64_t cols;
  if (offsets.has_value()) {
    effective_offsets = *offsets;
    cols = effective_offsets[0];
    TORCH_CHECK(cols > 0, "nccl_reduce_scatter_columns: column width must be positive");
    // TODO: Remove this check when we support variable-width blocks.
    for (int i = 1; i < n_col_blocks; i++) {
      const int64_t w = effective_offsets[i] - effective_offsets[i - 1];
      TORCH_CHECK(
          w == cols,
          "nccl_reduce_scatter_columns: all column blocks must have equal width");
    }
    TORCH_CHECK(
        effective_offsets[n_col_blocks - 1] <= input.size(1),
        "nccl_reduce_scatter_columns: offsets exceed input column count");
  } else {
    const int64_t total_cols = input.size(1);
    TORCH_CHECK(
        total_cols % group_size == 0,
        "nccl_reduce_scatter_columns: input columns (", total_cols,
        ") must be divisible by group size (", group_size, ")");
    cols = total_cols / group_size;
    offsets_vec.resize(n_col_blocks);
    for (int i = 0; i < n_col_blocks; i++) {
      offsets_vec[i] = (i + 1) * cols;
    }
    effective_offsets = at::IntArrayRef(offsets_vec);
  }

  TORCH_CHECK(n_col_blocks > 0, "nccl_reduce_scatter_columns: must have at least one block");
  TORCH_CHECK(
      n_col_blocks <= RS_MAX_COL_BLOCKS,
      "nccl_reduce_scatter_columns: too many column blocks: ", n_col_blocks,
      " (max ", RS_MAX_COL_BLOCKS, ")");
  TORCH_CHECK(
      static_cast<int>(effective_dst_ranks.size()) == n_col_blocks,
      "nccl_reduce_scatter_columns: dst_ranks.size() must match offsets.size()");

  const int rows = static_cast<int>(input.size(0));
  const int64_t outer_stride = input.stride(0);

  // Collect owned blocks (in order) to validate out and compute ctas_per_col_block.
  std::vector<int> owned_indices;
  for (int i = 0; i < n_col_blocks; i++) {
    if (static_cast<int>(effective_dst_ranks[i]) == my_rank) {
      owned_indices.push_back(i);
    }
  }
  const int n_owned = static_cast<int>(owned_indices.size());
  TORCH_CHECK(
      n_owned * group_size == n_col_blocks,
      "nccl_reduce_scatter_columns: dst_ranks must distribute blocks evenly "
      "(rank owns ", n_owned, "/", n_col_blocks, ", group_size=", group_size, ")");

  TORCH_CHECK(
      static_cast<int>(out.size()) == n_owned,
      "nccl_reduce_scatter_columns: out.size() must be ", n_owned);
  for (int j = 0; j < n_owned; j++) {
    TORCH_CHECK(
        out[j].size(0) == rows && out[j].size(1) == cols,
        "nccl_reduce_scatter_columns: out[", j, "] must have shape (", rows, ", ", cols, ")");
    TORCH_CHECK(
        out[j].is_contiguous(),
        "nccl_reduce_scatter_columns: out[", j, "] must be contiguous");
    TORCH_CHECK(
        out[j].scalar_type() == input.scalar_type(),
        "nccl_reduce_scatter_columns: out[", j, "] must have the same dtype as input");
  }

  // ctas_per_col_block: enough CTAs to cover all rows*cols elements, capped at the
  // maximum to bound the LSA barrier count.
  const int numel = rows * static_cast<int>(cols);
  const int unroll = 4 * 16 / static_cast<int>(input.element_size());
  const int elems_per_cta = RS_THREADS_PER_CTA * unroll;
  const int ctas_per_col_block = std::max(1, std::min(
      (numel + elems_per_cta - 1) / elems_per_cta,
      RS_MAX_CTAS_PER_COL_BLOCK));
  const size_t window_base_offset = nccl_hdl->get_offset();

  // Build the per-slot info struct.  byte_offsets encodes the position of each
  // owned column block within the symmetric memory window.
  ReduceScatterColumnsInfo info;
  for (int j = 0; j < n_owned; j++) {
    const int i = owned_indices[j];
    const int64_t col_start = (i > 0 ? effective_offsets[i - 1] : 0);
    info.byte_offsets[j] =
        window_base_offset +
        static_cast<size_t>(input.storage_offset() + col_start) *
            input.element_size();
    info.dst_ptrs[j] = out[j].data_ptr();
    info.dst_cols[j] = cols;
  }

  auto window = nccl_hdl->get_window();
  TORCH_CHECK(window != nullptr, "nccl_reduce_scatter_columns: NCCL window is null");

  // Launch one CTA per (owned col block, row_tile) pair.  All n_col_blocks * ctas_per_col_block
  // CTAs across all ranks participate in the LSA barrier, so every rank must
  // launch the same grid shape even if n_owned == 0.
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16,
      at::kHalf,
      input.scalar_type(),
      "nccl_reduce_scatter_columns",
      [&]() {
        if (use_multimem) {
          reduce_scatter_columns_kernel<scalar_t, true>
              <<<dim3(n_owned, ctas_per_col_block),
                 RS_THREADS_PER_CTA,
                 0,
                 stream>>>(window, info, rows, outer_stride, devcomm);
        } else {
          reduce_scatter_columns_kernel<scalar_t, false>
              <<<dim3(n_owned, ctas_per_col_block),
                 RS_THREADS_PER_CTA,
                 0,
                 stream>>>(window, info, rows, outer_stride, devcomm);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
#else
  TORCH_CHECK(
      false,
      "nccl_reduce_scatter_columns requires NCCL >= 2.29.7 with reduce copy support");
#endif // NCCL_DEVICE_HAS_REDUCE_COPY
}

} // namespace c10d::nccl_extension
