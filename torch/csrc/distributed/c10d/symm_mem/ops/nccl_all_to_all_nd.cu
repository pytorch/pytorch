#include <c10/cuda/CUDAGuard.h>
#include <c10/macros/Macros.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_extension.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_devcomm_manager.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/NCCLSymmetricMemory.hpp>

// Permute-free all-to-all for Ulysses-style sequence parallelism.
//
// (scatter_dim=1, gather_dim=0): input [rows, p*local_cols] or [rows, p, local_cols];
//   out [p, rows, local_cols] or [p*rows, local_cols].
//   Each rank r reads column block r from every peer's window.
//
// (scatter_dim=0, gather_dim=1): input [p*local_rows, cols] or [p, local_rows, cols];
//   out [local_rows, p, cols] or [local_rows, p*cols].
//   Each rank r reads row block r from every peer's window into the gather-dim slice for that peer.
//
// Synchronization uses per-CTA LSA barriers (same as nccl_reduce_scatter_columns)
// so all ranks must launch the same grid shape.

namespace c10d::nccl_extension {

using namespace c10d::symmetric_memory;

#ifdef NCCL_HAS_SYMMEM_DEVICE_SUPPORT

constexpr int A2A_MAX_SLOTS = 64;           // max group size (p)
constexpr int A2A_MAX_CTAS_PER_SLOT = 16;   // max CTAs assigned to one slot
constexpr int A2A_THREADS_PER_CTA = 256;
constexpr int A2A_MAX_CTA_COUNT = A2A_MAX_SLOTS * A2A_MAX_CTAS_PER_SLOT;
// Target 16-byte vectors per thread before adding another CTA to a slot.
constexpr int64_t A2A_VECS_PER_THREAD = 4;

// clang-format off
// Decomposition.  For col-scatter this rank copies its own column block out of
// every source row of a peer's [rows, G*local_cols] matrix; the unit of work is
// one `copy_row_bytes = local_cols*esize`-wide segment (a "row"):
//
//            |<-local_cols->|                            total_cols
//            +--------------+--------------+-----+--------------+
//   row 0    |   rank 0     |   rank 1     | ... |   rank G-1   |
//            +--------------+--------------+-----+--------------+
//   row 1    |   rank 0     |   rank 1     | ... |   rank G-1   |
//            +--------------+--------------+-----+--------------+
//   ...      |     ...      |              |     |              |
//            +--------------+--------------+-----+--------------+
//   row R-1  |   rank 0     |   rank 1     | ... |   rank G-1   |
//            +--------------+--------------+-----+--------------+
//              ^^^^^^^^^^^^  this rank's block, copied from every row
//
// A "row" is wide or narrow, measured in 16-byte vectors:
//
//   wide   (local_cols=1024 bf16 = 128 vecs):  [v0][v1]...[v127]
//   narrow (local_cols=8    bf16 =   1 vec ):  [v0]
//
// One CTA per row wastes threads on narrow rows (a 16-byte row uses 1 of 256
// threads).  Instead the slot is flattened into all (row, vec) pairs and every
// thread grid-strides over them: wide rows stay coalesced (consecutive threads
// -> consecutive vecs of one row), narrow rows pack many rows across the block,
// so no thread idles.
// clang-format on

// Grid: dim3(p, ctas_per_slot).
//   blockIdx.x = peer_idx — LSA peer to read from (matches output slot index)
//   blockIdx.y           — work tile within that peer's slot
//
// Each CTA holds a dedicated LSA barrier so all ranks synchronize per-CTA.
// The acquire ensures the peer has written its data; the release signals that
// this rank is done reading that peer's window memory.
//
// (scatter_dim=1, gather_dim=0): base_src_byte_offset = tensor_leading_offset +
//   my_rank*local_cols*element_size; src_row_stride_bytes = total_cols*esize;
//   copy_row_bytes = local_cols*esize; peer_stride_bytes = rows*local_cols*esize;
//   dst_row_stride_bytes = local_cols*esize; num_rows = rows.
// (scatter_dim=0, gather_dim=1): base_src_byte_offset = tensor_leading_offset +
//   my_rank*local_rows*cols*esize; src_row_stride_bytes = cols*esize;
//   copy_row_bytes = cols*esize; peer_stride_bytes = cols*esize;
//   dst_row_stride_bytes = p*cols*esize; num_rows = local_rows.
__global__ void all_to_all_lsa_kernel(
    ncclWindow_t window,
    size_t base_src_byte_offset, // first source row for this rank (byte offset in window)
    unsigned char* out, // contiguous output
    int num_rows,
    size_t src_row_stride_bytes, // bytes between consecutive source rows
    size_t copy_row_bytes, // bytes per row (vectorized copy width)
    size_t peer_stride_bytes, // bytes between consecutive peer slots in out
    size_t dst_row_stride_bytes, // bytes between consecutive rows within one peer slot
    ncclDevComm devComm) {
  const int peer_idx = blockIdx.x;
  const ncclCoopCta coop{};

  ncclLsaBarrierSession<ncclCoopCta> bar{
      coop,
      devComm,
      ncclTeamLsa(devComm),
      devComm.lsaBarrier,
      blockIdx.x * gridDim.y + blockIdx.y};
  bar.sync(coop, cuda::memory_order_acquire);

  // The LSA pointer is offsettable within the peer's window, so resolve this
  // rank's source block base once and index it by arithmetic rather than
  // re-resolving per row.
  const char* src_peer_base = reinterpret_cast<const char*>(
      ncclGetLsaPointer(window, base_src_byte_offset, peer_idx));
  char* dst_peer_base =
      reinterpret_cast<char*>(out) + static_cast<size_t>(peer_idx) * peer_stride_bytes;

  CUDA_KERNEL_ASSERT((reinterpret_cast<uintptr_t>(src_peer_base) & 15) == 0);
  CUDA_KERNEL_ASSERT((reinterpret_cast<uintptr_t>(dst_peer_base) & 15) == 0);

  // Flatten the slot into 16-byte vectors over (row, col) and grid-stride every
  // thread over it, so no thread idles on a narrow row (one CTA on a single
  // copy_row_bytes-wide row would otherwise use only copy_row_bytes/16 threads).
  // Consecutive vectors in a wide row map to consecutive threads (coalesced);
  // narrow rows spread their few vectors across threads, packing many rows.
  // Unroll the grid-stride loop kUnroll-deep, issuing all loads before the
  // stores so several loads stay in flight per thread (memory-level
  // parallelism), which hides the remote-read latency at the low occupancy this
  // kernel runs at.  The destination offset is cached to avoid recomputing the
  // row/col divide in the store phase.
  constexpr int kUnroll = 4;
  const int64_t vecs_per_row = static_cast<int64_t>(copy_row_bytes >> 4);
  const int64_t total_vecs = static_cast<int64_t>(num_rows) * vecs_per_row;
  const int64_t stride = static_cast<int64_t>(gridDim.y) * blockDim.x;
  int64_t gv = static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
  for (; gv + (kUnroll - 1) * stride < total_vecs; gv += kUnroll * stride) {
    at::native::memory::Vec<16> chunk[kUnroll];
    size_t doff[kUnroll];
#pragma unroll
    for (int k = 0; k < kUnroll; ++k) {
      const int64_t g = gv + static_cast<int64_t>(k) * stride;
      const int64_t row = g / vecs_per_row;
      const int64_t v = g - row * vecs_per_row;
      doff[k] = static_cast<size_t>(row) * dst_row_stride_bytes +
          (static_cast<size_t>(v) << 4);
      const size_t soff = static_cast<size_t>(row) * src_row_stride_bytes +
          (static_cast<size_t>(v) << 4);
      chunk[k] = at::native::memory::ld_vec<16>(src_peer_base + soff);
    }
#pragma unroll
    for (int k = 0; k < kUnroll; ++k) {
      at::native::memory::st_vec<16>(dst_peer_base + doff[k], chunk[k]);
    }
  }
  for (; gv < total_vecs; gv += stride) {
    const int64_t row = gv / vecs_per_row;
    const int64_t v = gv - row * vecs_per_row;
    const size_t soff = static_cast<size_t>(row) * src_row_stride_bytes +
        (static_cast<size_t>(v) << 4);
    const size_t doff = static_cast<size_t>(row) * dst_row_stride_bytes +
        (static_cast<size_t>(v) << 4);
    at::native::memory::st_vec<16>(
        dst_peer_base + doff,
        at::native::memory::ld_vec<16>(src_peer_base + soff));
  }

  bar.sync(coop, cuda::memory_order_release);
}

#endif // NCCL_HAS_SYMMEM_DEVICE_SUPPORT

// Host entry point.  Validates arguments, builds the devcomm (cached), and
// launches the kernel.  See file-level comment for semantics.
void nccl_all_to_all_nd(
    const at::Tensor& input,
    at::Tensor& out,
    int64_t scatter_dim,
    int64_t gather_dim,
    const std::string& group_name) {
#ifdef NCCL_HAS_SYMMEM_DEVICE_SUPPORT
  TORCH_CHECK(
      input.stride(-1) == 1,
      "nccl_all_to_all_nd: innermost dimension must be contiguous (stride[-1] == 1)");
  const bool col_scatter =
      (scatter_dim == 1 && gather_dim == 0);
  const bool row_scatter =
      (scatter_dim == 0 && gather_dim == 1);
  TORCH_CHECK(
      col_scatter || row_scatter,
      "nccl_all_to_all_nd: unsupported (scatter_dim, gather_dim) = (", scatter_dim, ", ",
      gather_dim, "); supported pairs are (1, 0) and (0, 1)");

  auto symm_mem = c10d::symmetric_memory::rendezvous(input, group_name);
  TORCH_CHECK(
      symm_mem != nullptr,
      "nccl_all_to_all_nd: input must be allocated via NCCL symmetric memory "
      "(use empty_strided_p2p with NCCL backend)");

  auto* nccl_hdl = dynamic_cast<NCCLSymmetricMemory*>(symm_mem.get());
  TORCH_CHECK(nccl_hdl != nullptr, "nccl_all_to_all_nd: requires NCCL symmetric memory backend");

  c10::cuda::CUDAGuard guard(input.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto device = input.device();

  auto& manager = c10d::symmetric_memory::NCCLDevCommManager::get(device);
  ncclComm_t comm = manager.get_comm(group_name);

  static constexpr char const kDevcommKey[] = "nccl_all_to_all_nd";
  auto devcomm_opt = manager.get_devcomm(group_name, kDevcommKey);
  if (!devcomm_opt) {
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.lsaBarrierCount = A2A_MAX_CTA_COUNT;
    ncclDevComm devcomm;
    C10D_NCCL_CHECK(
        ncclDevCommCreate(comm, &reqs, &devcomm),
        "ncclDevCommCreate failed in nccl_all_to_all_nd");
    devcomm_opt = manager.register_devcomm(group_name, devcomm, kDevcommKey);
  }
  ncclDevComm& devcomm = devcomm_opt->get();

  const int my_rank = devcomm.rank;
  const int p = devcomm.nRanks;

  TORCH_CHECK(
      p <= A2A_MAX_SLOTS,
      "nccl_all_to_all_nd: group size (", p, ") exceeds maximum supported (", A2A_MAX_SLOTS, ")");

  TORCH_CHECK(out.is_contiguous(), "nccl_all_to_all_nd: out must be contiguous");
  TORCH_CHECK(
      out.scalar_type() == input.scalar_type(),
      "nccl_all_to_all_nd: out must have the same dtype as input");

  auto window = nccl_hdl->get_window();
  TORCH_CHECK(window != nullptr, "nccl_all_to_all_nd: NCCL window is null");

  const size_t window_base_offset = nccl_hdl->get_offset();

  const int64_t esize = input.element_size();
  const size_t tensor_leading_offset =
      window_base_offset +
      static_cast<size_t>(input.storage_offset()) * static_cast<size_t>(esize);
  TORCH_CHECK(
      tensor_leading_offset % 16 == 0,
      "nccl_all_to_all_nd: tensor byte offset within the symmetric window must be 16-byte aligned");
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0,
      "nccl_all_to_all_nd: input tensor data pointer must be 16-byte aligned");
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
      "nccl_all_to_all_nd: output tensor data pointer must be 16-byte aligned");

  constexpr int64_t kVecsPerCta = A2A_THREADS_PER_CTA * A2A_VECS_PER_THREAD;
  int ctas_per_slot = 1;

  if (col_scatter) {
    const int rows = static_cast<int>(input.size(0));
    int64_t total_cols = 0;
    int local_cols = 0;
    if (input.dim() == 2) {
      total_cols = input.size(1);
      TORCH_CHECK(
          total_cols % p == 0,
          "nccl_all_to_all_nd: input columns (", total_cols, ") must be divisible by group size (",
          p, ")");
      local_cols = static_cast<int>(total_cols / p);
    } else {
      TORCH_CHECK(
          input.dim() == 3,
          "nccl_all_to_all_nd: for scatter_dim=1, gather_dim=0, input must be 2-D or 3-D");
      TORCH_CHECK(
          input.size(1) == p,
          "nccl_all_to_all_nd: 3-D input must have shape [rows, G, local_cols] with size(1) equal "
          "to group size (",
          p, "); got ",
          input.size(1));
      const int64_t local_cols_i64 = input.size(2);
      total_cols = static_cast<int64_t>(p) * local_cols_i64;
      TORCH_CHECK(
          input.stride(1) == local_cols_i64 && input.stride(0) == total_cols,
          "nccl_all_to_all_nd: 3-D input must be row-major contiguous in the last two dimensions "
          "(stride(1)=local_cols, stride(0)=G*local_cols)");
      local_cols = static_cast<int>(local_cols_i64);
    }

    const bool out_shape_3d =
        out.dim() == 3 && out.size(0) == p && out.size(1) == rows && out.size(2) == local_cols;
    const bool out_shape_2d = out.dim() == 2 &&
        out.size(0) == static_cast<int64_t>(p) * static_cast<int64_t>(rows) &&
        out.size(1) == local_cols;
    TORCH_CHECK(
        out_shape_3d || out_shape_2d,
        "nccl_all_to_all_nd: out must have shape [", p, ", ", rows, ", ", local_cols, "] or [",
        static_cast<int64_t>(p) * static_cast<int64_t>(rows), ", ", local_cols,
        "] for scatter_dim=1, gather_dim=0");

    const size_t row_bytes =
        static_cast<size_t>(local_cols) * static_cast<size_t>(esize);
    TORCH_CHECK(
        row_bytes % 16 == 0,
        "nccl_all_to_all_nd: local column span in bytes (local_cols * element_size) must be "
        "divisible by 16 for vectorized copy");

    const int64_t total_vecs =
        static_cast<int64_t>(rows) * static_cast<int64_t>(row_bytes >> 4);
    ctas_per_slot = static_cast<int>(std::max<int64_t>(
        1,
        std::min<int64_t>(
            (total_vecs + kVecsPerCta - 1) / kVecsPerCta, A2A_MAX_CTAS_PER_SLOT)));

    const size_t esz_u = static_cast<size_t>(esize);
    const size_t rank_block_elems =
        static_cast<size_t>(my_rank) * static_cast<size_t>(local_cols);
    const size_t base_src_byte_offset =
        tensor_leading_offset + rank_block_elems * esz_u;
    const size_t src_row_stride_bytes =
        static_cast<size_t>(total_cols) * esz_u;
    const size_t peer_stride_bytes =
        static_cast<size_t>(rows * local_cols) * esz_u;
    const size_t dst_row_stride_bytes =
        static_cast<size_t>(local_cols) * esz_u;

    all_to_all_lsa_kernel<<<dim3(p, ctas_per_slot), A2A_THREADS_PER_CTA, 0, stream>>>(
        window,
        base_src_byte_offset,
        reinterpret_cast<unsigned char*>(out.data_ptr()),
        rows,
        src_row_stride_bytes,
        row_bytes,
        peer_stride_bytes,
        dst_row_stride_bytes,
        devcomm);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    int local_rows = 0;
    int cols = 0;
    if (input.dim() == 2) {
      const int64_t total_rows = input.size(0);
      cols = static_cast<int>(input.size(1));
      TORCH_CHECK(
          total_rows % p == 0,
          "nccl_all_to_all_nd: input rows (", total_rows, ") must be divisible by group size (",
          p, ")");
      local_rows = static_cast<int>(total_rows / p);
    } else {
      TORCH_CHECK(
          input.dim() == 3,
          "nccl_all_to_all_nd: for scatter_dim=0, gather_dim=1, input must be 2-D or 3-D");
      TORCH_CHECK(
          input.size(0) == p,
          "nccl_all_to_all_nd: 3-D input must have shape [G, local_rows, cols] with size(0) equal "
          "to group size (",
          p, "); got ",
          input.size(0));
      local_rows = static_cast<int>(input.size(1));
      const int64_t cols_i64 = input.size(2);
      cols = static_cast<int>(cols_i64);
      const int64_t stride01 = static_cast<int64_t>(local_rows) * cols_i64;
      TORCH_CHECK(
          input.stride(1) == cols_i64 && input.stride(0) == stride01,
          "nccl_all_to_all_nd: 3-D input must be row-major contiguous in the last two dimensions "
          "(stride(1)=cols, stride(0)=local_rows*cols)");
    }

    const bool out_shape_3d =
        out.dim() == 3 && out.size(0) == local_rows && out.size(1) == p && out.size(2) == cols;
    const bool out_shape_2d = out.dim() == 2 && out.size(0) == local_rows &&
        out.size(1) == static_cast<int64_t>(p) * static_cast<int64_t>(cols);
    TORCH_CHECK(
        out_shape_3d || out_shape_2d,
        "nccl_all_to_all_nd: out must have shape [", local_rows, ", ", p, ", ", cols, "] or [",
        local_rows, ", ", static_cast<int64_t>(p) * static_cast<int64_t>(cols),
        "] for scatter_dim=0, gather_dim=1");

    const size_t row_bytes =
        static_cast<size_t>(cols) * static_cast<size_t>(esize);
    TORCH_CHECK(
        row_bytes % 16 == 0,
        "nccl_all_to_all_nd: full row in bytes (cols * element_size) must be divisible by 16 "
        "for vectorized copy");

    const int64_t total_vecs =
        static_cast<int64_t>(local_rows) * static_cast<int64_t>(row_bytes >> 4);
    ctas_per_slot = static_cast<int>(std::max<int64_t>(
        1,
        std::min<int64_t>(
            (total_vecs + kVecsPerCta - 1) / kVecsPerCta, A2A_MAX_CTAS_PER_SLOT)));

    const size_t esz_u = static_cast<size_t>(esize);
    const size_t cols_u = static_cast<size_t>(cols);
    const size_t rank_block_elems =
        static_cast<size_t>(my_rank * local_rows) * cols_u;
    const size_t base_src_byte_offset =
        tensor_leading_offset + rank_block_elems * esz_u;
    const size_t src_row_stride_bytes = cols_u * esz_u;
    const size_t peer_stride_bytes = cols_u * esz_u;
    const size_t dst_row_stride_bytes =
        static_cast<size_t>(p) * cols_u * esz_u;

    all_to_all_lsa_kernel<<<dim3(p, ctas_per_slot), A2A_THREADS_PER_CTA, 0, stream>>>(
        window,
        base_src_byte_offset,
        reinterpret_cast<unsigned char*>(out.data_ptr()),
        local_rows,
        src_row_stride_bytes,
        row_bytes,
        peer_stride_bytes,
        dst_row_stride_bytes,
        devcomm);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
#else
  TORCH_CHECK(false, "nccl_all_to_all_nd requires NCCL >= 2.28 with symmetric memory device support");
#endif // NCCL_HAS_SYMMEM_DEVICE_SUPPORT
}

} // namespace c10d::nccl_extension
