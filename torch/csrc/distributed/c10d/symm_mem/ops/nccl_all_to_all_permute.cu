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

namespace {

// Caller must ensure 16-byte aligned addresses and nbytes divisible by 16.
__device__ inline void copy_bytes_vec16_aligned(
    const char* src,
    char* dst,
    size_t nbytes,
    size_t tid,
    size_t stride) {
  const size_t n_vec = nbytes / 16;
  constexpr int kUnroll = 4;
  size_t vec_idx = tid;
  for (; vec_idx + static_cast<size_t>(kUnroll - 1) * stride < n_vec;
       vec_idx += static_cast<size_t>(kUnroll) * stride) {
    at::native::memory::Vec<16> chunk[kUnroll];
#pragma unroll 4
    for (int u = 0; u < kUnroll; ++u) {
      const size_t i = vec_idx + static_cast<size_t>(u) * stride;
      chunk[u] = at::native::memory::ld_vec<16>(src + i * 16);
    }
#pragma unroll 4
    for (int u = 0; u < kUnroll; ++u) {
      const size_t i = vec_idx + static_cast<size_t>(u) * stride;
      at::native::memory::st_vec<16>(dst + i * 16, chunk[u]);
    }
  }
  for (; vec_idx < n_vec; vec_idx += stride) {
    const char* src_ptr = src + vec_idx * 16;
    char* dst_ptr = dst + vec_idx * 16;
    auto v = at::native::memory::ld_vec<16>(src_ptr);
    at::native::memory::st_vec<16>(dst_ptr, v);
  }
}

} // namespace

constexpr int A2A_MAX_SLOTS = 64;           // max group size (p)
constexpr int A2A_MAX_CTAS_PER_SLOT = 16;   // max CTAs assigned to one slot's rows
constexpr int A2A_THREADS_PER_CTA = 128;
constexpr int A2A_MAX_CTA_COUNT = A2A_MAX_SLOTS * A2A_MAX_CTAS_PER_SLOT;

// Grid: dim3(p, ctas_per_slot).
//   blockIdx.x = peer_idx — LSA peer to read from (matches output slot index)
//   blockIdx.y           — row tile within that peer's slot
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
  const int local_block = blockIdx.y;
  const ncclCoopCta coop{};

  ncclLsaBarrierSession<ncclCoopCta> bar{
      coop,
      devComm,
      ncclTeamLsa(devComm),
      devComm.lsaBarrier,
      blockIdx.x * gridDim.y + blockIdx.y};
  bar.sync(coop, cuda::memory_order_acquire);

  unsigned char* dst_peer_base =
      out + static_cast<size_t>(peer_idx) * peer_stride_bytes;

  CUDA_KERNEL_ASSERT((base_src_byte_offset & 15) == 0);
  CUDA_KERNEL_ASSERT((reinterpret_cast<uintptr_t>(dst_peer_base) & 15) == 0);

  for (int i = local_block; i < num_rows; i += gridDim.y) {
    const size_t row_byte_offset =
        base_src_byte_offset + static_cast<size_t>(i) * src_row_stride_bytes;
    const void* src_row = ncclGetLsaPointer(window, row_byte_offset, peer_idx);
    unsigned char* dst_row =
        dst_peer_base + static_cast<size_t>(i) * dst_row_stride_bytes;

    copy_bytes_vec16_aligned(
        reinterpret_cast<const char*>(src_row),
        reinterpret_cast<char*>(dst_row),
        copy_row_bytes,
        coop.thread_rank(),
        coop.size());
  }

  bar.sync(coop, cuda::memory_order_release);
}

#endif // NCCL_HAS_SYMMEM_DEVICE_SUPPORT

// Host entry point.  Validates arguments, builds the devcomm (cached), and
// launches the kernel.  See file-level comment for semantics.
void nccl_all_to_all_permute(
    const at::Tensor& input,
    at::Tensor& out,
    int64_t scatter_dim,
    int64_t gather_dim,
    const std::string& group_name) {
#ifdef NCCL_HAS_SYMMEM_DEVICE_SUPPORT
  TORCH_CHECK(
      input.stride(-1) == 1,
      "nccl_all_to_all_permute: innermost dimension must be contiguous (stride[-1] == 1)");
  const bool col_scatter =
      (scatter_dim == 1 && gather_dim == 0);
  const bool row_scatter =
      (scatter_dim == 0 && gather_dim == 1);
  TORCH_CHECK(
      col_scatter || row_scatter,
      "nccl_all_to_all_permute: unsupported (scatter_dim, gather_dim) = (", scatter_dim, ", ",
      gather_dim, "); supported pairs are (1, 0) and (0, 1)");

  auto symm_mem = c10d::symmetric_memory::rendezvous(input, group_name);
  TORCH_CHECK(
      symm_mem != nullptr,
      "nccl_all_to_all_permute: input must be allocated via NCCL symmetric memory "
      "(use empty_strided_p2p with NCCL backend)");

  auto* nccl_hdl = dynamic_cast<NCCLSymmetricMemory*>(symm_mem.get());
  TORCH_CHECK(nccl_hdl != nullptr, "nccl_all_to_all_permute: requires NCCL symmetric memory backend");

  c10::cuda::CUDAGuard guard(input.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto device = input.device();

  auto& manager = c10d::symmetric_memory::NCCLDevCommManager::get(device);
  ncclComm_t comm = manager.get_comm(group_name);

  static constexpr char const kDevcommKey[] = "nccl_all_to_all_permute";
  auto devcomm_opt = manager.get_devcomm(group_name, kDevcommKey);
  if (!devcomm_opt) {
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.lsaBarrierCount = A2A_MAX_CTA_COUNT;
    ncclDevComm devcomm;
    C10D_NCCL_CHECK(
        ncclDevCommCreate(comm, &reqs, &devcomm),
        "ncclDevCommCreate failed in nccl_all_to_all_permute");
    devcomm_opt = manager.register_devcomm(group_name, devcomm, kDevcommKey);
  }
  ncclDevComm& devcomm = devcomm_opt->get();

  const int my_rank = devcomm.rank;
  const int p = devcomm.nRanks;

  TORCH_CHECK(
      p <= A2A_MAX_SLOTS,
      "nccl_all_to_all_permute: group size (", p, ") exceeds maximum supported (", A2A_MAX_SLOTS, ")");

  TORCH_CHECK(out.is_contiguous(), "nccl_all_to_all_permute: out must be contiguous");
  TORCH_CHECK(
      out.scalar_type() == input.scalar_type(),
      "nccl_all_to_all_permute: out must have the same dtype as input");

  auto window = nccl_hdl->get_window();
  TORCH_CHECK(window != nullptr, "nccl_all_to_all_permute: NCCL window is null");

  const size_t window_base_offset = nccl_hdl->get_offset();

  const int64_t esize = input.element_size();
  const size_t tensor_leading_offset =
      window_base_offset +
      static_cast<size_t>(input.storage_offset()) * static_cast<size_t>(esize);
  TORCH_CHECK(
      tensor_leading_offset % 16 == 0,
      "nccl_all_to_all_permute: tensor byte offset within the symmetric window must be 16-byte aligned");
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0,
      "nccl_all_to_all_permute: input tensor data pointer must be 16-byte aligned");
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
      "nccl_all_to_all_permute: output tensor data pointer must be 16-byte aligned");

  const int unroll = 4 * 16 / static_cast<int>(input.element_size());
  const int elems_per_cta = A2A_THREADS_PER_CTA * unroll;
  int ctas_per_slot = 1;

  if (col_scatter) {
    const int rows = static_cast<int>(input.size(0));
    int64_t total_cols = 0;
    int local_cols = 0;
    if (input.dim() == 2) {
      total_cols = input.size(1);
      TORCH_CHECK(
          total_cols % p == 0,
          "nccl_all_to_all_permute: input columns (", total_cols, ") must be divisible by group size (",
          p, ")");
      local_cols = static_cast<int>(total_cols / p);
    } else {
      TORCH_CHECK(
          input.dim() == 3,
          "nccl_all_to_all_permute: for scatter_dim=1, gather_dim=0, input must be 2-D or 3-D");
      TORCH_CHECK(
          input.size(1) == p,
          "nccl_all_to_all_permute: 3-D input must have shape [rows, G, local_cols] with size(1) equal "
          "to group size (",
          p, "); got ",
          input.size(1));
      const int64_t local_cols_i64 = input.size(2);
      total_cols = static_cast<int64_t>(p) * local_cols_i64;
      TORCH_CHECK(
          input.stride(1) == local_cols_i64 && input.stride(0) == total_cols,
          "nccl_all_to_all_permute: 3-D input must be row-major contiguous in the last two dimensions "
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
        "nccl_all_to_all_permute: out must have shape [", p, ", ", rows, ", ", local_cols, "] or [",
        static_cast<int64_t>(p) * static_cast<int64_t>(rows), ", ", local_cols,
        "] for scatter_dim=1, gather_dim=0");

    const size_t row_bytes =
        static_cast<size_t>(local_cols) * static_cast<size_t>(esize);
    TORCH_CHECK(
        row_bytes % 16 == 0,
        "nccl_all_to_all_permute: local column span in bytes (local_cols * element_size) must be "
        "divisible by 16 for vectorized copy");

    ctas_per_slot = std::max(1, std::min(
        (rows * local_cols + elems_per_cta - 1) / elems_per_cta,
        A2A_MAX_CTAS_PER_SLOT));

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
          "nccl_all_to_all_permute: input rows (", total_rows, ") must be divisible by group size (",
          p, ")");
      local_rows = static_cast<int>(total_rows / p);
    } else {
      TORCH_CHECK(
          input.dim() == 3,
          "nccl_all_to_all_permute: for scatter_dim=0, gather_dim=1, input must be 2-D or 3-D");
      TORCH_CHECK(
          input.size(0) == p,
          "nccl_all_to_all_permute: 3-D input must have shape [G, local_rows, cols] with size(0) equal "
          "to group size (",
          p, "); got ",
          input.size(0));
      local_rows = static_cast<int>(input.size(1));
      const int64_t cols_i64 = input.size(2);
      cols = static_cast<int>(cols_i64);
      const int64_t stride01 = static_cast<int64_t>(local_rows) * cols_i64;
      TORCH_CHECK(
          input.stride(1) == cols_i64 && input.stride(0) == stride01,
          "nccl_all_to_all_permute: 3-D input must be row-major contiguous in the last two dimensions "
          "(stride(1)=cols, stride(0)=local_rows*cols)");
    }

    const bool out_shape_3d =
        out.dim() == 3 && out.size(0) == local_rows && out.size(1) == p && out.size(2) == cols;
    const bool out_shape_2d = out.dim() == 2 && out.size(0) == local_rows &&
        out.size(1) == static_cast<int64_t>(p) * static_cast<int64_t>(cols);
    TORCH_CHECK(
        out_shape_3d || out_shape_2d,
        "nccl_all_to_all_permute: out must have shape [", local_rows, ", ", p, ", ", cols, "] or [",
        local_rows, ", ", static_cast<int64_t>(p) * static_cast<int64_t>(cols),
        "] for scatter_dim=0, gather_dim=1");

    const size_t row_bytes =
        static_cast<size_t>(cols) * static_cast<size_t>(esize);
    TORCH_CHECK(
        row_bytes % 16 == 0,
        "nccl_all_to_all_permute: full row in bytes (cols * element_size) must be divisible by 16 "
        "for vectorized copy");

    ctas_per_slot = std::max(1, std::min(
        (local_rows * cols + elems_per_cta - 1) / elems_per_cta,
        A2A_MAX_CTAS_PER_SLOT));

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
  TORCH_CHECK(false, "nccl_all_to_all_permute requires NCCL >= 2.28 with symmetric memory device support");
#endif // NCCL_HAS_SYMMEM_DEVICE_SUPPORT
}

} // namespace c10d::nccl_extension
