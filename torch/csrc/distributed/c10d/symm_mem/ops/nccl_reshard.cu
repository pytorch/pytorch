#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/macros.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_extension.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_devcomm_manager.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/NCCLSymmetricMemory.hpp>

// Public C API for the user-window NCCL reshard primitive. It is provided
// by an installed NCCL M2N shared library (libnccl_m2n.so), or by NCCL
// itself once the symbols land in libnccl.so.
// Gated behind `NCCL_HAS_RESHARD_API` so the binding can build a stub when
// the header and shared-library symbols are unavailable.
#if defined(NCCL_HAS_RESHARD_API)
#if !defined(NCCL_HAS_SYMMEM_DEVICE_SUPPORT)
#error "NCCL_HAS_RESHARD_API requires NCCL_HAS_SYMMEM_DEVICE_SUPPORT (NCCL >= 2.28)"
#endif
#include <nccl_m2n.h>
#endif

#include <limits>
#include <mutex>
#include <optional>

namespace c10d::nccl_extension {

using namespace c10d::symmetric_memory;

#if defined(NCCL_HAS_RESHARD_API)

namespace {

ncclDataType_t to_nccl_dtype(at::ScalarType st) {
  return c10d::getNcclDataType(st);
}

void fill_mesh(
    ::ncclMesh_t& m,
    at::IntArrayRef dims,
    int64_t start_rank,
    at::IntArrayRef placement) {
  TORCH_CHECK(
      dims.size() == 2,
      "nccl_reshard: mesh_dims must have length 2, got ",
      dims.size());
  TORCH_CHECK(
      placement.size() == 2,
      "nccl_reshard: placement must have length 2, got ",
      placement.size());
  m.dims[0] = static_cast<int>(dims[0]);
  m.dims[1] = static_cast<int>(dims[1]);
  m.startRank = static_cast<int>(start_rank);
  m.placement[0] = static_cast<int>(placement[0]);
  m.placement[1] = static_cast<int>(placement[1]);
}

std::mutex m2n_lifecycle_mutex;
bool m2n_initialized = false;

void nccl_m2n_init_locked(std::optional<int64_t> max_cta) {
  if (m2n_initialized) {
    TORCH_CHECK(
        !max_cta.has_value(),
        "nccl_m2n_init: NCCL M2N is already initialized; "
        "max_cta must be configured before the first init or reshard call");
    return;
  }

  ::ncclM2nConfig_t config = NCCL_M2N_CONFIG_INITIALIZER;
  ::ncclM2nConfig_t* config_ptr = nullptr;
  if (max_cta.has_value()) {
    TORCH_CHECK(
        *max_cta > 0 && *max_cta <= std::numeric_limits<int>::max(),
        "nccl_m2n_init: max_cta must be a positive int32 value, got ",
        *max_cta);
    config.maxCta = static_cast<int>(*max_cta);
    config_ptr = &config;
  }

  C10D_NCCL_CHECK(
      ::ncclM2nInit(config_ptr), "ncclM2nInit failed in nccl_m2n_init");
  m2n_initialized = true;
}

void ensure_m2n_initialized() {
  std::lock_guard<std::mutex> lock(m2n_lifecycle_mutex);
  nccl_m2n_init_locked(std::nullopt);
}

bool release_m2n_state() {
  std::lock_guard<std::mutex> lock(m2n_lifecycle_mutex);
  const bool should_finalize = m2n_initialized;
  m2n_initialized = false;
  return should_finalize;
}

} // namespace

void nccl_reshard(
    at::Tensor& buf,
    at::IntArrayRef src_local_shape,
    at::IntArrayRef src_mesh_dims,
    int64_t src_mesh_start_rank,
    at::IntArrayRef src_placement,
    at::IntArrayRef dst_local_shape,
    at::IntArrayRef dst_mesh_dims,
    int64_t dst_mesh_start_rank,
    at::IntArrayRef dst_placement,
    const std::string& group_name) {
  TORCH_CHECK(buf.is_cuda(), "nccl_reshard: buf must be a CUDA tensor");
  TORCH_CHECK(buf.is_contiguous(), "nccl_reshard: buf must be contiguous");
  // src_local_shape carries tensor rank on every rank, even when this rank is
  // destination-only and its source-side shape values are zero-filled.
  const int ndims = static_cast<int>(src_local_shape.size());
  TORCH_CHECK(
      ndims >= 1 && ndims <= NCCL_RESHARD_MAX_TENSOR_DIMS,
      "nccl_reshard: ndims must be in [1, ",
      NCCL_RESHARD_MAX_TENSOR_DIMS,
      "], got ",
      ndims);
  TORCH_CHECK(
      static_cast<int>(dst_local_shape.size()) == ndims,
      "nccl_reshard: local shape ranks must match; dst_local_shape rank (",
      dst_local_shape.size(),
      ") != src_local_shape rank (",
      ndims,
      ")");

  // A rank participates as "src" iff its src_local_shape is fully positive,
  // and as "dst" iff its dst_local_shape is fully positive. Zero-shape on a
  // side signals the rank is absent there; ncclDistTensor_t::dataPtr is
  // then NULL, matching the user-window API contract for non-participating
  // ranks (cf. PyTorch DTensor's size-0 local tensor).
  int64_t src_numel = 1;
  int64_t dst_numel = 1;
  for (int d = 0; d < ndims; ++d) {
    TORCH_CHECK(
        src_local_shape[d] >= 0 && dst_local_shape[d] >= 0,
        "nccl_reshard: local shapes must be non-negative; got src=",
        src_local_shape,
        ", dst=",
        dst_local_shape);
    src_numel *= src_local_shape[d];
    dst_numel *= dst_local_shape[d];
  }
  const bool is_src_role = src_numel > 0;
  const bool is_dst_role = dst_numel > 0;
  TORCH_CHECK(
      is_src_role || is_dst_role,
      "nccl_reshard: at least one of src_local_shape or dst_local_shape "
      "must be non-empty; got src=",
      src_local_shape,
      ", dst=",
      dst_local_shape);

  int64_t required_numel = 0;
  if (is_src_role) {
    required_numel = std::max(required_numel, src_numel);
  }
  if (is_dst_role) {
    required_numel = std::max(required_numel, dst_numel);
  }
  TORCH_CHECK(
      buf.numel() >= required_numel,
      "nccl_reshard: buf.numel() (",
      buf.numel(),
      ") must be >= max(src_numel, dst_numel) = ",
      required_numel,
      " (src_numel=",
      src_numel,
      ", dst_numel=",
      dst_numel,
      ")");

  // The buffer must live in NCCL symmetric memory; ncclReshardWithWindow
  // operates on the registered ncclWindow_t.
  auto symm_mem = c10d::symmetric_memory::rendezvous(buf, group_name);
  TORCH_CHECK(
      symm_mem != nullptr,
      "nccl_reshard: buf must be allocated via NCCL symmetric memory "
      "(use symm_mem.empty with NCCL backend)");
  auto* nccl_hdl = dynamic_cast<NCCLSymmetricMemory*>(symm_mem.get());
  TORCH_CHECK(
      nccl_hdl != nullptr,
      "nccl_reshard: requires NCCL symmetric memory backend");

  c10::cuda::CUDAGuard guard(buf.device());
  const auto device_index = buf.device().index();
  auto caller_stream = at::cuda::getCurrentCUDAStream(device_index);
  auto stream = caller_stream;
  const auto raw_caller_stream = caller_stream.stream();
  const bool use_side_stream =
      caller_stream == at::cuda::getDefaultCUDAStream(device_index) ||
      raw_caller_stream == nullptr ||
      raw_caller_stream == cudaStreamLegacy ||
      raw_caller_stream == cudaStreamPerThread;
  at::cuda::CUDAEvent side_stream_start(cudaEventDisableTiming);
  at::cuda::CUDAEvent side_stream_done(cudaEventDisableTiming);

  if (use_side_stream) {
    // ncclReshardWithWindow runs default-stream callers on a library-owned
    // side stream that does not wait for prior default-stream work.  Bridge
    // the caller stream ourselves so the op preserves PyTorch stream ordering.
    stream = at::cuda::getStreamFromPool(
        /*isHighPriority=*/false, device_index);
    side_stream_start.record(caller_stream);
    side_stream_start.block(stream);
  }

  auto& manager =
      c10d::symmetric_memory::NCCLDevCommManager::get(buf.device());
  ncclComm_t comm = manager.get_comm(group_name);
  ncclWindow_t window = nccl_hdl->get_window();
  TORCH_CHECK(window != nullptr, "nccl_reshard: NCCL window is null");

  ::ncclMesh_t src_mesh{};
  ::ncclMesh_t dst_mesh{};
  fill_mesh(src_mesh, src_mesh_dims, src_mesh_start_rank, src_placement);
  fill_mesh(dst_mesh, dst_mesh_dims, dst_mesh_start_rank, dst_placement);

  // Pack ncclDistTensor_t descriptors. Both descriptors are required on
  // every rank — they each carry one side's mesh, which the library reads
  // everywhere to compute who-talks-to-whom. Placements live on the mesh
  // descriptors, while dataPtr=NULL signals the rank does not own a tile on
  // that side.
  ::ncclDistTensor_t src{};
  ::ncclDistTensor_t dst{};
  const ncclDataType_t dtype = to_nccl_dtype(buf.scalar_type());
  src.ndims = ndims;
  src.dtype = dtype;
  src.mesh = &src_mesh;
  src.dataPtr = is_src_role ? buf.mutable_data_ptr() : nullptr;
  for (int d = 0; d < ndims; ++d) {
    src.localShape[d] =
        is_src_role ? static_cast<size_t>(src_local_shape[d]) : 0;
  }
  dst.ndims = ndims;
  dst.dtype = dtype; // src.dtype must equal dst.dtype (same in-place buffer)
  dst.mesh = &dst_mesh;
  dst.dataPtr = is_dst_role ? buf.mutable_data_ptr() : nullptr;
  for (int d = 0; d < ndims; ++d) {
    dst.localShape[d] =
        is_dst_role ? static_cast<size_t>(dst_local_shape[d]) : 0;
  }

  ensure_m2n_initialized();
  C10D_NCCL_CHECK(
      ::ncclReshardWithWindow(comm, window, &src, &dst, stream),
      "ncclReshardWithWindow failed in nccl_reshard");

  if (use_side_stream) {
    side_stream_done.record(stream);
    side_stream_done.block(caller_stream);
  }
}

void nccl_m2n_init(std::optional<int64_t> max_cta) {
  std::lock_guard<std::mutex> lock(m2n_lifecycle_mutex);
  nccl_m2n_init_locked(max_cta);
}

void nccl_m2n_finalize() {
  // Best-effort release of the reshard library's internal caches and
  // transpose buffer. Idempotent: subsequent calls are no-ops. Call this
  // before tearing down the process group; it is not suitable for atexit
  // cleanup after NCCL communicator shutdown. Errors during shutdown are
  // logged, never thrown.
  if (release_m2n_state()) {
    auto rc = ::ncclM2nFinalize();
    if (rc != ncclSuccess) {
      LOG(WARNING) << "ncclM2nFinalize returned " << rc
                   << " - symm_mem.reshard resources may not be fully released.";
    }
  }
}

#else // !NCCL_HAS_RESHARD_API

void nccl_reshard(
    at::Tensor& /*buf*/,
    at::IntArrayRef /*src_local_shape*/,
    at::IntArrayRef /*src_mesh_dims*/,
    int64_t /*src_mesh_start_rank*/,
    at::IntArrayRef /*src_placement*/,
    at::IntArrayRef /*dst_local_shape*/,
    at::IntArrayRef /*dst_mesh_dims*/,
    int64_t /*dst_mesh_start_rank*/,
    at::IntArrayRef /*dst_placement*/,
    const std::string& /*group_name*/) {
  TORCH_CHECK(
      false,
      "nccl_reshard requires the user-window reshard API from NCCL M2N "
      "(libnccl_m2n.so) or an NCCL build exporting it "
      "from libnccl.so. NCCL_HAS_RESHARD_API was not defined at build time.");
}

void nccl_m2n_init(std::optional<int64_t> /*max_cta*/) {
  TORCH_CHECK(
      false,
      "nccl_m2n_init requires the NCCL M2N API "
      "(libnccl_m2n.so) or an NCCL build exporting it "
      "from libnccl.so. NCCL_HAS_RESHARD_API was not defined at build time.");
}

void nccl_m2n_finalize() {
  // Stub when the reshard API isn't compiled in — nothing to release.
}

#endif // NCCL_HAS_RESHARD_API

} // namespace c10d::nccl_extension
