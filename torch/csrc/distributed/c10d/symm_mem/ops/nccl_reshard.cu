#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/macros.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_extension.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_devcomm_manager.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/NCCLSymmetricMemory.hpp>

// Build a stub unless the external NCCL M2N API is available.
#if defined(NCCL_HAS_RESHARD_API)
#if !defined(NCCL_HAS_SYMMEM_DEVICE_SUPPORT)
#error "NCCL_HAS_RESHARD_API requires NCCL_HAS_SYMMEM_DEVICE_SUPPORT (NCCL >= 2.28)"
#endif
#include <nccl_m2n.h>
#endif

#include <array>
#include <limits>
#include <mutex>
#include <optional>
#include <set>

namespace c10d::nccl_extension {

using namespace c10d::symmetric_memory;

#if defined(NCCL_HAS_RESHARD_API)

namespace {

ncclDataType_t to_nccl_dtype(at::ScalarType st) {
  return c10d::getNcclDataType(st);
}

void fill_mesh(
    ::ncclMesh_t& mesh,
    std::array<int, NCCL_RESHARD_MAX_MESH_DIMS>& mesh_dims,
    at::IntArrayRef dims,
    int64_t start_rank) {
  TORCH_CHECK(
      dims.size() == 2,
      "nccl_reshard: mesh_dims must have length 2, got ",
      dims.size());
  mesh_dims[0] = static_cast<int>(dims[0]);
  mesh_dims[1] = static_cast<int>(dims[1]);
  mesh.ndims = NCCL_RESHARD_MAX_MESH_DIMS;
  mesh.dims = mesh_dims.data();
  mesh.startRank = static_cast<int>(start_rank);
}

std::mutex m2n_lifecycle_mutex;
::ncclM2nHandle_t m2n_handle = nullptr;
std::set<c10::DeviceIndex> m2n_devices;

void nccl_m2n_init_locked(std::optional<int64_t> max_cta) {
  if (m2n_handle != nullptr) {
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

  C10D_NCCL_CHECK(::ncclM2nInit(&m2n_handle, config_ptr), "ncclM2nInit failed in nccl_m2n_init");
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

  // Every rank supplies both shapes; mesh membership determines participation.
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
  // ncclReshardWithWindow operates on the registered symmetric-memory window.
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
  auto stream = at::cuda::getCurrentCUDAStream(device_index);

  auto& manager =
      c10d::symmetric_memory::NCCLDevCommManager::get(buf.device());
  ncclComm_t comm = manager.get_comm(group_name);
  ncclWindow_t window = nccl_hdl->get_window();
  TORCH_CHECK(window != nullptr, "nccl_reshard: NCCL window is null");

  std::array<int, NCCL_RESHARD_MAX_MESH_DIMS> src_mesh_dims_storage;
  std::array<int, NCCL_RESHARD_MAX_MESH_DIMS> dst_mesh_dims_storage;
  ::ncclMesh_t src_mesh = NCCL_M2N_MESH_INITIALIZER;
  ::ncclMesh_t dst_mesh = NCCL_M2N_MESH_INITIALIZER;
  fill_mesh(src_mesh, src_mesh_dims_storage, src_mesh_dims, src_mesh_start_rank);
  fill_mesh(dst_mesh, dst_mesh_dims_storage, dst_mesh_dims, dst_mesh_start_rank);
  TORCH_CHECK(src_placement.size() == src_mesh.ndims && dst_placement.size() == dst_mesh.ndims,
              "nccl_reshard: placement rank must match mesh rank");
  std::array<int, NCCL_RESHARD_MAX_MESH_DIMS> src_placements;
  std::array<int, NCCL_RESHARD_MAX_MESH_DIMS> dst_placements;
  for (int d = 0; d < src_mesh.ndims; ++d) {
    src_placements[d] = static_cast<int>(src_placement[d]);
    dst_placements[d] = static_cast<int>(dst_placement[d]);
  }
  int comm_rank = 0;
  C10D_NCCL_CHECK(::ncclCommUserRank(comm, &comm_rank), "ncclCommUserRank failed in nccl_reshard");
  const auto in_mesh = [comm_rank](const ::ncclMesh_t& mesh) {
    int size = 1;
    for (int d = 0; d < mesh.ndims; ++d) size *= mesh.dims[d];
    return comm_rank >= mesh.startRank && comm_rank < mesh.startRank + size;
  };
  const bool is_src_role = in_mesh(src_mesh);
  const bool is_dst_role = in_mesh(dst_mesh);
  const int64_t required_numel = std::max(src_numel, dst_numel);
  TORCH_CHECK(
      buf.numel() >= required_numel,
      "nccl_reshard: buf must hold the larger source or destination local tile");

  // Every rank supplies both descriptors. A null dataPtr marks an inactive side.
  std::array<size_t, NCCL_RESHARD_MAX_TENSOR_DIMS> src_shape;
  std::array<size_t, NCCL_RESHARD_MAX_TENSOR_DIMS> dst_shape;
  ::ncclDistTensor_t src = NCCL_M2N_DIST_TENSOR_INITIALIZER;
  ::ncclDistTensor_t dst = NCCL_M2N_DIST_TENSOR_INITIALIZER;
  const ncclDataType_t dtype = to_nccl_dtype(buf.scalar_type());
  src.ndims = ndims;
  src.dtype = dtype;
  src.mesh = &src_mesh;
  src.placements = src_placements.data();
  src.localShape = src_shape.data();
  src.dataPtr = is_src_role ? buf.mutable_data_ptr() : nullptr;
  for (int d = 0; d < ndims; ++d) {
    src.localShape[d] = static_cast<size_t>(src_local_shape[d]);
  }
  dst.ndims = ndims;
  dst.dtype = dtype; // src.dtype must equal dst.dtype (same in-place buffer)
  dst.mesh = &dst_mesh;
  dst.placements = dst_placements.data();
  dst.localShape = dst_shape.data();
  dst.dataPtr = is_dst_role ? buf.mutable_data_ptr() : nullptr;
  for (int d = 0; d < ndims; ++d) {
    dst.localShape[d] = static_cast<size_t>(dst_local_shape[d]);
  }

  std::lock_guard<std::mutex> lock(m2n_lifecycle_mutex);
  nccl_m2n_init_locked(std::nullopt);
  m2n_devices.insert(device_index);
  C10D_NCCL_CHECK(
      ::ncclReshardWithWindow(m2n_handle, comm, window, &src, &dst, stream),
      "ncclReshardWithWindow failed in nccl_reshard");
}

void nccl_m2n_init(std::optional<int64_t> max_cta) {
  std::lock_guard<std::mutex> lock(m2n_lifecycle_mutex);
  nccl_m2n_init_locked(max_cta);
}

bool nccl_m2n_is_available() {
  return true;
}

void nccl_m2n_finalize() {
  // Finalize before communicator teardown.
  std::lock_guard<std::mutex> lock(m2n_lifecycle_mutex);
  if (m2n_handle != nullptr) {
    for (const auto device : m2n_devices) {
      c10::cuda::CUDAGuard device_guard(
          c10::Device(c10::DeviceType::CUDA, device));
      C10_CUDA_CHECK(cudaDeviceSynchronize());
    }
    C10D_NCCL_CHECK(
        ::ncclM2nFinalize(m2n_handle),
        "ncclM2nFinalize failed in nccl_m2n_finalize");
    m2n_handle = nullptr;
    m2n_devices.clear();
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

bool nccl_m2n_is_available() {
  return false;
}

void nccl_m2n_finalize() {
}

#endif // NCCL_HAS_RESHARD_API

} // namespace c10d::nccl_extension
