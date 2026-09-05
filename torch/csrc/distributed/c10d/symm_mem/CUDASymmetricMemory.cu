#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/ParamCommsUtils.hpp>
#include <torch/csrc/distributed/c10d/cuda/utils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryUtils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory-inl.cuh>
#include <torch/csrc/distributed/c10d/symm_mem/GroupStreamGuard.hpp>

#include <ATen/ceil_div.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/PeerToPeerAccess.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/env.h>
#include <c10/util/error.h>

#include <sys/socket.h>
#include <unistd.h>

#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
#include <c10/cuda/driver_api.h>
#elif defined(USE_ROCM)
#include <hip/hip_runtime_api.h>
#endif

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12030
#define CUDART_SUPPORTS_MULTICAST
#endif

namespace c10d::symmetric_memory {

/* Start of CUDASymmetricMemory implementation */

// A set of exchange methods with prefix "CUDASymmetricMemory"
static StoreExchange storeExchange = StoreExchange("CUDASymmetricMemory");

AllocationRef::AllocationRef(
    void* ptr,
    HandleType handle,
    size_t block_size,
    int device_idx,
    bool is_multicast)
    : ptr(ptr),
      handle(handle),
      block_size(block_size),
      device_idx(device_idx),
      is_multicast(is_multicast) {}

AllocationRef::~AllocationRef() {
  if (is_finalizing()) {
    return;
  }
  c10::cuda::CUDAGuard guard(device_idx);
  C10_CUDA_CHECK(cudaDeviceSynchronize());
#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
  // Leak the cuda allocations during static deinitialization
  auto driver_api = c10::cuda::DriverAPI::get();
  C10_CUDA_DRIVER_CHECK(
      driver_api->cuMemUnmap_(reinterpret_cast<CUdeviceptr>(ptr), block_size));
#if defined(CUDART_SUPPORTS_MULTICAST)
  if (is_multicast) {
    C10_CUDA_DRIVER_CHECK(
        driver_api->cuMulticastUnbind_(handle, device_idx, 0, block_size));
  }
#endif
  C10_CUDA_DRIVER_CHECK(driver_api->cuMemRelease_(handle));
#elif defined(USE_ROCM)
  C10_CUDA_CHECK(
      hipMemUnmap(reinterpret_cast<hipDeviceptr_t>(ptr), block_size));
  C10_CUDA_CHECK(hipMemRelease(handle));
#else
  TORCH_CHECK(
      false, "CUDASymmetricMemory requires PYTORCH_C10_DRIVER_API_SUPPORTED");
#endif
}

CUDAPeerAllocInfo::CUDAPeerAllocInfo(
    std::vector<c10::intrusive_ptr<AllocationRef>> alloc_refs,
    std::vector<void*> buffers,
    std::vector<void*> signal_pads,
    void* mc_signal_pad_addr,
    HandleType mc_handle,
    void* mc_addr,
    size_t buffer_size,
    int local_device_idx,
    int rank,
    int world_size,
    std::string group_name)
    : alloc_refs_(std::move(alloc_refs)),
      buffers_(std::move(buffers)),
      signal_pads_(std::move(signal_pads)),
      mc_signal_pad_addr_(mc_signal_pad_addr),
      mc_handle_(mc_handle),
      mc_addr_(mc_addr),
      buffer_size_(buffer_size),
      local_device_idx_(local_device_idx),
      rank_(rank),
      world_size_(world_size),
      group_name_(std::move(group_name)) {
  const size_t arr_size = sizeof(void*) * world_size_;
  buffers_dev_ = reinterpret_cast<void**>(
      c10::cuda::CUDACachingAllocator::raw_alloc(arr_size));
  signal_pads_dev_ = reinterpret_cast<void**>(
      c10::cuda::CUDACachingAllocator::raw_alloc(arr_size));

  c10::cuda::CUDAGuard guard(local_device_idx);
  // Upload on the current stream, then sync. The sync is required because
  // callers may launch kernels that dereference these arrays on a stream other
  // than the one the copies were issued on; same-stream consumers alone would
  // be ordered by the async copies.
  auto stream = at::cuda::getCurrentCUDAStream(
      static_cast<c10::DeviceIndex>(local_device_idx));
  AT_CUDA_CHECK(cudaMemcpyAsync(
      buffers_dev_, buffers_.data(), arr_size, cudaMemcpyHostToDevice, stream));
  AT_CUDA_CHECK(cudaMemcpyAsync(
      signal_pads_dev_,
      signal_pads_.data(),
      arr_size,
      cudaMemcpyHostToDevice,
      stream));
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));
}

/* Start of CUDASymmetricMemory */

// This is mostly a shallow copy that shares the pointer to `CUDAPeerAllocInfo`
// which corresponds to the base Block. The CUDASymmetricMemory handle is
// specified by the offset to the base ptr.
CUDASymmetricMemory::CUDASymmetricMemory(
    const c10::intrusive_ptr<CUDAPeerAllocInfo>& pai,
    size_t offset)
    : local_device_idx_(pai->local_device_idx_),
      rank_(pai->rank_),
      world_size_(pai->world_size_),
      pai_(pai),
      offset_(offset) {
  // offset is specific per symm_mem handle
  TORCH_INTERNAL_ASSERT(offset_ < pai_->buffer_size_, "offset out of range");
}

std::vector<void*> CUDASymmetricMemory::get_buffer_ptrs() {
  return pai_->buffers_;
}

std::vector<void*> CUDASymmetricMemory::get_signal_pad_ptrs() {
  return pai_->signal_pads_;
}

void** CUDASymmetricMemory::get_buffer_ptrs_dev() {
  return pai_->buffers_dev_;
}

void** CUDASymmetricMemory::get_signal_pad_ptrs_dev() {
  return pai_->signal_pads_dev_;
}

size_t CUDASymmetricMemory::get_buffer_size() {
  return pai_->buffer_size_;
}

bool CUDASymmetricMemory::has_multicast_support() {
  return pai_->mc_addr_ != nullptr;
}

void* CUDASymmetricMemory::get_multicast_ptr() {
  if (!has_multicast_support()) {
    return nullptr;
  }
  return static_cast<char*>(pai_->mc_addr_) + offset_;
}

size_t CUDASymmetricMemory::get_offset() {
  return offset_;
}

void CUDASymmetricMemory::barrier(int channel, size_t timeout_ms) {
  check_channel(channel, world_size_, get_signal_pad_size());
  auto pg = c10d::resolve_process_group(pai_->group_name_);
  RECORD_PARAM_COMMS(
      static_cast<int64_t>(0),
      std::make_tuple(pg->getGroupName(), pg->getGroupDesc()),
      rank_,
      "symm_mem::barrier",
      0,
      0,
      at::kByte,
      std::vector<int64_t>(),
      std::vector<int64_t>(),
      -1,
      -1,
      world_size_);
  c10::cuda::CUDAGuard device_guard(local_device_idx_);
  GroupStreamGuard stream_guard(pai_->group_name_, pg);
  if (get_multicast_ptr() != nullptr) {
    multimem_barrier_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
        static_cast<uint32_t*>(pai_->signal_pads_[rank_]),
        static_cast<uint32_t*>(pai_->mc_signal_pad_addr_),
        channel,
        rank_,
        world_size_,
        timeout_ms);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    barrier_kernel<<<
        1,
        max(at::cuda::warp_size(), world_size_),
        0,
        at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<uint32_t**>(pai_->signal_pads_dev_),
        channel,
        rank_,
        world_size_,
        timeout_ms);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

void CUDASymmetricMemory::put_signal(
    int dst_rank,
    int channel,
    size_t timeout_ms) {
  check_channel(channel, world_size_, get_signal_pad_size());
  check_rank(dst_rank, world_size_);
  auto pg = c10d::resolve_process_group(pai_->group_name_);
  RECORD_PARAM_COMMS(
      static_cast<int64_t>(0),
      std::make_tuple(pg->getGroupName(), pg->getGroupDesc()),
      rank_,
      "symm_mem::put_signal",
      0,
      0,
      at::kByte,
      std::vector<int64_t>(),
      std::vector<int64_t>(),
      -1,
      -1,
      world_size_);
  c10::cuda::CUDAGuard device_guard(local_device_idx_);
  GroupStreamGuard stream_guard(pai_->group_name_, pg);
  put_signal_kernel<<<
      1,
      at::cuda::warp_size(),
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<uint32_t**>(pai_->signal_pads_dev_),
      dst_rank,
      channel,
      rank_,
      world_size_,
      timeout_ms);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void CUDASymmetricMemory::wait_signal(
    int src_rank,
    int channel,
    size_t timeout_ms) {
  check_channel(channel, world_size_, get_signal_pad_size());
  check_rank(src_rank, world_size_);
  auto pg = c10d::resolve_process_group(pai_->group_name_);
  RECORD_PARAM_COMMS(
      static_cast<int64_t>(0),
      std::make_tuple(pg->getGroupName(), pg->getGroupDesc()),
      rank_,
      "symm_mem::wait_signal",
      0,
      0,
      at::kByte,
      std::vector<int64_t>(),
      std::vector<int64_t>(),
      -1,
      -1,
      world_size_);
  c10::cuda::CUDAGuard device_guard(local_device_idx_);
  GroupStreamGuard stream_guard(pai_->group_name_, pg);
  wait_signal_kernel<<<
      1,
      at::cuda::warp_size(),
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<uint32_t**>(pai_->signal_pads_dev_),
      src_rank,
      channel,
      rank_,
      world_size_,
      timeout_ms);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int CUDASymmetricMemory::get_rank() {
  return rank_;
}

int CUDASymmetricMemory::get_world_size() {
  return world_size_;
}

c10::Device CUDASymmetricMemory::get_device() {
  return c10::Device(c10::DeviceType::CUDA, local_device_idx_);
}

bool CUDASymmetricMemory::world_within_direct_access() {
  return true;
}

/* End of CUDASymmetricMemory */

Block::Block(
    c10::intrusive_ptr<AllocationRef> alloc_ref,
    int device_idx,
    size_t block_size,
    size_t buffer_size,
    size_t buffer_offset,
    const std::optional<std::string>& group_name)
    : alloc_ref(std::move(alloc_ref)),
      device_idx(device_idx),
      block_size(block_size),
      buffer_size(buffer_size),
      buffer_offset(buffer_offset),
      default_group_name(std::move(group_name)) {}

namespace {
using Expandable_Segments_Handle_Type =
    c10::cuda::CUDACachingAllocator::Expandable_Segments_Handle_Type;
}

// Allocates a symmetric-memory region laid out as [signal pad | data buffer]:
// the signal pad occupies [0, buffer_offset) and the user data buffer starts at
// buffer_offset. Returns the data buffer pointer (alloc_base + buffer_offset),
// NOT the allocation base -- the signal pad stays hidden in front, and
// free()/rendezvous() key off this returned data pointer.
void* CUDASymmetricMemoryAllocator::alloc(
    size_t size,
    int device_idx,
    const std::optional<std::string>& group_name) {
  // buffer_offset is the signal pad size rounded up to signal_pad_alignment so
  // the data buffer stays aligned.
  size_t buffer_offset =
      at::round_up(get_signal_pad_size(), signal_pad_alignment);
  size_t block_size = buffer_offset + at::round_up(size, 16UL);
  c10::cuda::CUDAGuard guard(device_idx);
  device_idx = static_cast<int>(guard.current_device().index());
#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  // NOLINTNEXTLINE(bugprone-signed-char-misuse)
  prop.location.id = device_idx;
  bool has_fabric_support = at::cuda::get_fabric_access(device_idx);
  LOG(INFO) << "CUDASymmetricMemoryAllocator::alloc: has_fabric_support "
            << has_fabric_support;
  if (handle_type_ == Expandable_Segments_Handle_Type::UNSPECIFIED) {
    handle_type_ = has_fabric_support
        ? Expandable_Segments_Handle_Type::FABRIC_HANDLE
        : Expandable_Segments_Handle_Type::POSIX_FD;
  }
  if (handle_type_ == Expandable_Segments_Handle_Type::POSIX_FD) {
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  } else {
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
  }

  auto driver_api = c10::cuda::DriverAPI::get();
  int rdma_flag = 0;
  C10_CUDA_DRIVER_CHECK(driver_api->cuDeviceGetAttribute_(
      &rdma_flag,
      CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
      device_idx));
  if (rdma_flag)
    prop.allocFlags.gpuDirectRDMACapable = 1;

  size_t granularity;
  C10_CUDA_DRIVER_CHECK(driver_api->cuMemGetAllocationGranularity_(
      &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
  block_size = at::round_up(block_size, granularity);

  HandleType handle;
  C10_CUDA_DRIVER_CHECK(
      driver_api->cuMemCreate_(&handle, block_size, &prop, 0));

#elif defined(USE_ROCM)
  handle_type_ = Expandable_Segments_Handle_Type::POSIX_FD;
  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  // NOLINTNEXTLINE(bugprone-signed-char-misuse)
  prop.location.id = device_idx;
  prop.requestedHandleType = hipMemHandleTypePosixFileDescriptor;

  size_t granularity;
  C10_CUDA_CHECK(hipMemGetAllocationGranularity(
      &granularity, &prop, hipMemAllocationGranularityRecommended));
  block_size = at::round_up(block_size, granularity);

  HandleType handle;
  C10_CUDA_CHECK(hipMemCreate(
      reinterpret_cast<hipMemGenericAllocationHandle_t*>(&handle),
      block_size,
      &prop,
      0));

#else
  TORCH_CHECK(
      false, "CUDASymmetricMemory requires PYTORCH_C10_DRIVER_API_SUPPORTED");
#endif
  void* alloc_base = nullptr;
  map_block(&alloc_base, handle, block_size, device_idx);

  // Zero the signal pad (at the front, [0, buffer_offset)) to initialize it for
  // the CAS-based barrier() protocol; the data buffer that follows does not
  // need zeroing. Zero on the current stream, then sync so the signal pad is
  // fully zeroed before rendezvous can expose it to peers.
  auto stream =
      at::cuda::getCurrentCUDAStream(static_cast<c10::DeviceIndex>(device_idx));
  AT_CUDA_CHECK(cudaMemsetAsync(alloc_base, 0, buffer_offset, stream));
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));

  // Hand back the data buffer pointer, not alloc_base; the signal pad stays
  // hidden in front. Returning the data ptr (rather than the alloc ptr) is safe
  // for free(): the whole block is owned by the AllocationRef held in the
  // Block, so free() only needs the data ptr to find and drop the Block; the
  // block (and thus alloc_base) is released internally by ~AllocationRef.
  void* buffer_ptr = static_cast<char*>(alloc_base) + buffer_offset;

  auto alloc_ref = c10::make_intrusive<AllocationRef>(
      alloc_base, handle, block_size, device_idx);
  auto block = c10::make_intrusive<Block>(
      std::move(alloc_ref),
      device_idx,
      block_size,
      size,
      buffer_offset,
      group_name);
  {
    std::unique_lock lock(mutex_);
    // Key by the data pointer we return (that's what free()/rendezvous see).
    ptr_to_block_.emplace(buffer_ptr, std::move(block));
  }
  return buffer_ptr;
}

void CUDASymmetricMemoryAllocator::free(void* ptr) {
  std::unique_lock lock(mutex_);
  ptr_to_block_.erase(ptr);
}

size_t CUDASymmetricMemoryAllocator::get_alloc_size(void* ptr) {
  auto block = find_block(ptr);
  TORCH_CHECK(
      block != nullptr,
      "CUDASymmetricMemoryAllocator::get_alloc_size: input must be allocated ",
      "via CUDASymmetricMemoryAllocator::alloc");
  return block->buffer_size;
}

struct RendezvousRequest {
  int device_idx;
  int pid;
  size_t block_size;
  size_t buffer_size;
  size_t buffer_offset;
  bool has_multicast_support;
  int clique_id;
  char hostname[HOST_NAME_MAX + 1];
};

static std::string import_err_msg(
    int rank,
    int peer,
    const std::vector<RendezvousRequest>& reqs) {
  std::ostringstream oss;
  oss << ". Rank " << rank << " (host: " << reqs[rank].hostname
      << ", device: " << reqs[rank].device_idx << ", fabric_info: {"
      << at::cuda::get_nvml_fabric_info(reqs[rank].device_idx)
      << "}) failed to import memory from rank " << peer
      << " (host: " << reqs[peer].hostname
      << ", device: " << reqs[peer].device_idx << ", NCCL_MNNVL_CLIQUE_ID: "
      << c10::utils::get_env("NCCL_MNNVL_CLIQUE_ID").value_or("unset") << ").";
  return std::move(oss).str();
}

void validate_rendezvous_requests(
    const std::vector<RendezvousRequest>& reqs,
    int world_size) {
  TORCH_CHECK(reqs.size() == (size_t)world_size);

  // For NVL72 systems, multiple hosts can be within a single nvlink domain.
  // Multiple blocks will have same device_idx but they are on different hosts.
  // Use (hostname, device_idx) pair to uniquely identify each allocation.
  std::set<std::pair<std::string, int>> device_host_pairs;
  for (auto req : reqs) {
    device_host_pairs.insert(
        std::make_pair(std::string(req.hostname), req.device_idx));
  }
  if (!allow_overlapping_devices() &&
      device_host_pairs.size() < (size_t)world_size) {
    TORCH_CHECK(
        false,
        "CUDASymmetricMemoryAllocator::rendezvous: ",
        "detected allocations from overlapping devices ",
        "from different ranks.");
  }

  for (int r = 1; r < world_size; ++r) {
    TORCH_CHECK(reqs[r].block_size == reqs[0].block_size);
    TORCH_CHECK(reqs[r].buffer_size == reqs[0].buffer_size);
    TORCH_CHECK(reqs[r].buffer_offset == reqs[0].buffer_offset);
  }
}

// All ranks must be in the same NVLink domain (same clique_id). Detect
// mismatches early before the import fails with an opaque CUDA error.
static void validate_nvlink_fabric_support(
    const std::vector<RendezvousRequest>& reqs,
    int world_size) {
  std::unordered_set<int> clique_ids;
  for (const auto& req : reqs) {
    if (req.clique_id >= 0) {
      clique_ids.insert(req.clique_id);
    }
  }
  if (clique_ids.size() > 1) {
    std::ostringstream oss;
    oss << "CUDASymmetricMemory::rendezvous: "
        << "ranks have mismatched NVLink clique_ids. "
        << "All ranks using fabric handles must be in the same NVLink domain. "
        << "Per-rank info: ";
    for (int r = 0; r < world_size; ++r) {
      if (r > 0) {
        oss << ", ";
      }
      oss << "rank " << r << " (host: " << reqs[r].hostname
          << ", device: " << reqs[r].device_idx
          << ", clique_id: " << reqs[r].clique_id << ')';
    }
    TORCH_CHECK(false, std::move(oss).str());
  }
}

static bool check_group_multicast_support(
    const std::vector<RendezvousRequest>& reqs) {
  std::vector<size_t> ranks_with_multicast_support;
  for (size_t r = 0; r < reqs.size(); ++r) {
    if (reqs[r].has_multicast_support) {
      ranks_with_multicast_support.push_back(r);
    }
  }
  if (ranks_with_multicast_support.size() == reqs.size()) {
    return true;
  } else {
    // We don't expect this to happen. But we want to let the user to know if
    // this happens.
    if (!ranks_with_multicast_support.empty()) {
      LOG(WARNING)
          << "Only a subset of ranks in the group has multicast support: "
          << ranks_with_multicast_support << " (world_size=" << reqs.size()
          << "). Skipping multicast initialization because this is unexpected.";
    }
    return false;
  }
}

#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED) && \
    defined(CUDART_SUPPORTS_MULTICAST)
namespace {

// Owns everything multicast setup acquires and releases it in reverse order
// unless commit() hands it to the caller.
template <bool use_fabric_handle>
class MulticastSetup {
 public:
  using McHandleType =
      std::conditional_t<use_fabric_handle, CUmemFabricHandle, int>;
  static constexpr CUmemAllocationHandleType kHandleType = use_fabric_handle
      ? CU_MEM_HANDLE_TYPE_FABRIC
      : CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

  explicit MulticastSetup(c10::intrusive_ptr<Block> block)
      : block_(std::move(block)), driver_api_(c10::cuda::DriverAPI::get()) {}
  ~MulticastSetup() {
    release();
  }
  MulticastSetup(const MulticastSetup&) = delete;
  MulticastSetup& operator=(const MulticastSetup&) = delete;

  // Rank 0 only. A multicast object is a handle that lets multiple devices or
  // processes access the same allocation coherently.
  bool create_and_export(int world_size, McHandleType& exported) {
    CUmulticastObjectProp mc_prop{};
    mc_prop.numDevices = world_size;
    mc_prop.handleTypes = kHandleType;
    mc_prop.size = block_->block_size;

    auto err = driver_api_->cuMulticastCreate_(&created_handle_, &mc_prop);
    if (err != CUDA_SUCCESS) {
      created_handle_ = 0;
      C10_CUDA_DRIVER_CHECK_WARN(
          err,
          "SymmetricMemory[CREATE_EXPORT]: failed to create multicast object");
      return false;
    }
    err = driver_api_->cuMemExportToShareableHandle_(
        &exported, created_handle_, kHandleType, 0);
    if (err != CUDA_SUCCESS) {
      C10_CUDA_DRIVER_CHECK_WARN(
          err,
          "SymmetricMemory[CREATE_EXPORT]: failed to export multicast handle");
      return false;
    }
    return true;
  }

  void adopt_received_handle(const McHandleType& handle) {
    recv_handle_ = handle;
    if constexpr (!use_fabric_handle) {
      owned_fd_ = handle;
    }
  }

  bool import_and_add_device() {
    CUresult err{};
    if constexpr (!use_fabric_handle) {
      // The fd is the handle: widen it to pointer size, then reinterpret it as
      // the void* osHandle the driver expects.
      err = driver_api_->cuMemImportFromShareableHandle_(
          &imported_handle_,
          reinterpret_cast<void*>(static_cast<uintptr_t>(recv_handle_)),
          kHandleType);
    } else {
      err = driver_api_->cuMemImportFromShareableHandle_(
          &imported_handle_, static_cast<void*>(&recv_handle_), kHandleType);
    }
    if (err != CUDA_SUCCESS) {
      imported_handle_ = 0;
      C10_CUDA_DRIVER_CHECK_WARN(
          err,
          "SymmetricMemory[IMPORT_HANDLE]: failed to import multicast handle");
      return false;
    }
    err = driver_api_->cuMulticastAddDevice_(
        imported_handle_, block_->device_idx);
    if (err != CUDA_SUCCESS) {
      C10_CUDA_DRIVER_CHECK_WARN(
          err,
          "SymmetricMemory[IMPORT_HANDLE]: failed to add device to "
          "multicast object");
      return false;
    }
    return true;
  }

  bool bind_and_map() {
    auto err = driver_api_->cuMulticastBindMem_(
        imported_handle_,
        0,
        block_->alloc_ref->handle,
        0,
        block_->block_size,
        0);
    if (err != CUDA_SUCCESS) {
      C10_CUDA_DRIVER_CHECK_WARN(
          err,
          "SymmetricMemory[BIND_AND_MAP]: failed to bind memory to "
          "multicast object");
      return false;
    }
    bound_ = true;
    // map_block throws, and addr_ holds whatever VA it reserved before it did,
    // which release() still has to undo.
    try {
      map_block(
          &addr_, imported_handle_, block_->block_size, block_->device_idx);
    } catch (const std::exception& e) {
      LOG(WARNING) << "SymmetricMemory[BIND_AND_MAP]: failed to map multicast "
                      "handle.\n"
                   << e.what();
      return false;
    }
    return true;
  }

  // Hands the multicast handle and its mapping to the caller. The received
  // handle stays owned here; the destructor closes it either way.
  void commit(HandleType& out_handle, void*& out_addr) {
    out_handle = imported_handle_;
    out_addr = addr_;
    imported_handle_ = 0;
    addr_ = nullptr;
    bound_ = false;
    // Rank 0 may only drop the reference cuMulticastCreate gave it after every
    // rank has completed setup. The exported handle stops resolving to this
    // multicast object as soon as that reference is gone: a late importer
    // silently gets a fresh, empty object instead, and then every rank blocks
    // forever in cuMulticastBindMem waiting for a device that was added to
    // somebody else's object.
    HandleType created = created_handle_;
    created_handle_ = 0;
    if (created != 0) {
      C10_CUDA_DRIVER_CHECK_MSG(
          driver_api_->cuMemRelease_(created),
          ". SymmetricMemory[COMMIT]: failed to release the multicast handle "
          "cuMulticastCreate returned on rank 0.");
    }
  }

 private:
  void release() {
    if (addr_ != nullptr) {
      C10_CUDA_DRIVER_CHECK_WARN(
          driver_api_->cuMemUnmap_(
              reinterpret_cast<CUdeviceptr>(addr_), block_->block_size),
          "SymmetricMemory[CLEANUP]: failed to unmap multicast address");
      addr_ = nullptr;
    }
    if (imported_handle_ != 0) {
      if (bound_) {
        C10_CUDA_DRIVER_CHECK_WARN(
            driver_api_->cuMulticastUnbind_(
                imported_handle_, block_->device_idx, 0, block_->block_size),
            "SymmetricMemory[CLEANUP]: failed to unbind multicast memory");
        bound_ = false;
      }
      C10_CUDA_DRIVER_CHECK_WARN(
          driver_api_->cuMemRelease_(imported_handle_),
          "SymmetricMemory[CLEANUP]: failed to release imported multicast "
          "handle");
      imported_handle_ = 0;
    }
    if (created_handle_ != 0) {
      C10_CUDA_DRIVER_CHECK_WARN(
          driver_api_->cuMemRelease_(created_handle_),
          "SymmetricMemory[CLEANUP]: failed to release created multicast "
          "handle");
      created_handle_ = 0;
    }
    if (owned_fd_ >= 0) {
      close(owned_fd_);
      owned_fd_ = -1;
    }
  }

  c10::intrusive_ptr<Block> block_;
  c10::cuda::DriverAPI* driver_api_;
  HandleType created_handle_ = 0;
  HandleType imported_handle_ = 0;
  void* addr_ = nullptr;
  bool bound_ = false;
  McHandleType recv_handle_{};
  // POSIX path only: the fd this object must close.
  int owned_fd_ = -1;
};

} // namespace
#endif

template <bool use_fabric_handle>
static void init_multicast_for_block(
    HandleType& mc_handle,
    void*& mc_addr,
    const c10::intrusive_ptr<Block>& block,
    std::conditional_t<!use_fabric_handle, IpcChannel&, int&> ipc_channel,
    const std::vector<int>& pids,
    const c10::intrusive_ptr<c10d::ProcessGroup>& group,
    bool use_pg,
    int rank,
    int world_size) {
#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED) && \
    defined(CUDART_SUPPORTS_MULTICAST)
  using Setup = MulticastSetup<use_fabric_handle>;
  using McHandleType = typename Setup::McHandleType;
  auto store = group->getStore();

  McHandleType invalidator;
  std::memset(&invalidator, UINT8_MAX, sizeof(McHandleType));

  // Every rank reaches both rendezvous points below no matter which step it
  // failed at, so a rank never decides on its own to stop: it reports its own
  // outcome and the whole group degrades together.
  auto all_ranks_succeeded = [&](bool local_success) {
    auto flag = static_cast<uint8_t>(local_success);
    auto rank_flags = use_pg
        ? pg_all_gather(group, block->device_idx, flag)
        : storeExchange.all_gather(store, rank, world_size, flag);
    bool all_succeed = true;
    for (int r = 0; r < world_size; ++r) {
      all_succeed &= (rank_flags[r] != 0);
    }
    return all_succeed;
  };

  Setup setup(block);

  // Phase 1: create and export the multicast object (rank 0 only). On failure
  // rank 0 broadcasts the invalidator so that peers skip multicast gracefully.
  McHandleType exported_handle{};
  if (rank == 0 && !setup.create_and_export(world_size, exported_handle)) {
    exported_handle = invalidator;
  }

  // Phase 2: exchange the handle
  McHandleType recv_handle = invalidator;
  if constexpr (!use_fabric_handle) {
    recv_handle = ipc_channel.broadcast_fds(rank, 0, pids, exported_handle);
  } else if (use_pg) {
    recv_handle = pg_broadcast(group, block->device_idx, 0, exported_handle);
  } else {
    // TODO implement storeExchange.broadcast
    auto gathered_handles =
        storeExchange.all_gather(store, rank, world_size, exported_handle);
    recv_handle = std::move(gathered_handles[0]);
  }
  if (memcmp(&recv_handle, &invalidator, sizeof(McHandleType)) == 0) {
    LOG(WARNING) << "SymmetricMemory[EXCHANGE_HANDLE]: gracefully skipping "
                    "multicast initialization, rank 0 could not export a "
                    "handle.";
    return;
  }
  setup.adopt_received_handle(recv_handle);

  // Phase 3: import the handle and join the device team, then agree before
  // anyone binds.
  if (!all_ranks_succeeded(setup.import_and_add_device())) {
    LOG(WARNING) << "SymmetricMemory[IMPORT_HANDLE]: gracefully skipping "
                    "multicast initialization, not every rank imported the "
                    "handle and joined the multicast object.";
    return;
  }

  // Phase 4: bind and map, then publish success only once the mapping exists.
  if (!all_ranks_succeeded(setup.bind_and_map())) {
    LOG(WARNING) << "SymmetricMemory[BIND_AND_MAP]: gracefully skipping "
                    "multicast initialization, not every rank bound and mapped "
                    "the multicast object.";
    return;
  }

  setup.commit(mc_handle, mc_addr);
#endif
}

namespace {
template <bool use_fabric_handle>
c10::intrusive_ptr<CUDAPeerAllocInfo> make_peer_alloc_info(
    c10::intrusive_ptr<Block> block,
    const std::string& group_name) {
#if defined(USE_ROCM)
  using BlockHandleType = int;
#else
  using BlockHandleType =
      std::conditional_t<use_fabric_handle, CUmemFabricHandle, int>;
#endif
  BlockHandleType block_handle;
  c10::cuda::CUDAGuard guard(block->device_idx);
  if constexpr (!use_fabric_handle) {
    LOG(INFO) << "using posix fd to import symmetric memory handles.";
  } else {
    LOG(INFO) << "using fabric handle to import symmetric memory handles.";
  }

  auto group = resolve_process_group(group_name);
  auto rank = group->getRank();
  auto world_size = group->getSize();
  auto store = group->getStore();

  // Currently, IpcChannel is using a file based socket for inter-process
  // communication
  // Note: don't move ipc_channel construction closer to the use
  // there needs to be a barrier between constructor and first use,
  // and this barrier is provided when we are exchanging rendezvous requests
  using IpcChannelType = std::conditional_t<use_fabric_handle, int, IpcChannel>;
  IpcChannelType ipc_channel;

#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
  auto driver_api = c10::cuda::DriverAPI::get();
  // using the CUDA Driver API to export a GPU memory block as a
  // POSIX file descriptor (FD), so it can be shared across processes via IPC.
  C10_CUDA_DRIVER_CHECK(driver_api->cuMemExportToShareableHandle_(
      &block_handle,
      block->alloc_ref->handle,
      use_fabric_handle ? CU_MEM_HANDLE_TYPE_FABRIC
                        : CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
      0));
#elif defined(USE_ROCM)
  C10_CUDA_CHECK(hipMemExportToShareableHandle(
      &block_handle,
      block->alloc_ref->handle,
      hipMemHandleTypePosixFileDescriptor,
      0));
#else
  TORCH_CHECK(
      false, "CUDASymmetricMemory requires PYTORCH_C10_DRIVER_API_SUPPORTED");
#endif

  auto local_req = RendezvousRequest{
      .device_idx = block->device_idx,
      .pid = getpid(),
      .block_size = block->block_size,
      .buffer_size = block->buffer_size,
      .buffer_offset = block->buffer_offset,
      .has_multicast_support = device_has_multicast_support(block->device_idx),
      .clique_id = at::cuda::get_fabric_clique_id(block->device_idx)};

  // Populate hostname field for host identification
  gethostname(local_req.hostname, sizeof(local_req.hostname));
  // At large rank counts, TCPStore gets overloaded during the metadata
  // exchange. When PG rendezvous is enabled, route the metadata exchange
  // through the process group's NCCL allgather instead.
  bool use_pg = group->hasBackendForDeviceType(c10::DeviceType::CUDA) &&
      group->getBackend(c10::DeviceType::CUDA)->getUsePgForSymmMemRendezvous();
  std::vector<RendezvousRequest> reqs = use_pg
      ? pg_all_gather(group, block->device_idx, local_req)
      : storeExchange.all_gather(store, rank, world_size, local_req);
  validate_nvlink_fabric_support(reqs, world_size);
  validate_rendezvous_requests(reqs, world_size);

  std::vector<int> pids(world_size);
  for (int r = 0; r < world_size; ++r) {
    pids[r] = reqs[r].pid;
  }

  std::vector<BlockHandleType> imported_handles;
  if constexpr (!use_fabric_handle) {
    imported_handles = ipc_channel.all_gather_fds(rank, pids, block_handle);
  } else {
    imported_handles = use_pg
        ? pg_all_gather(group, block->device_idx, block_handle)
        : storeExchange.all_gather(store, rank, world_size, block_handle);
  }

  std::vector<HandleType> handles(world_size);
  // signal_pads[r] is peer r's mapped base (the signal pad lives at the base,
  // and it is the address AllocationRef unmaps); buffers[r] is the data buffer
  // pointer (base + buffer_offset).
  std::vector<void*> buffers(world_size, nullptr);
  std::vector<void*> signal_pads(world_size, nullptr);

  for (int r = 0; r < world_size; ++r) {
    if (r == rank) {
      // Derive pointers from the allocation base (not the rendezvous ptr, which
      // may be an interior MemPool pointer): this pai is shared by every handle
      // on the allocation, and per-handle offsets are applied separately.
      handles[r] = block->alloc_ref->handle;
      signal_pads[r] = block->alloc_ref->ptr;
      buffers[r] = static_cast<char*>(signal_pads[r]) + block->buffer_offset;
      continue;
    }
    // This api imports a GPU memory allocation that was previously exported as
    // a file descriptor or fabric handle and it returns a memory handle.
#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
    // note how in one case it's directly imported_handles[r] and in another
    // &(imported_handles[r]) so can't do with just type definitions
    if constexpr (!use_fabric_handle) {
      C10_CUDA_DRIVER_CHECK_MSG(
          driver_api->cuMemImportFromShareableHandle_(
              &handles[r],
              (void*)(uintptr_t)imported_handles[r],
              CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
          import_err_msg(rank, r, reqs));
    } else {
      C10_CUDA_DRIVER_CHECK_MSG(
          driver_api->cuMemImportFromShareableHandle_(
              &handles[r],
              (void*)&(imported_handles[r]),
              CU_MEM_HANDLE_TYPE_FABRIC),
          import_err_msg(rank, r, reqs));
    }
#elif defined(USE_ROCM)
    C10_CUDA_CHECK(hipMemImportFromShareableHandle(
        &handles[r],
#if ROCM_VERSION >= 70100
        reinterpret_cast<void*>(static_cast<uintptr_t>(imported_handles[r])),
#else
        (void*)(uintptr_t)&(imported_handles[r]),
#endif
        hipMemHandleTypePosixFileDescriptor));
#else
    TORCH_CHECK(
        false, "CUDASymmetricMemory requires PYTORCH_C10_DRIVER_API_SUPPORTED");
#endif
    // map_block returns the mapped base (== signal pad base); the data buffer
    // follows at buffer_offset.
    map_block(
        &signal_pads[r], handles[r], block->block_size, block->device_idx);
    buffers[r] = static_cast<char*>(signal_pads[r]) + block->buffer_offset;
    if constexpr (!use_fabric_handle) {
      close(imported_handles[r]);
    }
  }
  if (use_pg) {
    pg_barrier(group, block->device_idx);
  } else {
    storeExchange.barrier(store, rank, world_size);
  }
  if constexpr (!use_fabric_handle) {
    close(block_handle);
  }

  HandleType mc_handle{};
  void* mc_addr = nullptr;
  bool group_has_multicast_support = check_group_multicast_support(reqs);
  if (!allow_overlapping_devices() && group_has_multicast_support) {
    init_multicast_for_block<use_fabric_handle>(
        mc_handle,
        mc_addr,
        block,
        ipc_channel,
        pids,
        group,
        use_pg,
        rank,
        world_size);
  }

  std::vector<c10::intrusive_ptr<AllocationRef>> alloc_refs;
  for (int r = 0; r < world_size; ++r) {
    if (r == rank) {
      if (mc_addr != nullptr) {
        alloc_refs.push_back(
            c10::make_intrusive<AllocationRef>(
                mc_addr,
                mc_handle,
                block->block_size,
                block->device_idx,
                true));
      }
      // Note that in B200, cuMulticastUnbind can error if the mapped buffers
      // are free'd before the multicast object is free'd. That's why the
      // alloc_ref for the multicast object is added first into the vector,
      // such that ~AllocationRef can release it first. For more context,
      // see: https://github.com/pytorch/pytorch/issues/162429
      alloc_refs.emplace_back(block->alloc_ref);
      continue;
    }
    // signal_pads[r] is peer r's mapped base, i.e. the address AllocationRef
    // unmaps.
    alloc_refs.push_back(
        c10::make_intrusive<AllocationRef>(
            signal_pads[r], handles[r], block->block_size, block->device_idx));
  }

  // The multicast mapping mirrors the block layout: the signal pad is at the
  // base and the data buffer lives at buffer_offset within it.
  void* mc_signal_pad_addr = mc_addr;
  void* mc_buffer_addr = mc_addr != nullptr
      ? static_cast<char*>(mc_addr) + block->buffer_offset
      : nullptr;

  auto pai = c10::make_intrusive<CUDAPeerAllocInfo>(
      std::move(alloc_refs),
      std::move(buffers),
      std::move(signal_pads),
      mc_signal_pad_addr,
      mc_handle,
      mc_buffer_addr,
      block->buffer_size,
      block->device_idx,
      rank,
      world_size,
      group_name);

  return pai;
}

} // namespace

c10::intrusive_ptr<SymmetricMemory> CUDASymmetricMemoryAllocator::rendezvous(
    void* ptr,
    const std::optional<std::string>& group_name) {
  // In case of MemPool, the `ptr` passed in (i.e. tensor storage ptr) may not
  // be the same as the allocation base pointer, so we need to find the block
  // that covers the `ptr`
  size_t offset = 0;
  auto block = find_block_covering(ptr, offset);
  if (block == nullptr) {
    TORCH_WARN(
        "Pointer not within any SymmetricMemory allocation, "
        "is the tensor allocated from SymmetricMemory?");
    return nullptr;
  }
  // The group_name passed to rendezvous() takes precedence over
  // the default group_name specified during allocation.
  std::string group_name_;
  // Treat empty string and std::nullopt the same as empty string seems to be
  // implicitly used that way
  if (group_name.has_value() && group_name != "") {
    group_name_ = *group_name;
  } else {
    if (!block->default_group_name.has_value()) {
      TORCH_CHECK(
          false,
          "CUDASymmetricMemory::rendezvous: `group_name` is neither "
          "specified during allocation nor passed to rendezvous().");
    }
    group_name_ = *block->default_group_name;
  }

  // If found, this block has been rendezvous by the given group
  auto it = block->symm_mems.find(group_name_);
  if (it == block->symm_mems.end()) {
    // Create PeerAllocInfo for this block (this is the costly part)
    TORCH_INTERNAL_ASSERT(
        handle_type_ != Expandable_Segments_Handle_Type::UNSPECIFIED)
    bool use_fabric =
        handle_type_ == Expandable_Segments_Handle_Type::FABRIC_HANDLE;
    // PeerAllocInfo captures this block's rendezvous info
    auto pai = use_fabric ? make_peer_alloc_info<true>(block, group_name_)
                          : make_peer_alloc_info<false>(block, group_name_);
    // Cache it with the group name
    it = block->symm_mems.emplace(group_name_, pai).first;
  }

  // Create symm mem handle for this tensor, specified by its offset
  auto pai = it->second;
  return c10::make_intrusive<CUDASymmetricMemory>(pai, offset);
}

bool CUDASymmetricMemoryAllocator::has_multicast_support(int device_idx) {
  return device_has_multicast_support(device_idx);
}

c10::DeviceType CUDASymmetricMemoryAllocator::supported_device_type() {
  return c10::DeviceType::CUDA;
}

std::string CUDASymmetricMemoryAllocator::name() {
  return "CUDA";
}

c10::intrusive_ptr<Block> CUDASymmetricMemoryAllocator::find_block(void* ptr) {
  std::shared_lock lock(mutex_);
  auto it = ptr_to_block_.find(ptr);
  if (it == ptr_to_block_.end()) {
    return nullptr;
  }
  return it->second;
}

/* Search for a block that covers the given ptr, and write back the offset to
 * the base ptr; error out if not found */
c10::intrusive_ptr<Block> CUDASymmetricMemoryAllocator::find_block_covering(
    void* ptr,
    size_t& offset) {
  std::shared_lock lock(mutex_);
  // In case of MemPool, tensor.storage().data_ptr() may not match
  // exactly an allocation's base address. Thus we perform the search by
  // testing if the former is within an allocation's range.
  auto alloc_it = std::find_if(
      ptr_to_block_.begin(), ptr_to_block_.end(), [&](const auto& pair) {
        auto& block = pair.second;
        auto ptr_int = reinterpret_cast<uintptr_t>(ptr);
        // pair.first is buffer_ptr, the key alloc()
        // stored (alloc_base + buffer_offset), i.e. the
        // data buffer start past the signal pad.
        auto buffer_ptr = reinterpret_cast<uintptr_t>(pair.first);
        // Modify offset so that it is returned
        offset = ptr_int - buffer_ptr;
        return ptr_int >= buffer_ptr && offset < block->buffer_size;
      });

  if (alloc_it == ptr_to_block_.end()) {
    return nullptr;
  }

  return alloc_it->second;
}

bool CUDASymmetricMemoryAllocator::has_allocation(void* ptr) {
  return find_block(ptr) != nullptr;
}

struct RegisterCUDASymmetricMemoryAllocator {
  RegisterCUDASymmetricMemoryAllocator() {
    auto allocator = c10::make_intrusive<CUDASymmetricMemoryAllocator>();
    // Query backend used for CUDA tensor
    // "CUDA" backend stands for this implementation
    if (getSymmMemBackendCUDA() == "CUDA") {
      // Direct set (static registration)
      register_allocator(c10::DeviceType::CUDA, allocator);
    } else {
      // Register availability in case `set_backend` is called dynamically
      register_availability("CUDA", allocator);
    }
  }
};

static RegisterCUDASymmetricMemoryAllocator register_allocator_;

} // namespace c10d::symmetric_memory
