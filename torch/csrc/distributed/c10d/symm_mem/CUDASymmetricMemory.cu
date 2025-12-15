#include <torch/csrc/distributed/c10d/cuda/utils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory-inl.h>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryUtils.hpp>

#include <ATen/ceil_div.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/PeerToPeerAccess.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
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

// add these definitions so that we can compile with CUDA < 12.3
// borrowed from
// https://github.com/NVIDIA/nccl/blob/3ea7eedf3b9b94f1d9f99f4e55536dfcbd23c1ca/src/include/p2p.h#L20
#if CUDA_VERSION < 12030
#define CU_MEM_HANDLE_TYPE_FABRIC ((CUmemAllocationHandleType)0x8ULL)
#define CU_IPC_HANDLE_SIZE 64
typedef struct CUmemFabricHandle_st {
  unsigned char data[CU_IPC_HANDLE_SIZE];
} CUmemFabricHandle_v1;
typedef CUmemFabricHandle_v1 CUmemFabricHandle;
#endif

namespace c10d {
namespace symmetric_memory {

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
  C10_HIP_CHECK(hipMemUnmap(reinterpret_cast<hipDeviceptr_t>(ptr), block_size));
  C10_HIP_CHECK(hipMemRelease(handle));
#else
  TORCH_CHECK(
      false, "CUDASymmetricMemory requires PYTORCH_C10_DRIVER_API_SUPPORTED");
#endif
}

CUDAPeerAllocInfo::CUDAPeerAllocInfo(
    std::vector<c10::intrusive_ptr<AllocationRef>> alloc_refs,
    std::vector<void*> buffers,
    std::vector<void*> signal_pads,
    HandleType mc_handle,
    void* mc_addr,
    size_t buffer_size,
    int local_device_idx,
    int rank,
    int world_size)
    : alloc_refs_(std::move(alloc_refs)),
      buffers_(std::move(buffers)),
      signal_pads_(std::move(signal_pads)),
      mc_handle_(mc_handle),
      mc_addr_(mc_addr),
      buffer_size_(buffer_size),
      local_device_idx_(local_device_idx),
      rank_(rank),
      world_size_(world_size) {
  const size_t arr_size = sizeof(void*) * world_size_;
  buffers_dev_ = reinterpret_cast<void**>(
      c10::cuda::CUDACachingAllocator::raw_alloc(arr_size));
  signal_pads_dev_ = reinterpret_cast<void**>(
      c10::cuda::CUDACachingAllocator::raw_alloc(arr_size));

  c10::cuda::CUDAGuard guard(local_device_idx);
  AT_CUDA_CHECK(cudaMemcpy(
      buffers_dev_, buffers_.data(), arr_size, cudaMemcpyHostToDevice));
  AT_CUDA_CHECK(cudaMemcpy(
      signal_pads_dev_, signal_pads_.data(), arr_size, cudaMemcpyHostToDevice));
}

/* Start of CUDASymmetricMemory */

// This is mostly a shallow copy that shares the pointer to `CUDAPeerAllocInfo`
// which corresponds to the base Block. The CUDASymmetricMemory handle is
// specified by the offset to the base ptr.
CUDASymmetricMemory::CUDASymmetricMemory(const c10::intrusive_ptr<CUDAPeerAllocInfo>& pai, size_t offset)
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
  return pai_->mc_addr_;
}

size_t CUDASymmetricMemory::get_offset() {
  return offset_;
}

void check_channel(int channel, int world_size) {
  TORCH_CHECK(
      channel >= 0,
      "channel for barrier(), put_signal() and wait_signal() ",
      "must be greater than 0 (got ",
      channel,
      ")");
  const size_t num_channels = c10d::symmetric_memory::get_signal_pad_size() /
      sizeof(uint32_t) * world_size;
  TORCH_CHECK(
      static_cast<size_t>(channel) < num_channels,
      "The maximum supported channel for barrier(), put_signal() and wait_signal() is ",
      num_channels - 1,
      " (got ",
      channel,
      ")");
}

static __global__ void barrier_kernel(
    uint32_t** signal_pads,
    int channel,
    int rank,
    int world_size,
    size_t timeout_ms) {
  if (threadIdx.x < world_size) {
    auto target_rank = threadIdx.x;
    if (target_rank == rank) {
      return;
    }
    auto put_success = try_put_signal<std::memory_order_release>(
        signal_pads[target_rank] + world_size * channel + rank, timeout_ms);
    if (!put_success) {
      printf(
          "[FATAL] CUDASymmetricMemory::barrier: rank %d failed to send signal "
          "to rank %d on channel %d after %lu microseconds\n",
          rank,
          target_rank,
          channel,
          timeout_ms);
      trap();
    }
    auto wait_success = try_wait_signal<std::memory_order_acquire>(
        signal_pads[rank] + world_size * channel + target_rank, timeout_ms);
    if (!wait_success) {
      printf(
          "[FATAL] CUDASymmetricMemory::barrier: rank %d failed to receive signal "
          "from rank %d on channel %d after %lu microseconds\n",
          rank,
          target_rank,
          channel,
          timeout_ms);
      trap();
    }
  }
}

void CUDASymmetricMemory::barrier(int channel, size_t timeout_ms) {
  check_channel(channel, world_size_);
  c10::cuda::CUDAGuard guard(local_device_idx_);
  barrier_kernel<<<
      1,
      at::cuda::warp_size(),
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<uint32_t**>(pai_->signal_pads_dev_),
      channel,
      rank_,
      world_size_,
      timeout_ms);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static __global__ void put_signal_kernel(
    uint32_t** signal_pads,
    int dst_rank,
    int channel,
    int rank,
    int world_size,
    size_t timeout_ms) {
  if (threadIdx.x == 0) {
    bool success = try_put_signal<std::memory_order_release>(
        signal_pads[dst_rank] + world_size * channel + rank, timeout_ms);
    if (!success) {
      printf(
          "[FATAL] CUDASymmetricMemory::put_signal: rank %d failed to send signal "
          "to rank %d on channel %d after %lu microseconds\n",
          rank,
          dst_rank,
          channel,
          timeout_ms);
      trap();
    }
  }
}

void CUDASymmetricMemory::put_signal(
    int dst_rank,
    int channel,
    size_t timeout_ms) {
  check_channel(channel, world_size_);
  c10::cuda::CUDAGuard guard(local_device_idx_);
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

static __global__ void wait_signal_kernel(
    uint32_t** signal_pads,
    int src_rank,
    int channel,
    int rank,
    int world_size,
    size_t timeout_ms) {
  if (threadIdx.x == 0) {
    bool success = try_wait_signal<std::memory_order_acquire>(
        signal_pads[rank] + world_size * channel + src_rank, timeout_ms);
    if (!success) {
      printf(
          "[FATAL] CUDASymmetricMemory::wait_signal rank %d failed to receive signal "
          "from rank %d on channel %d after %lu microseconds\n",
          rank,
          src_rank,
          channel,
          timeout_ms);
#if !defined(USE_ROCM)
      __trap();
#else
      assert(0);
#endif
    }
  }
  __threadfence_system();
}

void CUDASymmetricMemory::wait_signal(
    int src_rank,
    int channel,
    size_t timeout_ms) {
  check_channel(channel, world_size_);
  c10::cuda::CUDAGuard guard(local_device_idx_);
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
    size_t signal_pad_offset,
    const std::optional<std::string>& group_name)
    : alloc_ref(std::move(alloc_ref)),
      device_idx(device_idx),
      block_size(block_size),
      buffer_size(buffer_size),
      signal_pad_offset(signal_pad_offset),
      default_group_name(std::move(group_name)) {}

namespace {
using Expandable_Segments_Handle_Type =
    c10::cuda::CUDACachingAllocator::Expandable_Segments_Handle_Type;
}

void* CUDASymmetricMemoryAllocator::alloc(
    size_t size,
    int device_idx,
    const std::optional<std::string>& group_name) {
  size_t signal_pad_offset = at::round_up(size, 16UL);
  size_t block_size = signal_pad_offset + get_signal_pad_size();
  c10::cuda::CUDAGuard guard(device_idx);
  device_idx = static_cast<int>(guard.current_device().index());
#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  // NOLINTNEXTLINE(bugprone-signed-char-misuse)
  prop.location.id = device_idx;
  bool has_fabric_support = at::cuda::get_fabric_access(device_idx);
  LOG(INFO) << "CUDASymmetricMemoryAllocator::alloc: has_fabric_support " << has_fabric_support;
  if (handle_type_ == Expandable_Segments_Handle_Type::UNSPECIFIED) {
    handle_type_ = has_fabric_support ? Expandable_Segments_Handle_Type::FABRIC_HANDLE : Expandable_Segments_Handle_Type::POSIX_FD;
  }
  if (handle_type_ == Expandable_Segments_Handle_Type::POSIX_FD) {
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  } else {
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
  }

  size_t granularity;
  auto driver_api = c10::cuda::DriverAPI::get();
  C10_CUDA_DRIVER_CHECK(driver_api->cuMemGetAllocationGranularity_(
      &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
  block_size = at::round_up(block_size, granularity);

  HandleType handle;
  C10_CUDA_DRIVER_CHECK(driver_api->cuMemCreate_(&handle, block_size, &prop, 0));

#elif defined(USE_ROCM)
  handle_type_ = Expandable_Segments_Handle_Type::POSIX_FD;
  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  // NOLINTNEXTLINE(bugprone-signed-char-misuse)
  prop.location.id = device_idx;
  prop.requestedHandleType = hipMemHandleTypePosixFileDescriptor;

  size_t granularity;
  C10_HIP_CHECK(hipMemGetAllocationGranularity(
      &granularity, &prop, hipMemAllocationGranularityRecommended));
  block_size = at::round_up(block_size, granularity);

  HandleType handle;
  C10_HIP_CHECK(hipMemCreate(
      reinterpret_cast<hipMemGenericAllocationHandle_t*>(&handle),
      block_size,
      &prop,
      0));

#else
  TORCH_CHECK(
      false, "CUDASymmetricMemory requires PYTORCH_C10_DRIVER_API_SUPPORTED");
#endif
  void* ptr = nullptr;
  map_block(&ptr, handle, block_size, device_idx);

  AT_CUDA_CHECK(cudaMemset(ptr, 0, block_size));

  auto alloc_ref =
      c10::make_intrusive<AllocationRef>(ptr, handle, block_size, device_idx);
  auto block = c10::make_intrusive<Block>(
      std::move(alloc_ref),
      device_idx,
      block_size,
      size,
      signal_pad_offset,
      group_name);
  {
    std::unique_lock lock(mutex_);
    ptr_to_block_.emplace(ptr, std::move(block));
  }
  return ptr;
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
  size_t signal_pad_offset;
  bool has_multicast_support;
  char hostname[HOST_NAME_MAX + 1];
};

void validate_rendezvous_requests(
    const std::vector<RendezvousRequest>& reqs,
    int world_size) {
  TORCH_CHECK(reqs.size() == (size_t)world_size);

  // For NVL72 systems, multiple hosts can be within a single nvlink domain.
  // Multiple blocks will have same device_idx but they are on different hosts.
  // Use (hostname, device_idx) pair to uniquely identify each allocation.
  std::set<std::pair<std::string, int>> device_host_pairs;
  for (auto req : reqs) {
    device_host_pairs.insert(std::make_pair(std::string(req.hostname), req.device_idx));
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
    TORCH_CHECK(reqs[r].signal_pad_offset == reqs[0].signal_pad_offset);
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
    if (ranks_with_multicast_support.size() != 0) {
      LOG(WARNING)
          << "Only a subset of ranks in the group has multicast support: "
          << ranks_with_multicast_support << " (world_size=" << reqs.size()
          << "). Skipping multicast initialization because this is unexpected.";
    }
    return false;
  }
}

template <bool use_fabric_handle>
static void init_multicast_for_block(
    HandleType& mc_handle,
    void*& mc_addr,
    const c10::intrusive_ptr<Block>& block,
    std::conditional_t<!use_fabric_handle, IpcChannel&, int&> ipc_channel,
    const std::vector<int>& pids,
    const c10::intrusive_ptr<c10d::Store>& store,
    int rank,
    int world_size) {
#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED) && \
    defined(CUDART_SUPPORTS_MULTICAST)
  auto driver_api = c10::cuda::DriverAPI::get();
  auto handleType = use_fabric_handle
      ? CU_MEM_HANDLE_TYPE_FABRIC
      : CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  using McHandleType =
      std::conditional_t<use_fabric_handle, CUmemFabricHandle, int>;

  McHandleType invalidator;
  std::memset(&invalidator, UINT8_MAX, sizeof(McHandleType));

  // Phase 1: export handle (rank 0 only)
  McHandleType mc_exported_handle{};
  if (rank == 0) {
    CUmulticastObjectProp mc_prop{};
    mc_prop.numDevices = world_size;
    mc_prop.handleTypes = handleType;
    mc_prop.size = block->block_size;

    // create a multicast object, which acts as a handle that allows multiple
    // devices or processes to access the same memory allocation coherently.
    try {
      C10_CUDA_DRIVER_CHECK(
          driver_api->cuMulticastCreate_(&mc_handle, &mc_prop));
      // using the CUDA Driver API to export a multicast object into a POSIX file
      // descriptor.
      C10_CUDA_DRIVER_CHECK(driver_api->cuMemExportToShareableHandle_(
          &mc_exported_handle, mc_handle, handleType, 0));
    } catch (const std::exception& e) {
      // Allow peers gracefully skip multicast initialization by sending -1
      mc_exported_handle = invalidator;
      LOG(WARNING)
          << "SymmetricMemory: fail to export multicast handle.\n"
          << e.what();
    }
  }

  // Phase 2: Exchange handle
  McHandleType recv_handle;
  if constexpr (!use_fabric_handle) {
    recv_handle = ipc_channel.broadcast_fds(rank, 0, pids, mc_exported_handle);
  } else {
    // TODO implement storeExchange.broadcast
    auto gathered_handles = storeExchange.all_gather(store, rank, world_size, mc_exported_handle);
    recv_handle = std::move(gathered_handles[0]);
  }

  // Check exchange result
  if (memcmp(&recv_handle, &invalidator, sizeof(McHandleType)) == 0) {
    LOG(WARNING) << "Gracefully skipping multicast initialization.";
    return;
  }

  // Flip to true after all CUDA steps finish
  bool success_end = false;

  // Phase 3: Import handle (non-0 ranks only)
  if (rank != 0) {
    if constexpr (!use_fabric_handle) {
      // Convert back to a handle from the broadcasted POSIX file descriptor.
      C10_CUDA_DRIVER_CHECK_GOTO(driver_api->cuMemImportFromShareableHandle_(
          &mc_handle,
          (void*)(uintptr_t)recv_handle,
          CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR), check_all);
    } else {
      C10_CUDA_DRIVER_CHECK_GOTO(driver_api->cuMemImportFromShareableHandle_(
          &mc_handle, (void*)&(recv_handle), CU_MEM_HANDLE_TYPE_FABRIC), check_all);
    }
  }

  // Phase 4: Bind memory
  // All rank adds their physical allocation to the multicast object
  C10_CUDA_DRIVER_CHECK_GOTO(
      driver_api->cuMulticastAddDevice_(mc_handle, block->device_idx), check_all);
  C10_CUDA_DRIVER_CHECK_GOTO(driver_api->cuMulticastBindMem_(
      mc_handle, 0, block->alloc_ref->handle, 0, block->block_size, 0), check_all);

  success_end = true;

check_all:
  // Whether all ranks have succeeded
  bool all_succeed = true;
  auto rank_successes = storeExchange.all_gather(store, rank, world_size, success_end);
  for (int r = 0; r < world_size; ++r) {
    all_succeed &= rank_successes[r];
  }
  // Close the file descriptor before exit
  if constexpr (!use_fabric_handle) {
    close(recv_handle);
  }
  if (!all_succeed) {
    LOG(WARNING) << "Gracefully skipping multicast initialization.";
    return;
  }

  // Phase 5: Map to virtual memory
  map_block(&mc_addr, mc_handle, block->block_size, block->device_idx);
#endif
}

namespace {
template <bool use_fabric_handle>
c10::intrusive_ptr<CUDAPeerAllocInfo> make_peer_alloc_info(
    void* ptr,
    c10::intrusive_ptr<Block> block,
    const GroupInfo& group_info) {
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

  auto store = group_info.store;
  int rank = group_info.rank;
  int world_size = group_info.world_size;

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
  C10_HIP_CHECK(hipMemExportToShareableHandle(
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
      .signal_pad_offset = block->signal_pad_offset,
      .has_multicast_support = device_has_multicast_support(block->device_idx)};

  // Populate hostname field for host identification
  gethostname(local_req.hostname, sizeof(local_req.hostname));
  auto reqs = storeExchange.all_gather(store, rank, world_size, local_req);
  validate_rendezvous_requests(reqs, world_size);

  std::vector<int> pids(world_size);
  for (int r = 0; r < world_size; ++r) {
    pids[r] = reqs[r].pid;
  }

  std::vector<BlockHandleType> imported_handles;
  if constexpr (!use_fabric_handle) {
    imported_handles = ipc_channel.all_gather_fds(rank, pids, block_handle);
  } else {
    imported_handles =
        storeExchange.all_gather(store, rank, world_size, block_handle);
  }

  std::vector<HandleType> handles(world_size);
  std::vector<void*> buffers(world_size, nullptr);
  std::vector<void*> signal_pads(world_size, nullptr);

  for (int r = 0; r < world_size; ++r) {
    if (r == rank) {
      handles[r] = block->alloc_ref->handle;
      buffers[r] = ptr;
      signal_pads[r] = (void*)((uintptr_t)ptr + block->signal_pad_offset);
      continue;
    }
    // This api imports a GPU memory allocation that was previously exported as
    // a file descriptor or fabric handle and it returns a memory handle.
#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
    // note how in one case it's directly imported_handles[r] and in another
    // &(imported_handles[r]) so can't do with just type definitions
    if constexpr (!use_fabric_handle) {
      C10_CUDA_DRIVER_CHECK(driver_api->cuMemImportFromShareableHandle_(
          &handles[r],
          (void*)(uintptr_t)imported_handles[r],
          CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
    } else {
      C10_CUDA_DRIVER_CHECK(driver_api->cuMemImportFromShareableHandle_(
          &handles[r],
          (void*)&(imported_handles[r]),
          CU_MEM_HANDLE_TYPE_FABRIC));
    }
#elif defined(USE_ROCM)
    C10_HIP_CHECK(hipMemImportFromShareableHandle(
        &handles[r],
#if ROCM_VERSION >= 70100
        reinterpret_cast<void*>(static_cast<uintptr_t>(imported_handles[r])),
#else
        (void*)(uintptr_t) & (imported_handles[r]),
#endif
        hipMemHandleTypePosixFileDescriptor));
#else
    TORCH_CHECK(
        false, "CUDASymmetricMemory requires PYTORCH_C10_DRIVER_API_SUPPORTED");
#endif
    map_block(&buffers[r], handles[r], block->block_size, block->device_idx);
    signal_pads[r] = (void*)((uintptr_t)buffers[r] + block->signal_pad_offset);
    if constexpr (!use_fabric_handle) {
      close(imported_handles[r]);
    }
  }
  storeExchange.barrier(store, rank, world_size);
  if constexpr (!use_fabric_handle) {
    close(block_handle);
  }

  HandleType mc_handle{};
  void* mc_addr = nullptr;
  bool group_has_multicast_support = check_group_multicast_support(reqs);
  if (!allow_overlapping_devices() && group_has_multicast_support) {
    init_multicast_for_block<use_fabric_handle>(
        mc_handle, mc_addr, block, ipc_channel, pids, store, rank, world_size);
  }

  std::vector<c10::intrusive_ptr<AllocationRef>> alloc_refs;
  for (int r = 0; r < world_size; ++r) {
    if (r == rank) {
      if (mc_addr != nullptr) {
        alloc_refs.push_back(c10::make_intrusive<AllocationRef>(
            mc_addr, mc_handle, block->block_size, block->device_idx, true));
      }
      // Note that in B200, cuMulticastUnbind can error if the mapped buffers
      // are free'd before the multicast object is free'd. That's why the
      // alloc_ref for the multicast object is added first into the vector,
      // such that ~AllocationRef can release it first. For more context,
      // see: https://github.com/pytorch/pytorch/issues/162429
      alloc_refs.emplace_back(block->alloc_ref);
      continue;
    }
    alloc_refs.push_back(c10::make_intrusive<AllocationRef>(
        buffers[r], handles[r], block->block_size, block->device_idx));
  }

  auto pai = c10::make_intrusive<CUDAPeerAllocInfo>(
      std::move(alloc_refs),
      std::move(buffers),
      std::move(signal_pads),
      mc_handle,
      mc_addr,
      block->buffer_size,
      block->device_idx,
      group_info.rank,
      group_info.world_size);

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
    auto group_info = get_group_info(group_name_);
    TORCH_INTERNAL_ASSERT(
        handle_type_ != Expandable_Segments_Handle_Type::UNSPECIFIED)
    bool use_fabric =
        handle_type_ == Expandable_Segments_Handle_Type::FABRIC_HANDLE;
    // PeerAllocInfo captures this block's rendezvous info
    auto pai = use_fabric ? make_peer_alloc_info<true>(ptr, block, group_info)
                          : make_peer_alloc_info<false>(ptr, block, group_info);
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
c10::intrusive_ptr<Block> CUDASymmetricMemoryAllocator::find_block_covering(void* ptr, size_t& offset) {
  std::shared_lock lock(mutex_);
  // In case of MemPool, tensor.storage().data_ptr() may not match
  // exactly an allocation's base address. Thus we perform the search by
  // testing if the former is within an allocation's range.
  auto alloc_it = std::find_if(ptr_to_block_.begin(), ptr_to_block_.end(),
                             [&](const auto& pair){
                                auto& block = pair.second;
                                auto& allocation = block->alloc_ref;
                                auto ptr_int = reinterpret_cast<uintptr_t>(ptr);
                                auto base_ptr = reinterpret_cast<uintptr_t>(allocation->ptr);
                                // Modify offset so that it is returned
                                offset = ptr_int - base_ptr;
                                return ptr_int >= base_ptr && offset < block->buffer_size; });

  if (alloc_it == ptr_to_block_.end()) {
    return nullptr;
  }

  return alloc_it->second;
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

} // namespace symmetric_memory
} // namespace c10d
