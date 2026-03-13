#include <torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp>

#ifdef NCCL_HAS_SYMMEM_SUPPORT

#include <algorithm>
#include <vector_types.h>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/cuda/utils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory-inl.cuh>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryUtils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryTypes.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/NCCLSymmetricMemory.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_devcomm_manager.hpp>

#include <ATen/ceil_div.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/error.h>
#include <mutex>
#include <c10/util/flat_hash_map.h>
#include <c10/util/hash.h>

#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
#include <c10/cuda/driver_api.h>
#endif

namespace c10d {
namespace symmetric_memory {

/* Start of NCCLAllocation implementation */

static StoreExchange storeExchange = StoreExchange("NCCLAllocation");

struct NCCLAllocation {
  void* ptr;
  size_t buffer_size;
  int device_idx;
  std::mutex mutex;
  // Map of group name to peer alloc info
  ska::flat_hash_map<std::string, c10::intrusive_ptr<NCCLPeerAllocInfo>>
      peer_alloc_infos_;

  NCCLAllocation(void* ptr, size_t buffer_size, int device_idx)
      : ptr(ptr), buffer_size(buffer_size), device_idx(device_idx) {}

  ~NCCLAllocation() {
    // Avoid calling CUDA functions after driver shutting down
    if (is_finalizing()) {
      return;
    }
    c10::cuda::CUDAGuard guard(device_idx);
    ncclResult_t res = ncclMemFree(ptr);
    if (res != ncclSuccess) {
        LOG(WARNING) << "ncclMemFree failed in NCCLAllocation dtor: "
                      << ncclGetErrorString(res);
    }
  }
};

namespace {

struct NCCLSymmMemKey {
  void* ptr;
  std::string group_name;

  bool operator==(const NCCLSymmMemKey& other) const noexcept {
    return ptr == other.ptr && group_name == other.group_name;
  }
};

struct NCCLSymmMemKeyHash {
  size_t operator()(const NCCLSymmMemKey& key) const {
    auto seed = c10::hash_combine(0, std::hash<void*>{}(key.ptr));
    return c10::hash_combine(seed, std::hash<std::string>{}(key.group_name));
  }
};

// Base allocation ptr -> owning NCCL allocation metadata.
using NCCLAllocMap = ska::flat_hash_map<void*, std::unique_ptr<NCCLAllocation>>;
// (Tensor storage/data ptr, group name) -> cached SymmetricMemory handle.
using NCCLSymmMemMap = ska::flat_hash_map<
    NCCLSymmMemKey,
    c10::intrusive_ptr<NCCLSymmetricMemory>,
    NCCLSymmMemKeyHash>;
// Base allocation ptr -> cached `(tensor ptr, group)` keys derived from it.
using NCCLSymmMemKeysByAlloc =
    ska::flat_hash_map<void*, ska::flat_hash_set<NCCLSymmMemKey, NCCLSymmMemKeyHash>>;

bool pointer_in_allocation(void* ptr, const NCCLAllocation& allocation) {
  auto ptr_int = reinterpret_cast<uintptr_t>(ptr);
  auto base_ptr = reinterpret_cast<uintptr_t>(allocation.ptr);
  return ptr_int >= base_ptr && ptr_int < base_ptr + allocation.buffer_size;
}

NCCLAllocMap::iterator find_allocation_covering_linear(
    void* ptr,
    NCCLAllocMap& allocations) {
  return std::find_if(
      allocations.begin(),
      allocations.end(),
      [&](const auto& entry) {
        return pointer_in_allocation(ptr, *entry.second);
      });
}

NCCLAllocMap::iterator find_allocation_covering(
    void* ptr,
    NCCLAllocMap& allocations) {
  auto alloc_it = allocations.find(ptr);
  if (alloc_it != allocations.end()) {
    return alloc_it;
  }
#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
  auto driver_api = c10::cuda::DriverAPI::get();
  CUdeviceptr base_ptr = 0;
  // Recover the CUDA allocation base for interior pointers before falling
  // back to the linear scan below when the direct lookup cannot help.
  auto status = driver_api->cuMemGetAddressRange_(
      &base_ptr,
      nullptr,
      reinterpret_cast<CUdeviceptr>(ptr));
  if (status == CUDA_SUCCESS) {
    alloc_it = allocations.find(reinterpret_cast<void*>(base_ptr));
    if (alloc_it != allocations.end()) {
      return alloc_it;
    }
  }
#else
  // No driver API support here, so fall through to the linear scan below.
#endif
  return find_allocation_covering_linear(ptr, allocations);
}

} // namespace

#ifdef NCCL_HAS_SYMMEM_DEVICE_SUPPORT
static __global__ void build_ptr_dev(
  ncclWindow_t  handle,
  size_t  offset,  // byte offset inside the window
  void**  buffer,  // symmetric memory buffer
  int  world_size)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int peer = tid; peer < world_size; peer += stride) {
      buffer[peer] = ncclGetLsaPointer(handle, offset, peer);
  }
}
#endif

class NCCLPeerAllocInfo : public c10::intrusive_ptr_target {
 public:
  NCCLPeerAllocInfo(
      NCCLAllocation* allocation,
      std::string group_name)
      : buffer_size_(allocation->buffer_size),
        device_idx_(allocation->device_idx),
        group_name_(std::move(group_name))
  {
    c10::cuda::CUDAGuard guard(device_idx_);
    auto group = resolve_process_group(group_name_);
    rank_ = group->getRank();
    world_size_ = group->getSize();
    auto* ncclPg = dynamic_cast<c10d::ProcessGroupNCCL*>(
        group->getBackend(c10::DeviceType::CUDA).get());
    TORCH_CHECK(ncclPg != nullptr, "backend must be a NCCL process group");
    ncclComm_t comm = reinterpret_cast<ncclComm_t>(ncclPg->getCommPtr());

    C10D_NCCL_CHECK(
      ncclCommWindowRegister(comm, allocation->ptr, buffer_size_, &buffer_win_, NCCL_WIN_COLL_SYMMETRIC),
      c10::str(
          "Failed to window register segment with ptr ",
          allocation->ptr,
          ", size ",
          buffer_size_,
          " on rank ",
          rank_));

    void* signal_pad_ptr;
    const size_t signal_pad_size = get_signal_pad_size();
    C10D_NCCL_CHECK(
        ncclMemAlloc(&signal_pad_ptr, signal_pad_size), "ncclMemAlloc failed");
    C10D_NCCL_CHECK(
    ncclCommWindowRegister(comm, signal_pad_ptr, signal_pad_size, &signal_handle_, NCCL_WIN_COLL_SYMMETRIC),
    c10::str(
        "Failed to window register segment with ptr ",
        signal_pad_ptr,
        ", size ",
        signal_pad_size,
        " on rank ",
        rank_));

    // Starting from NCCL 2.28, we can use device communicators and get peer pointers
#ifdef NCCL_HAS_SYMMEM_DEVICE_SUPPORT
    // Create NCCL device communicator if it doesn't exist. Skip if it already exists.
    auto& mr = NCCLDevCommManager::get(c10::Device(c10::DeviceType::CUDA, device_idx_));
    // Each CTA will need a separate barrier. Assume `symm_max_nblocks` as a starting point.
    mr.try_emplace_devcomm(group_name_, comm, /*LSA*/ symm_max_nblocks, /*GIN*/ symm_max_nblocks);

    const size_t arr_size = sizeof(void*) * world_size_;
    buffers_dev_ = reinterpret_cast<void**>(
        c10::cuda::CUDACachingAllocator::raw_alloc(arr_size));
    signal_pads_dev_ = reinterpret_cast<void**>(
        c10::cuda::CUDACachingAllocator::raw_alloc(arr_size));
    buffers_.resize(world_size_);
    signal_pads_.resize(world_size_);

    // Fill out the peer pointer array
#if NCCL_VERSION_CODE < NCCL_VERSION(2, 29, 0)
    // Lack of host-side API to get peer pointers, so we get them inside a
    // kernel and copy the result to host.
    int threads = std::min(128, world_size_);
    auto stream = at::cuda::getCurrentCUDAStream();
    build_ptr_dev<<<1, threads, 0, stream>>>(buffer_win_, 0, buffers_dev_, world_size_);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    build_ptr_dev<<<1, threads, 0, stream>>>(signal_handle_, 0, signal_pads_dev_, world_size_);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    C10_CUDA_CHECK(cudaMemcpy(
      buffers_.data(),  // dst (host)
      buffers_dev_,  // src (device)
      arr_size,
      cudaMemcpyDeviceToHost));
    C10_CUDA_CHECK(cudaMemcpy(
      signal_pads_.data(),  // dst (host)
      signal_pads_dev_,  // src (device)
      arr_size,
      cudaMemcpyDeviceToHost));
#else
  // Starting from NCCL 2.29, we can use host-side APIs to get peer pointers.
  for (int i = 0; i < world_size_; i++) {
    // If peer is not accessible within LSA domain, `ncclGetPeerDevicePointer`
    // returns nullptr.
    C10D_NCCL_CHECK(
      ncclGetPeerDevicePointer(buffer_win_, 0, i, &buffers_[i]),
      "ncclGetPeerDevicePointer failed");
    C10D_NCCL_CHECK(
      ncclGetPeerDevicePointer(signal_handle_, 0, i, &signal_pads_[i]),
      "ncclGetPeerDevicePointer failed");
  }
  // Copy the peer access pointers to device arrays.
  C10_CUDA_CHECK(cudaMemcpy(
    buffers_dev_,  // dst (device)
    buffers_.data(),  // src (host)
    arr_size,
    cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
    signal_pads_dev_,  // dst (device)
    signal_pads_.data(),  // src (host)
    arr_size,
    cudaMemcpyHostToDevice));

  // Starting from NCCL 2.29, we can use `ncclGetLsaMultimemDevicePointer`
  // to get multicast address.
  void* mc_addr = nullptr;
  // Skip CHECK on purpose to improve fault tolerance since some machine's
  // Fabric Manager may be in bad NVLink Sharp state.
  if (ncclGetLsaMultimemDevicePointer(buffer_win_, 0, &mc_addr) == ncclSuccess) {
    mc_addr_ = mc_addr;
  }
#endif // NCCL_VERSION_CODE < NCCL_VERSION(2, 29, 0)
#endif // NCCL_HAS_SYMMEM_DEVICE_SUPPORT
  }

  // Exact copy is not needed / supported
  NCCLPeerAllocInfo(const NCCLPeerAllocInfo& other) = delete;
  NCCLPeerAllocInfo& operator=(const NCCLPeerAllocInfo& other) = delete;
  NCCLPeerAllocInfo(NCCLPeerAllocInfo&& other) = default;
  NCCLPeerAllocInfo& operator=(NCCLPeerAllocInfo&& other) = default;
  ~NCCLPeerAllocInfo() = default;

 private:
  size_t buffer_size_;
  int device_idx_;
  int rank_;
  int world_size_;
  std::vector<void*> buffers_;
  std::vector<void*> signal_pads_;
  void** buffers_dev_;
  void** signal_pads_dev_;
  std::string group_name_;
  ncclWindow_t buffer_win_;
  ncclWindow_t signal_handle_;
  // Multicast address
  void* mc_addr_{nullptr};

  friend class NCCLSymmetricMemory;
};

NCCLSymmetricMemory::NCCLSymmetricMemory(
    c10::intrusive_ptr<NCCLPeerAllocInfo> pai,
    size_t offset)
    : pai_(std::move(pai)),
      offset_(offset),
      rank_(pai_->rank_),
      world_size_(pai_->world_size_),
      device_idx_(pai_->device_idx_) {
  TORCH_INTERNAL_ASSERT(offset_ < pai_->buffer_size_, "offset out of range");
}

std::vector<void*> NCCLSymmetricMemory::get_buffer_ptrs() {
  return pai_->buffers_;
}

std::vector<void*> NCCLSymmetricMemory::get_signal_pad_ptrs() {
  return pai_->signal_pads_;
}

void** NCCLSymmetricMemory::get_buffer_ptrs_dev() {
  return pai_->buffers_dev_;
}

void** NCCLSymmetricMemory::get_signal_pad_ptrs_dev() {
  return pai_->signal_pads_dev_;
}

size_t NCCLSymmetricMemory::get_buffer_size() {
  return pai_->buffer_size_;
}

bool NCCLSymmetricMemory::has_multicast_support() {
  return pai_->mc_addr_ != nullptr;
}

void* NCCLSymmetricMemory::get_multicast_ptr() {
  if (!has_multicast_support()) {
    return nullptr;
  }
  return static_cast<char*>(pai_->mc_addr_) + offset_;
}

void NCCLSymmetricMemory::barrier(int channel, size_t timeout_ms) {
  TORCH_CHECK(false, "NYI");
}

void NCCLSymmetricMemory::put_signal(int dst_rank, int channel, size_t timeout_ms) {
#ifdef NCCL_HAS_ONE_SIDED_API
  TORCH_CHECK(channel == 0, "channel must be 0 (sigIdx is reserved for future use)");

  c10::cuda::CUDAGuard guard(device_idx_);
  auto stream = at::cuda::getCurrentCUDAStream();

  auto& manager = NCCLDevCommManager::get(c10::Device(c10::DeviceType::CUDA, device_idx_));
  ncclComm_t comm = manager.get_comm(pai_->group_name_);

  // use ncclSignal for pure signaling without data transfer
  C10D_NCCL_CHECK(
      ncclSignal(
          dst_rank,
          channel,
          0,
          0,
          comm,
          stream),
      c10::str("ncclSignal failed for dst_rank=", dst_rank, ", channel=", channel));
#else
  TORCH_CHECK(false, "NYI");
#endif
}

void NCCLSymmetricMemory::wait_signal(int src_rank, int channel, size_t timeout_ms) {
#ifdef NCCL_HAS_ONE_SIDED_API
  TORCH_CHECK(channel == 0, "channel must be 0 (sigIdx is reserved for future use)");

  c10::cuda::CUDAGuard guard(device_idx_);
  auto stream = at::cuda::getCurrentCUDAStream();

  auto& manager = NCCLDevCommManager::get(c10::Device(c10::DeviceType::CUDA, device_idx_));
  ncclComm_t comm = manager.get_comm(pai_->group_name_);

  // create signal descriptor for waiting - populate all fields
  ncclWaitSignalDesc_t signalDesc;
  signalDesc.opCnt = 1;
  signalDesc.peer = src_rank;
  signalDesc.sigIdx = channel;
  signalDesc.ctx = 0;

  C10D_NCCL_CHECK(
      ncclWaitSignal(
          1,
          &signalDesc,
          comm,
          stream),
      c10::str("ncclWaitSignal failed for src_rank=", src_rank, ", channel=", channel));
#else
  TORCH_CHECK(false, "NYI");
#endif
}

int NCCLSymmetricMemory::get_rank() {
  return rank_;
}

int NCCLSymmetricMemory::get_world_size() {
  return world_size_;
}

c10::Device NCCLSymmetricMemory::get_device() {
  return c10::Device(c10::DeviceType::CUDA, device_idx_);
}

ncclWindow_t NCCLSymmetricMemory::get_window() {
  return pai_->buffer_win_;
}

ncclWindow_t NCCLSymmetricMemory::get_signal_pad_handle() {
  return pai_->signal_handle_;
}

size_t NCCLSymmetricMemory::get_offset() {
  return offset_;
}

std::string NCCLSymmetricMemory::get_group_name() {
  return pai_->group_name_;
}

class NCCLSymmetricMemoryAllocator : public SymmetricMemoryAllocator {
 public:
  void* alloc(
      size_t size,
      int device_idx,
      const std::optional<std::string>& group_name) override {
    TORCH_CHECK(
        group_name == std::nullopt,
        "NCCLSymmetricMemoryAllocator::alloc "
        "must not be called with a group_name");

    c10::cuda::CUDAGuard guard(device_idx);
    // TODO: we might need to use a roundup or mempool for mem allocation.
    void* ptr;
    C10D_NCCL_CHECK(ncclMemAlloc(&ptr, size), "ncclMemAlloc");
    {
      std::lock_guard<std::mutex> lock(mutex_);
      allocations_.emplace(
          ptr, std::make_unique<NCCLAllocation>(ptr, size, device_idx));
    }
    return ptr;
  }

  void free(void* ptr) override {
    std::lock_guard<std::mutex> lock(mutex_);
    auto alloc_it = allocations_.find(ptr);
    if (alloc_it == allocations_.end()) {
      return;
    }
    auto cache_keys_it = symm_mem_keys_by_alloc_.find(ptr);
    if (cache_keys_it != symm_mem_keys_by_alloc_.end()) {
      for (const auto& key : cache_keys_it->second) {
        symm_mems_.erase(key);
      }
      symm_mem_keys_by_alloc_.erase(cache_keys_it);
    }
    allocations_.erase(alloc_it);
  };

  size_t get_alloc_size(void* ptr) override {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = allocations_.find(ptr);
    if (it == allocations_.end()) {
      TORCH_CHECK(
          false, ptr, " is not allocated with NCCLSymmetricMemoryAllocator");
    }
    return it->second->buffer_size;
  };

  c10::intrusive_ptr<SymmetricMemory> rendezvous(
      void* ptr,
      const std::optional<std::string>& group_name) override {
    TORCH_CHECK(group_name.has_value(), "group_name must be provided");
    NCCLAllocation* allocation;
    NCCLSymmMemKey key{ptr, *group_name};
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = symm_mems_.find(key);
      if (it != symm_mems_.end()) {
        return it->second;
      }

      // Find the allocation covering the ptr under the allocator lock.
      // We grab a raw pointer to the NCCLAllocation so we can release the
      // allocator lock before doing expensive per-allocation work.
      auto alloc_it = find_allocation_covering(ptr, allocations_);
      TORCH_CHECK(
          alloc_it != allocations_.end(),
          "Pointer not within any SymmetricMemory allocation, "
          "is the tensor allocated from SymmetricMemory?");
      allocation = alloc_it->second.get();
    }

    // Get or create peer alloc info for the group under the per-allocation
    // lock. This serializes concurrent rendezvous on the same allocation
    // for different groups (e.g., forward vs backward).
    std::lock_guard<std::mutex> alloc_lock(allocation->mutex);
    auto& peer_alloc_infos = allocation->peer_alloc_infos_;
    auto pai_it = peer_alloc_infos.find(*group_name);
    if (pai_it == peer_alloc_infos.end()) {
      // Never rendezvoused with this group before, create a new peer alloc info.
      pai_it = peer_alloc_infos.emplace_hint(
          pai_it,
          *group_name,
          c10::make_intrusive<NCCLPeerAllocInfo>(allocation, *group_name));
    }

    auto& pai = pai_it->second;
    size_t offset =
        reinterpret_cast<uintptr_t>(ptr) -
        reinterpret_cast<uintptr_t>(allocation->ptr);
    auto symm_mem = c10::make_intrusive<NCCLSymmetricMemory>(pai, offset);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = symm_mems_.find(key);
      if (it != symm_mems_.end()) {
        return it->second;
      }
      symm_mems_[key] = symm_mem;
      symm_mem_keys_by_alloc_[allocation->ptr].insert(std::move(key));
    }
    return symm_mem;
  }

  bool has_multicast_support(int device_idx) override {
    return device_has_multicast_support(device_idx);
  }

  c10::DeviceType supported_device_type() override {
    return c10::DeviceType::CUDA;
  }

  std::string name() override {
    return "NCCL";
  }

 private:
  std::mutex mutex_;
  NCCLAllocMap allocations_;
  NCCLSymmMemMap symm_mems_;
  NCCLSymmMemKeysByAlloc symm_mem_keys_by_alloc_;
};

struct RegisterNCCLSymmetricMemoryAllocator {
    RegisterNCCLSymmetricMemoryAllocator() {
    auto allocator = c10::make_intrusive<NCCLSymmetricMemoryAllocator>();
    // Query backend used for CUDA tensor
    if (getSymmMemBackendCUDA() == "NCCL") {
      // Direct set (static registration)
      register_allocator(
          c10::DeviceType::CUDA,
          allocator);
    } else {
      // Register availability in case `set_backend` is called dynamically
      register_availability("NCCL", allocator);
    }
  }
};

static RegisterNCCLSymmetricMemoryAllocator register_allocator_;

} // namespace symmetric_memory
} // namespace c10d
#endif // NCCL_HAS_SYMMEM_SUPPORT
