#include <torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp>

#ifdef NCCL_HAS_SYMMEM_SUPPORT

#include <algorithm>
#include <vector_types.h>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/cuda/utils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory-inl.cuh>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryTypes.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryUtils.hpp>
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

namespace c10d {
namespace symmetric_memory {

/* Start of NCCLAllocation implementation */

static StoreExchange storeExchange = StoreExchange("NCCLAllocation");

struct NCCLAllocation {
  // Combined ncclMemAlloc region. Layout (signal pad first):
  //   [0, buffer_offset)                            - signal pad
  //   [buffer_offset, buffer_offset + buffer_size)  - user data buffer
  // buffer_offset equals the signal pad size (already 16-aligned). alloc_base is
  // the ncclMemAlloc base (== signal pad base); alloc() hands back
  // `alloc_base + buffer_offset` (the data buffer).
  void* alloc_base;
  // Size of the user-visible data buffer in bytes, as requested by alloc().
  size_t buffer_size;
  // Byte offset from alloc_base to the start of the user buffer; the signal pad
  // occupies [0, buffer_offset).
  size_t buffer_offset;
  int device_idx;
  std::mutex mutex;
  // Map of group name to peer alloc info
  ska::flat_hash_map<std::string, c10::intrusive_ptr<NCCLPeerAllocInfo>>
      peer_alloc_infos_;

  NCCLAllocation(
      void* alloc_base,
      size_t buffer_size,
      size_t buffer_offset,
      int device_idx)
      : alloc_base(alloc_base),
        buffer_size(buffer_size),
        buffer_offset(buffer_offset),
        device_idx(device_idx) {}

  ~NCCLAllocation() {
    // Avoid calling CUDA functions after driver shutting down
    if (is_finalizing()) {
      return;
    }
    c10::cuda::CUDAGuard guard(device_idx);
    // Single free for the combined buffer + signal pad region.
    ncclResult_t res = ncclMemFree(alloc_base);
    if (res != ncclSuccess) {
        LOG(WARNING) << "ncclMemFree failed in NCCLAllocation dtor: "
                      << ncclGetErrorString(res);
    }
  }
};

namespace {

// Base allocation ptr -> owning NCCL allocation metadata.
using NCCLAllocMap = ska::flat_hash_map<void*, std::unique_ptr<NCCLAllocation>>;
// (Tensor storage/data ptr, group name) -> cached SymmetricMemory handle.
using NCCLSymmMemMap = ska::flat_hash_map<
    SymmMemKey,
    c10::intrusive_ptr<NCCLSymmetricMemory>,
    SymmMemKeyHash>;
// Base allocation ptr -> cached `(tensor ptr, group)` keys derived from it.
using NCCLSymmMemKeysByAlloc =
    ska::flat_hash_map<void*, ska::flat_hash_set<SymmMemKey, SymmMemKeyHash>>;

bool pointer_in_allocation(void* ptr, const NCCLAllocation& allocation) {
  auto ptr_int = reinterpret_cast<uintptr_t>(ptr);
  // The data buffer starts `buffer_offset` bytes into the allocation (past the
  // signal pad); only data-region pointers belong to this allocation.
  auto buffer_ptr = reinterpret_cast<uintptr_t>(allocation.alloc_base) +
      allocation.buffer_offset;
  return ptr_int >= buffer_ptr && ptr_int < buffer_ptr + allocation.buffer_size;
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
  // `ptr` is not an allocation key (a MemPool hands out interior pointers), so
  // scan for the allocation whose [buffer, buffer + size) range covers it. We
  // deliberately do not reconstruct the key from the process-global pad size:
  // get_signal_pad_size() may have changed via set_signal_pad_size() since
  // this allocation was created, whereas the scan uses each allocation's own
  // stored buffer_offset.
  // TODO: this linear std::find_if is O(n) in the number of live allocations.
  // Make it O(log n) by switching NCCLAllocMap to an ordered map and using
  // upper_bound to find the covering allocation.
  return find_allocation_covering_linear(ptr, allocations);
}

} // namespace

// Before NCCL 2.29, we can use device-side APIs to get peer pointers.
#if NCCL_VERSION_CODE < NCCL_VERSION(2, 29, 0)
#ifdef NCCL_HAS_SYMMEM_DEVICE_SUPPORT
// Fill both peer pointer arrays in a single kernel launch. For each peer,
// NCCL returns the window base (== signal pad base); the data buffer pointer
// is derived as `base + buffer_offset`, mirroring the host-side layout.
static __global__ void build_ptr_dev(
  ncclWindow_t  handle,
  size_t  buffer_offset,  // data buffer offset; signal pad occupies [0, buffer_offset)
  void**  buffers,        // out: peer buffer pointers
  void**  signal_pads,    // out: peer signal pad pointers
  int  world_size)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int peer = tid; peer < world_size; peer += stride) {
      void* buf = ncclGetLsaPointer(handle, 0, peer);
      signal_pads[peer] = buf;
      buffers[peer] = buf == nullptr
          ? nullptr
          : static_cast<char*>(buf) + buffer_offset;
  }
}
#endif // NCCL_HAS_SYMMEM_DEVICE_SUPPORT
#endif // NCCL_VERSION_CODE < NCCL_VERSION(2, 29, 0)

class NCCLPeerAllocInfo : public c10::intrusive_ptr_target {
 public:
  NCCLPeerAllocInfo(
      NCCLAllocation* allocation,
      std::string group_name)
      : buffer_size_(allocation->buffer_size),
        buffer_offset_(allocation->buffer_offset),
        device_idx_(allocation->device_idx),
        group_name_(std::move(group_name))
  {
    c10::cuda::CUDAGuard guard(device_idx_);
    auto group = resolve_process_group(group_name_);
    rank_ = group->getRank();
    world_size_ = group->getSize();
    // Look up the host ncclComm by group name in NCCLDevCommManager. Any
    // backend that owns a NCCL-compatible communicator (ProcessGroupNCCL, or
    // an external library exposing its ncclComm — torchcomms is one such
    // example) publishes into this registry at comm-init time, so symm_mem
    // doesn't need to know which backend the PG is wrapping.
    auto& mgr = NCCLDevCommManager::get(
        c10::Device(c10::DeviceType::CUDA, device_idx_));
    comm_ = mgr.get_comm(group_name_);
    TORCH_CHECK(
        comm_ != nullptr,
        "NCCL symmetric memory: NCCLDevCommManager returned a null comm for "
        "group '",
        group_name_,
        "'. If you are using ProcessGroups, please make sure its backend has "
        "been eagerly initialized by filling `device_id` in the "
        "`init_process_group` call.");

    // Register a single window over the combined signal pad + buffer region.
    // Layout inside the registration (signal pad first):
    //   [0, signal_pad_size)                          - signal pad
    //   [buffer_offset_, buffer_offset_ + buffer)     - user data buffer
    // The single registration sidesteps NCCL's window-alignment requirement
    // for the data sub-region: only the base pointer (returned by
    // ncclMemAlloc, already granularity-aligned) is registered.
    const size_t aligned_buffer_size = at::round_up(buffer_size_, 16UL);
    const size_t total_size = buffer_offset_ + aligned_buffer_size;
    C10D_NCCL_CHECK(
      ncclCommWindowRegister(comm_, allocation->alloc_base, total_size, &combined_win_, NCCL_WIN_COLL_SYMMETRIC),
      c10::str(
          "Failed to window register segment with ptr ",
          allocation->alloc_base,
          ", size ",
          total_size,
          " on rank ",
          rank_));

#ifdef NCCL_HAS_SYMMEM_DEVICE_SUPPORT
    // (Host comm is already published into NCCLDevCommManager by the
    // owning backend at comm-init time. The earlier mgr.get_comm() call
    // above relied on that. No re-register here.)

    // Starting from NCCL 2.28, we can get peer pointers.
    const size_t arr_size = sizeof(void*) * world_size_;
    buffers_dev_ = reinterpret_cast<void**>(
        c10::cuda::CUDACachingAllocator::raw_alloc(arr_size));
    signal_pads_dev_ = reinterpret_cast<void**>(
        c10::cuda::CUDACachingAllocator::raw_alloc(arr_size));
    buffers_.resize(world_size_);
    signal_pads_.resize(world_size_);

#if NCCL_VERSION_CODE < NCCL_VERSION(2, 29, 0)
    // Lack of host-side API to get peer pointers, so a kernel writes both
    // peer arrays at once and copies the results to host.
    int threads = std::min(128, world_size_);
    auto stream = at::cuda::getCurrentCUDAStream();
    build_ptr_dev<<<1, threads, 0, stream>>>(
        combined_win_, buffer_offset_, buffers_dev_, signal_pads_dev_, world_size_);
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
  // ncclGetPeerDevicePointer returns each peer's window base, which is the
  // signal pad base (the signal pad is at the front of the window).
  for (int i = 0; i < world_size_; i++) {
    // If peer is not accessible within LSA domain, `ncclGetPeerDevicePointer`
    // returns nullptr.
    C10D_NCCL_CHECK(
      ncclGetPeerDevicePointer(combined_win_, 0, i, &signal_pads_[i]),
      "ncclGetPeerDevicePointer failed");
  }
  // Derive each peer's data buffer pointer from its window base; all ranks
  // share the same buffer_offset_ so we don't need to ask NCCL separately.
  for (int i = 0; i < world_size_; i++) {
    buffers_[i] = signal_pads_[i] == nullptr
        ? nullptr
        : static_cast<char*>(signal_pads_[i]) + buffer_offset_;
  }
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
  // Pass buffer_offset_ as the window offset so the returned multicast pointer
  // already points at the data buffer (past the signal pad); no manual add.
  if (ncclGetLsaMultimemDevicePointer(
          combined_win_, buffer_offset_, &mc_addr) == ncclSuccess &&
      mc_addr != nullptr) {
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

  ~NCCLPeerAllocInfo() {
    if (is_finalizing()) {
      return;
    }
    c10::cuda::CUDAGuard guard(device_idx_);
    if (combined_win_ != nullptr) {
      auto res = ncclCommWindowDeregister(comm_, combined_win_);
      if (res != ncclSuccess) {
        LOG(WARNING) << "ncclCommWindowDeregister failed: "
                     << ncclGetErrorString(res);
      }
    }
    if (buffers_dev_ != nullptr) {
      c10::cuda::CUDACachingAllocator::raw_delete(buffers_dev_);
    }
    if (signal_pads_dev_ != nullptr) {
      c10::cuda::CUDACachingAllocator::raw_delete(signal_pads_dev_);
    }
  }

 private:
  size_t buffer_size_;
  // Byte offset from the allocation base to the start of the user buffer; the
  // signal pad occupies [0, buffer_offset_).
  size_t buffer_offset_;
  int device_idx_;
  int rank_;
  int world_size_;
  std::vector<void*> buffers_;
  std::vector<void*> signal_pads_;
  void** buffers_dev_{nullptr};
  void** signal_pads_dev_{nullptr};
  std::string group_name_;
  // Single NCCL window covering both the user data buffer and the signal pad.
  ncclWindow_t combined_win_{nullptr};
  // Multicast address (data buffer base within the multicast mapping)
  void* mc_addr_{nullptr};
  ncclComm_t comm_{nullptr};
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
#ifdef NCCL_HAS_SYMMEM_DEVICE_SUPPORT
  TORCH_CHECK(
      pai_->signal_pads_dev_ != nullptr,
      "NCCLSymmetricMemory::barrier requires peer signal pad pointers, which "
      "are only populated when peers are accessible over the symmetric-memory "
      "(LSA/NVLink) domain.");
  check_channel(channel, world_size_, get_signal_pad_size());
  c10::cuda::CUDAGuard device_guard(device_idx_);
  barrier_kernel<<<
      1,
      std::max(at::cuda::warp_size(), world_size_),
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<uint32_t**>(pai_->signal_pads_dev_),
      channel,
      rank_,
      world_size_,
      timeout_ms);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#else
  TORCH_CHECK(false, "NYI");
#endif
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
  return pai_->combined_win_;
}

size_t NCCLSymmetricMemory::get_offset() {
  return offset_;
}

size_t NCCLSymmetricMemory::get_window_offset() {
  // The NCCL window starts at the signal pad; this handle's data lives
  // buffer_offset_ bytes further in, plus its own offset within the buffer.
  return pai_->buffer_offset_ + offset_;
}

std::string NCCLSymmetricMemory::get_group_name() {
  return pai_->group_name_;
}

class NCCLSymmetricMemoryAllocator : public SymmetricMemoryAllocator {
 public:
  // Allocates a symmetric-memory region laid out as [signal pad | data buffer]:
  // the signal pad occupies [0, buffer_offset) and the user data buffer starts
  // at buffer_offset. Returns the data buffer pointer (alloc_base +
  // buffer_offset), NOT the allocation base -- the signal pad stays hidden in
  // front, and free()/rendezvous() key off this returned data pointer.
  void* alloc(
      size_t size,
      int device_idx,
      const std::optional<std::string>& group_name) override {
    TORCH_CHECK(
        group_name == std::nullopt,
        "NCCLSymmetricMemoryAllocator::alloc "
        "must not be called with a group_name");

    c10::cuda::CUDAGuard guard(device_idx);
    // Allocate signal pad + buffer together in a single ncclMemAlloc call.
    // Layout: signal pad in [0, buffer_offset), data buffer after it.
    // buffer_offset is the signal pad size rounded up to signal_pad_alignment,
    // so the data buffer is aligned; the data size is rounded up as well. A
    // single window is registered over the whole region at rendezvous time, so
    // only the base pointer (already granularity-aligned by ncclMemAlloc) needs
    // to satisfy NCCL's window-alignment requirement.
    const size_t buffer_offset =
        at::round_up(get_signal_pad_size(), signal_pad_alignment);
    const size_t aligned_buffer_size = at::round_up(size, 16UL);
    const size_t total_size = buffer_offset + aligned_buffer_size;
    void* alloc_base;
    C10D_NCCL_CHECK(ncclMemAlloc(&alloc_base, total_size), "ncclMemAlloc");
    // ncclMemAlloc does not zero memory. Zero the signal pad (the first
    // buffer_offset bytes) so the CAS-based barrier() protocol starts from a
    // known all-zero state on first use.
    C10_CUDA_CHECK(cudaMemset(alloc_base, 0, buffer_offset));
    // Hand back the data buffer pointer, not alloc_base; the signal pad stays
    // hidden in front. Returning the data ptr is safe for free(): the whole
    // block is owned by the NCCLAllocation keyed below, which ncclMemFree's
    // alloc_base in its destructor, so free() only needs the data ptr to drop
    // the allocation entry.
    void* buffer_ptr = static_cast<char*>(alloc_base) + buffer_offset;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      // Key by the data pointer we return (that's what `free()` receives).
      allocations_.emplace(
          buffer_ptr,
          std::make_unique<NCCLAllocation>(
              alloc_base, size, buffer_offset, device_idx));
    }
    return buffer_ptr;
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
    // The covering allocation's map key is buffer_ptr (the data buffer base
    // alloc() returned, == alloc_base + buffer_offset); captured here so we
    // don't recompute it below.
    void* buffer_ptr_key = nullptr;
    SymmMemKey key{ptr, *group_name};
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
      buffer_ptr_key = alloc_it->first;
    }

    // Get or create peer alloc info for the group under the per-allocation
    // lock. This serializes concurrent rendezvous on the same allocation
    // for different groups (e.g., forward vs backward).
    std::lock_guard<std::mutex> alloc_lock(allocation->mutex);
    auto& peer_alloc_infos = allocation->peer_alloc_infos_;
    auto& pai = peer_alloc_infos[*group_name];
    if (!pai) {
      pai = c10::make_intrusive<NCCLPeerAllocInfo>(allocation, *group_name);
    }
    // Offset is relative to the data buffer base (past the signal pad).
    size_t offset = reinterpret_cast<uintptr_t>(ptr) -
        reinterpret_cast<uintptr_t>(buffer_ptr_key);
    // Create the SymmetricMemory handle.
    auto symm_mem = c10::make_intrusive<NCCLSymmetricMemory>(pai, offset);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      // Insert the SymmetricMemory handle into the map (cache), keyed by the
      // (Tensor storage ptr, group name) pair.
      auto [it, inserted] = symm_mems_.emplace(key, symm_mem);
      if (!inserted) {
        // This condition should rarely happen, only when another thread happens
        // to be concurrently rendezvousing with the same allocation for the
        // same group.  For safety, we return the existing SymmetricMemory
        // handle and discard the new one.
        return it->second;
      }
      // There is no more use of `key`; we can move it into the per-allocation
      // key set to avoid an extra copy. Key by the data pointer (the value
      // returned by alloc()), matching the lookup done in free().
      symm_mem_keys_by_alloc_[buffer_ptr_key].insert(std::move(key));
    }
    return symm_mem;
  }

  bool has_multicast_support(int device_idx) override {
    return device_has_multicast_support(device_idx);
  }

  bool has_allocation(void* ptr) override {
    std::lock_guard<std::mutex> lock(mutex_);
    return find_allocation_covering(ptr, allocations_) != allocations_.end();
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
