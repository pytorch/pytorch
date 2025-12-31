#include <torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp>

#ifdef NCCL_HAS_SYMMEM_SUPPORT

#include <vector_types.h>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/cuda/utils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory-inl.h>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryUtils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_devcomm_manager.hpp>

#include <ATen/ceil_div.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/error.h>

namespace c10d {
namespace symmetric_memory {

/* Start of NCCLAllocation implementation */

static StoreExchange storeExchange = StoreExchange("NCCLAllocation");

struct NCCLAllocation {
  void* ptr;
  size_t buffer_size;
  int device_idx;

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

class NCCLSymmetricMemory : public SymmetricMemory {
 public:
 NCCLSymmetricMemory(
      NCCLAllocation* allocation,
      const std::string& group_name,
      ncclWindow_t buffer_handle,
      ncclWindow_t signal_handle)
      : buffer_size_(allocation->buffer_size),
        device_idx_(allocation->device_idx),
        group_name_(group_name),
        buffer_handle_(buffer_handle),
        signal_handle_(signal_handle)
  {
    // For logging only
    static int exchanged_n_times = 0;
    c10::cuda::CUDAGuard guard(allocation->device_idx);

    auto global_rank = get_group_info("0").rank;
    GroupInfo& group_info = get_group_info(group_name);
    auto store = group_info.store;
    rank_ = group_info.rank;
    world_size_ = group_info.world_size;  // size of current group
    // Exchange rank to global rank mapping for this group.
    // If it is already available, skip the exchange.
    if (group_info.rank_to_global_rank.empty()) {
      group_info.rank_to_global_rank =
          storeExchange.all_gather(store, rank_, world_size_, global_rank);
      exchanged_n_times++;
      if (rank_ == 0) {
        LOG(INFO) << "[rank " << rank_ << ']'
                  << " rank_to_global_rank: " << group_info.rank_to_global_rank
                  << ", group_name: " << group_name
                  << ", exchanged_n_times: " << exchanged_n_times;
      }
    }

    TORCH_INTERNAL_ASSERT(!group_info.rank_to_global_rank.empty());
    rank_to_global_rank_ = group_info.rank_to_global_rank;

#ifdef NCCL_HAS_SYMMEM_DEVICE_SUPPORT
    const size_t arr_size = sizeof(void*) * world_size_;
    buffers_dev_ = reinterpret_cast<void**>(
        c10::cuda::CUDACachingAllocator::raw_alloc(arr_size));
    signal_pads_dev_ = reinterpret_cast<void**>(
        c10::cuda::CUDACachingAllocator::raw_alloc(arr_size));
    buffers_.resize(world_size_);
    signal_pads_.resize(world_size_);

    int threads = std::min(128, world_size_);
    auto stream = at::cuda::getCurrentCUDAStream();
    build_ptr_dev<<<1, threads, 0, stream>>>(buffer_handle, 0, buffers_dev_, world_size_);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    build_ptr_dev<<<1, threads, 0, stream>>>(signal_handle, 0, signal_pads_dev_, world_size_);
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
#endif
  }


  ~NCCLSymmetricMemory() override = default;

  std::vector<void*> get_buffer_ptrs() override {
    return buffers_;
  }

  std::vector<void*> get_signal_pad_ptrs() override {
    return signal_pads_;
  }

  void** get_buffer_ptrs_dev() override {
    return buffers_dev_;
  }

  void** get_signal_pad_ptrs_dev() override {
    return signal_pads_dev_;
  }

  size_t get_buffer_size() override {
    return buffer_size_;
  }

  bool has_multicast_support() override {
    // TODO
    return false;
  }

  void* get_multicast_ptr() override {
    // TODO
    return nullptr;
  }

  void barrier(int channel, size_t timeout_ms) override {
    // TODO
  }

  void put_signal(int dst_rank, int channel, size_t timeout_ms) override {
    // TODO
  }

  void wait_signal(int src_rank, int channel, size_t timeout_ms) override {
    // TODO
  }

  int get_rank() override {
    return rank_;
  }

  int get_world_size() override {
    return world_size_;
  }

  c10::Device get_device() override {
    return c10::Device(c10::DeviceType::CUDA, device_idx_);
  }

  const std::vector<int>& get_rank_to_global_rank() override {
    return rank_to_global_rank_;
  };

  int* get_rank_to_global_rank_dev() override {
    return nullptr;
  };

  ncclWindow_t get_buffer_handle() {
    return buffer_handle_;
  }

  ncclWindow_t get_signal_pad_handle() {
    return signal_handle_;
  }

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
  ncclWindow_t buffer_handle_;
  ncclWindow_t signal_handle_;
  std::vector<int> rank_to_global_rank_;
};

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

    auto group_info = get_group_info("0");
    auto store = group_info.store;
    c10::cuda::CUDAGuard guard(device_idx);
    // TODO: we might need to use a roundup or mempool for mem allocation.
    void* ptr;
    C10D_NCCL_CHECK(ncclMemAlloc(&ptr, size), "ncclMemAlloc");
    // TODO: thread safety
    allocations_.try_emplace(
        ptr, std::make_unique<NCCLAllocation>(ptr, size, device_idx));
    return ptr;
  }

  void free(void* ptr) override {
    // TODO: thread safety
    allocations_.erase(ptr);
  };

  size_t get_alloc_size(void* ptr) override {
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
    {
      auto it = symm_mems_.find(std::make_tuple(ptr, *group_name));
      if (it != symm_mems_.end()) {
        return it->second;
      }
    }
    auto it = allocations_.find(ptr);
    TORCH_CHECK(it != allocations_.end(), "memory needs to be first allocated before calling rendezvous.");

    auto group = resolve_process_group(group_name.value());
    auto& alloc = it->second;
    c10::cuda::CUDAGuard guard(alloc->device_idx);
    ncclWindow_t handle;
    ncclWindow_t signal_handle;

    auto group_info = get_group_info(group_name.value());
    auto buffer_size_map =
        storeExchange.all_gather(group_info.store, group_info.rank, group_info.world_size, it->second->buffer_size);

    LOG(INFO) << "[rank " << group_info.rank << ']'
              << "buffer_size_map: " << buffer_size_map;
    // NCCL window registration api requires all ranks to have the same buffer size
    // we have this check to make sure all ranks have the same buffer size.
    for (auto r = 0; r < group_info.world_size; ++r) {
      TORCH_CHECK(alloc->buffer_size == buffer_size_map[r], "buffer size mismatch");
    }
    auto* ncclPg = dynamic_cast<c10d::ProcessGroupNCCL*>(
        group->getBackend(c10::DeviceType::CUDA).get());
    TORCH_CHECK(ncclPg != nullptr, "backend must be a NCCL process group");
    ncclComm_t comm = reinterpret_cast<ncclComm_t>(ncclPg->getCommPtr());
    C10D_NCCL_CHECK(
      ncclCommWindowRegister(comm, ptr, alloc->buffer_size, (ncclWindow_t*)&handle, NCCL_WIN_COLL_SYMMETRIC),
      c10::str(
          "Failed to window register segment with ptr ",
          ptr,
          ", size ",
          alloc->buffer_size,
          " on ncclComm_ ",
          comm));

    void* signal_pad_ptr;
    const size_t signal_pad_size = get_signal_pad_size();
    C10D_NCCL_CHECK(
        ncclMemAlloc(&signal_pad_ptr, signal_pad_size), "ncclMemAlloc failed");
    C10D_NCCL_CHECK(
    ncclCommWindowRegister(comm, signal_pad_ptr, signal_pad_size, (ncclWindow_t*)&signal_handle, NCCL_WIN_COLL_SYMMETRIC),
    c10::str(
        "Failed to window register segment with ptr ",
        signal_pad_ptr,
        ", size ",
        signal_pad_size,
        " on ncclComm_ ",
        comm));

    // Create NCCL device communicator if it doesn't exist. Skip if it already exists.
#ifdef NCCL_HAS_SYMMEM_DEVICE_SUPPORT
    auto& mr = NCCLDevCommManager::get(c10::Device(c10::DeviceType::CUDA, alloc->device_idx));
    mr.try_emplace_devcomm(*group_name, comm);
#endif

    auto symm_mem =
        c10::make_intrusive<NCCLSymmetricMemory>(alloc.get(), *group_name, std::move(handle), std::move(signal_handle));

    symm_mems_[std::make_tuple(ptr, *group_name)] = symm_mem;
    return symm_mem;
  };

  bool has_multicast_support(int device_idx) override {
    // TODO
    return false;
  };

  c10::DeviceType supported_device_type() override {
    return c10::DeviceType::CUDA;
  }

  std::string name() override {
    return "NCCL";
  }

 private:
  std::unordered_map<void*, std::unique_ptr<NCCLAllocation>> allocations_;
  std::map<std::tuple<void*, std::string>, c10::intrusive_ptr<SymmetricMemory>>
      symm_mems_;
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
