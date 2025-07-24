#include <torch/csrc/distributed/c10d/cuda/utils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nvshmem_extension.cuh>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory-inl.h>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryUtils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

#include <ATen/ceil_div.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/error.h>

// Starting from NVSHMEM 3.3.9, nvshmem_host.h exists so that we can cleanly
// include only the nvshmem host library headers:
// #include <nvshmem_host.h>
// It translates into the following two lines:
#include <host/nvshmem_api.h>
#include <host/nvshmemx_api.h>
// For maximum compatibility, we use the "host/" style for now.

namespace c10d {
namespace symmetric_memory {

/* Start of CUDASymmetricMemory implementation */

static StoreExchange storeExchange = StoreExchange("NVSHMEMSymmetricMemory");

struct NVSHMEMAllocation {
  void* ptr;
  size_t buffer_size;
  int device_idx;

  NVSHMEMAllocation(void* ptr, size_t buffer_size, int device_idx)
      : ptr(ptr), buffer_size(buffer_size), device_idx(device_idx) {}

  ~NVSHMEMAllocation() {
    // Avoid calling CUDA functions after driver shutting down
    if (is_finalizing()) {
      return;
    }
    c10::cuda::CUDAGuard guard(device_idx);
    nvshmem_free(ptr);  // nvshmem_free has no return value
  }
};

class NVSHMEMSymmetricMemory : public SymmetricMemory {
 public:
  NVSHMEMSymmetricMemory(
      std::shared_ptr<NVSHMEMAllocation> allocation,
      const std::string& group_name)
      : allocation_(allocation),
        buffer_size_(allocation->buffer_size),
        device_idx_(allocation->device_idx),
        group_name_(group_name) {
    // For logging only
    static int exchanged_n_times = 0;
    c10::cuda::CUDAGuard guard(device_idx_);

    auto global_rank = get_group_info("0").rank;
    GroupInfo& group_info = get_group_info(group_name_);
    auto store = group_info.store;
    rank_ = group_info.rank;
    world_size_ = group_info.world_size;
    // Exchange rank to global rank mapping for this group.
    // If it is already available, skip the exchange.
    if (group_info.rank_to_global_rank.empty()) {
      group_info.rank_to_global_rank =
          storeExchange.all_gather(store, rank_, world_size_, global_rank);
      exchanged_n_times++;
      if (rank_ == 0) {
        LOG(INFO) << "[rank " << rank_ << "]"
                  << " rank_to_global_rank: " << group_info.rank_to_global_rank
                  << ", group_name: " << group_name_
                  << ", exchanged_n_times: " << exchanged_n_times;
      }
    }
    TORCH_INTERNAL_ASSERT(!group_info.rank_to_global_rank.empty());
    rank_to_global_rank_ = group_info.rank_to_global_rank;
    for (int r = 0; r < world_size_; ++r) {
      buffers_.push_back(nvshmem_ptr(
          allocation->ptr, rank_to_global_rank_[r]));
    }

    // TODO: use the same allocation for signal pad
    void* signal_pad_ptr = nvshmem_malloc(signal_pad_size);
    AT_CUDA_CHECK(cudaMemset(signal_pad_ptr, 0, signal_pad_size));

    for (int r = 0; r < world_size_; ++r) {
      signal_pads_.push_back(nvshmem_ptr(
          signal_pad_ptr, rank_to_global_rank_[r]));
    }

    const size_t arr_size = sizeof(void*) * world_size_;
    buffers_dev_ = reinterpret_cast<void**>(
        c10::cuda::CUDACachingAllocator::raw_alloc(arr_size));
    signal_pads_dev_ = reinterpret_cast<void**>(
        c10::cuda::CUDACachingAllocator::raw_alloc(arr_size));

    AT_CUDA_CHECK(cudaMemcpy(
        buffers_dev_, buffers_.data(), arr_size, cudaMemcpyHostToDevice));
    AT_CUDA_CHECK(cudaMemcpy(
        signal_pads_dev_,
        signal_pads_.data(),
        arr_size,
        cudaMemcpyHostToDevice));

    rank_to_global_rank_dev_ = reinterpret_cast<int*>(
        c10::cuda::CUDACachingAllocator::raw_alloc(sizeof(int) * world_size_));
    AT_CUDA_CHECK(cudaMemcpy(
        rank_to_global_rank_dev_,
        rank_to_global_rank_.data(),
        sizeof(int) * world_size_,
        cudaMemcpyHostToDevice));
  }

  ~NVSHMEMSymmetricMemory() override{
      // TODO
  };

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

  size_t get_signal_pad_size() override {
    return signal_pad_size;
  };

  bool has_multicast_support() override {
    // TODO
    return false;
  }

  void* get_multicast_ptr() override {
    // TODO
    return nullptr;
  }

  at::Tensor get_buffer(
      int rank,
      c10::IntArrayRef sizes,
      c10::ScalarType dtype,
      int64_t storage_offset) {
    // TODO: deduplicate
    const size_t numel = std::accumulate(
        sizes.begin(),
        sizes.end(),
        static_cast<size_t>(1),
        std::multiplies<size_t>());
    const auto element_size = c10::elementSize(dtype);
    const auto req_size = (numel + storage_offset) * element_size;
    TORCH_CHECK(
        req_size <= buffer_size_,
        "NVSHMEMSymmetricMemory::get_buffer: the requested size (",
        req_size,
        " bytes) exceeds the allocated size (",
        buffer_size_,
        " bytes)");
    auto data_ptr = reinterpret_cast<uint8_t*>(buffers_[rank]) +
        storage_offset * element_size;
    auto device = c10::Device(c10::DeviceType::CUDA, device_idx_);
    auto options = at::TensorOptions().dtype(dtype).device(device);
    return at::for_blob(data_ptr, sizes)
        .options(options)
        .target_device(device)
        .make_tensor();
  }

  at::Tensor get_signal_pad(
      int rank,
      c10::IntArrayRef sizes,
      std::optional<c10::ScalarType> dtype,
      int64_t storage_offset) override {
    // TODO: deduplicate
    // If the dtype is unspecified, default it to UInt32, as it
    // is the most common type for signaling purposes.
    if (!dtype.has_value()) {
      dtype = c10::ScalarType::UInt32;
    }

    // If the shape is unspecified, treat the signal pad as a 1d tensor.
    const auto element_size = c10::elementSize(*dtype);
    std::vector<int64_t> shape;
    if (!sizes.empty()) {
      shape = sizes.vec();
    } else {
      shape.push_back(signal_pad_size / element_size);
    }

    const size_t numel = std::accumulate(
        shape.begin(),
        shape.end(),
        static_cast<size_t>(1),
        std::multiplies<size_t>());
    const auto req_size = (numel + storage_offset) * element_size;
    TORCH_CHECK(
        req_size <= signal_pad_size,
        "NVSHMEMSymmetricMemory::get_signal_pad: the requested size (",
        req_size,
        " bytes) exceeds the allocated size (",
        signal_pad_size,
        " bytes)");
    auto data_ptr = reinterpret_cast<uint8_t*>(signal_pads_[rank]) +
        storage_offset * element_size;
    auto device = c10::Device(c10::DeviceType::CUDA, device_idx_);
    auto options = at::TensorOptions().dtype(*dtype).device(device);
    return at::for_blob(data_ptr, shape)
        .options(options)
        .target_device(device)
        .make_tensor();
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

  virtual const std::vector<int>& get_rank_to_global_rank() override {
    return rank_to_global_rank_;
  };

  int* get_rank_to_global_rank_dev() override {
    return rank_to_global_rank_dev_;
  };

 private:
  std::shared_ptr<NVSHMEMAllocation> allocation_;
  size_t buffer_size_;
  std::vector<void*> buffers_;
  std::vector<void*> signal_pads_;
  int device_idx_;
  int rank_;
  int world_size_;
  void** buffers_dev_;
  void** signal_pads_dev_;
  std::string group_name_;

  std::vector<int> rank_to_global_rank_;
  int* rank_to_global_rank_dev_;
};

// Bootstrap based on user's setting for NCCL
// Long term, this may be a bit unclean; short term, it improves UX
void maybe_initialize_env_vars() {
  auto nccl_socket_if_name = c10::utils::get_env("NCCL_SOCKET_IFNAME");
  auto nccl_hca_list = c10::utils::get_env("NCCL_IB_HCA");
  auto nccl_ib_gid_index = c10::utils::get_env("NCCL_IB_GID_INDEX");
  auto nvshmem_socket_if_name =
      c10::utils::get_env("NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME");
  auto nvshmem_hca_list = c10::utils::get_env("NCCL_IB_HCA");
  auto nvshmem_ib_gid_index = c10::utils::get_env("NVSHMEM_IB_GID_INDEX");

  if (!nvshmem_socket_if_name.has_value() && nccl_socket_if_name.has_value()) {
    c10::utils::set_env(
        "NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME", nccl_socket_if_name->c_str());
  }
  if (!nvshmem_hca_list.has_value() && nccl_hca_list.has_value()) {
    c10::utils::set_env("NVSHMEM_ENABLE_NIC_PE_MAPPING", "1");
    c10::utils::set_env("NVSHMEM_HCA_LIST", nccl_hca_list->c_str());
  }
  if (!nvshmem_ib_gid_index.has_value() && nccl_ib_gid_index.has_value()) {
    c10::utils::set_env("NVSHMEM_IB_GID_INDEX", nccl_ib_gid_index->c_str());
  }
}

void initialize_nvshmem_with_store(
    c10::intrusive_ptr<c10d::Store> store,
    int rank,
    int world_size) {
  static bool is_initialized = false;
  if (is_initialized) {
    return;
  }

  maybe_initialize_env_vars();

  nvshmemx_uniqueid_t unique_id;
  NVSHMEM_CHECK(
      nvshmemx_get_uniqueid(&unique_id), "nvshmemx_get_uniqueid failed");

  // Using an existing store_all_gather due to laziness.
  // TODO(yifu): should use broadcast
  auto unique_ids = storeExchange.all_gather(store, rank, world_size, unique_id);

  nvshmemx_init_attr_t attr;
  nvshmemx_set_attr_uniqueid_args(rank, world_size, &unique_ids[0], &attr);

  NVSHMEM_CHECK(
      nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr),
      "nvshmemx_init_attr failed");

  is_initialized = true;

  // Print version
  int major, minor;
  ::nvshmem_info_get_version(&major, &minor);
  LOG(INFO) << "NVSHMEM is available, version: " << major << '.' << minor;
}

class NVSHMEMSymmetricMemoryAllocator : public SymmetricMemoryAllocator {
 public:
  void* alloc(
      size_t size,
      int device_idx,
      const std::optional<std::string>& group_name) override {
    TORCH_CHECK(
        group_name == std::nullopt,
        "NVSHMEMSymmetricMemoryAllocator::alloc "
        "must not be called with a group_name");

    auto group_info = get_group_info("0");
    auto store = group_info.store;
    int rank = group_info.rank;
    int world_size = group_info.world_size;

    initialize_nvshmem_with_store(store, rank, world_size);
    auto ptr = nvshmem_malloc(size);
    auto allocation =
        std::make_shared<NVSHMEMAllocation>(ptr, size, device_idx);
    // TODO: thread safety
    allocations_.try_emplace(ptr, std::move(allocation));
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
          false, ptr, " is not allocated with NVSHMEMSymmetricMemoryAllocator");
    }
    return it->second->buffer_size;
  };

  c10::intrusive_ptr<SymmetricMemory> rendezvous(
      void* ptr,
      const std::optional<std::string>& group_name) override {
    TORCH_CHECK(group_name.has_value());
    {
      auto it = symm_mems_.find(std::make_tuple(ptr, *group_name));
      if (it != symm_mems_.end()) {
        return it->second;
      }
    }
    auto it = allocations_.find(ptr);
    TORCH_CHECK(it != allocations_.end());
    auto symm_mem =
        c10::make_intrusive<NVSHMEMSymmetricMemory>(it->second, *group_name);

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
    return "NVSHMEM";
  }

 private:
  std::unordered_map<void*, std::shared_ptr<NVSHMEMAllocation>> allocations_;
  std::map<std::tuple<void*, std::string>, c10::intrusive_ptr<SymmetricMemory>>
      symm_mems_;
};

struct RegisterNVSHMEMSymmetricMemoryAllocator {
  RegisterNVSHMEMSymmetricMemoryAllocator() {
    auto allocator = c10::make_intrusive<NVSHMEMSymmetricMemoryAllocator>();
    // Query backend used for CUDA tensor
    if (getSymmMemBackendCUDA() == "NVSHMEM") {
      // Direct set (static registration)
      register_allocator(
          c10::DeviceType::CUDA,
          allocator);
    } else {
      // Register availability in case `set_backend` is called dynamically
      register_availability("NVSHMEM", allocator);
    }
  }
};

static RegisterNVSHMEMSymmetricMemoryAllocator register_allocator_;

} // namespace symmetric_memory
} // namespace c10d
