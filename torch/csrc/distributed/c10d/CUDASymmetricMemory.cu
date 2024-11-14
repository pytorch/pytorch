#include <torch/csrc/distributed/c10d/CUDASymmetricMemory.hpp>

#include <torch/csrc/distributed/c10d/CUDASymmetricMemory-inl.h>

#include <ATen/ceil_div.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>

#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
#include <c10/cuda/driver_api.h>
#endif

#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <unistd.h>

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12030
#define CUDART_SUPPORTS_MULTICAST
#endif

namespace {

bool device_has_multicast_support(int device_idx) {
#if defined(CUDART_SUPPORTS_MULTICAST)
  if (c10::utils::check_env("TORCH_SYMM_MEM_DISABLE_MULTICAST") == true) {
    return false;
  }
  // Multicast support requirements:
  // - CUDA Runtime version >= 12030: Checked at compile time using
  // CUDART_VERSION.
  // - Driver version >= 535: Checked at runtime by verifying the existence of
  // cuMulticastCreate_.
  // - Device support: Determined by querying
  // CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED at runtime.
  auto driver_api = c10::cuda::DriverAPI::get();
  int multicast_supported;
  C10_CUDA_DRIVER_CHECK(driver_api->cuDeviceGetAttribute_(
      &multicast_supported,
      CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
      device_idx));
  return driver_api->cuMulticastCreate_ != nullptr && multicast_supported;
#else
  return false;
#endif
}

bool allow_overlapping_devices() {
  return c10::utils::check_env("TORCH_SYMM_MEM_ALLOW_OVERLAPPING_DEVICES") ==
      true;
}

class IpcChannel {
 public:
  IpcChannel() : socket_name_(get_socket_name(getpid())) {
    TORCH_CHECK(
        (socket_ = socket(AF_UNIX, SOCK_DGRAM, 0)) != 0,
        "Failed to create socket: ",
        strerror(errno));

    struct sockaddr_un addr = {.sun_family = AF_UNIX};
    std::copy(socket_name_.begin(), socket_name_.end(), addr.sun_path);

    TORCH_CHECK(
        bind(socket_, (struct sockaddr*)&addr, SUN_LEN(&addr)) == 0,
        "Failed to bind socket: ",
        strerror(errno));
  }

  ~IpcChannel() {
    close(socket_);
    unlink(socket_name_.c_str());
  }

  void send_fd(int dst_pid, int fd) {
    struct sockaddr_un addr = {.sun_family = AF_UNIX};
    auto socket_name = get_socket_name(dst_pid);
    std::copy(socket_name.begin(), socket_name.end(), addr.sun_path);

    struct iovec io = {.iov_base = (void*)("fd"), .iov_len = 2};

    char cbuf[CMSG_SPACE(sizeof(int))];
    memset(cbuf, 0, sizeof(cbuf));

    struct msghdr msg {
      .msg_name = (void*)&addr, .msg_namelen = sizeof(struct sockaddr_un),
      .msg_iov = &io, .msg_iovlen = 1, .msg_control = cbuf,
      .msg_controllen = sizeof(cbuf)
    };

    auto cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;

    if (fd != -1) {
      std::copy(
          reinterpret_cast<const char*>(&fd),
          reinterpret_cast<const char*>(&fd) + sizeof(fd),
          reinterpret_cast<char*>(CMSG_DATA(cmsg)));
    } else {
      msg.msg_controllen = 0;
    }

    TORCH_CHECK(
        sendmsg(socket_, &msg, 0) > 0, "Failed to send fd: ", strerror(errno));
  }

  int recv_fd() {
    char buf[2];
    struct iovec io = {.iov_base = (void*)buf, .iov_len = sizeof(buf)};

    char cbuf[CMSG_SPACE(sizeof(int))];
    memset(cbuf, 0, sizeof(cbuf));

    struct msghdr msg = {
        .msg_iov = &io,
        .msg_iovlen = 1,
        .msg_control = cbuf,
        .msg_controllen = sizeof(cbuf)};

    TORCH_CHECK(
        recvmsg(socket_, &msg, 0) > 0,
        "Failed to receive fd: ",
        strerror(errno));

    if (msg.msg_controllen == 0) {
      return -1;
    }

    auto cmsg = CMSG_FIRSTHDR(&msg);
    TORCH_CHECK(cmsg != NULL);
    TORCH_CHECK(cmsg->cmsg_len == CMSG_LEN(sizeof(int)));
    TORCH_CHECK(
        cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS);
    return *reinterpret_cast<int*>(CMSG_DATA(cmsg));
  }

  std::vector<int> all_gather_fds(
      int rank,
      const std::vector<int>& pids,
      int fd) {
    size_t world_size = pids.size();
    std::vector<int> fds(pids.size());
    fds[rank] = fd;

    int dst_rank = (rank + 1) % world_size;
    for (size_t step = 1; step < world_size; ++step) {
      int src_rank = (rank + world_size - step) % world_size;
      send_fd(pids[dst_rank], fd);
      fd = recv_fd();
      fds[src_rank] = fd;
    }
    return fds;
  }

  int broadcast_fds(
      int rank,
      int src_rank,
      const std::vector<int>& pids,
      int fd) {
    size_t world_size = pids.size();

    if (rank == src_rank) {
      for (int dst_rank = 0; dst_rank < (int)world_size; ++dst_rank) {
        if (dst_rank == rank) {
          continue;
        }
        send_fd(pids[dst_rank], fd);
      }
      return fd;
    }
    return recv_fd();
  }

 private:
  static std::string get_socket_name(int pid) {
    const char* tmp_dir = "/tmp";
    for (const char* env_var : {"TMPDIR", "TMP", "TEMP", "TEMPDIR"}) {
      if (const char* path = getenv(env_var)) {
        tmp_dir = path;
        break;
      }
    }
    std::ostringstream oss;
    oss << tmp_dir << "/symm_mem-" << pid;
    return oss.str();
  }

  std::string socket_name_;
  int socket_;
};

constexpr size_t signal_pad_size = 2048;
const std::string store_comm_prefix = "CUDASymmetricMemory";

static size_t store_comm_seq_id = 0;

template <typename T>
std::vector<T> store_all_gather(
    const c10::intrusive_ptr<c10d::Store>& store,
    int rank,
    int world_size,
    T val) {
  static_assert(std::is_trivially_copyable_v<T>);

  std::vector<std::string> peer_keys;
  for (int r = 0; r < world_size; ++r) {
    std::ostringstream oss;
    oss << store_comm_prefix << "/" << store_comm_seq_id << "/" << r;
    peer_keys.push_back(oss.str());
  }
  ++store_comm_seq_id;

  {
    std::vector<uint8_t> payload(
        reinterpret_cast<uint8_t*>(&val),
        reinterpret_cast<uint8_t*>(&val) + sizeof(T));
    store->set(peer_keys[rank], payload);
  }

  std::vector<T> peer_vals;
  for (int r = 0; r < world_size; ++r) {
    if (r == rank) {
      peer_vals.push_back(val);
      continue;
    }
    store->wait({peer_keys[r]});
    auto payload = store->get(peer_keys[r]);
    TORCH_CHECK(payload.size() == sizeof(T));
    T peer_val{};
    std::memcpy(&peer_val, payload.data(), sizeof(T));
    peer_vals.push_back(peer_val);
  }
  return peer_vals;
}

void store_barrier(
    const c10::intrusive_ptr<c10d::Store>& store,
    int rank,
    int world_size) {
  store_all_gather(store, rank, world_size, 0);
}

void map_block(
    void** ptr,
    c10d::symmetric_memory::HandleType handle,
    size_t size,
    int device_idx) {
#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
  auto driver_api = c10::cuda::DriverAPI::get();
  auto dev_ptr = reinterpret_cast<CUdeviceptr*>(ptr);
  C10_CUDA_DRIVER_CHECK(
      driver_api->cuMemAddressReserve_(dev_ptr, size, 0ULL, 0, 0ULL));
  C10_CUDA_DRIVER_CHECK(driver_api->cuMemMap_(*dev_ptr, size, 0, handle, 0ULL));

  CUmemAccessDesc desc;
  desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  // NOLINTNEXTLINE(bugprone-signed-char-misuse)
  desc.location.id = static_cast<int>(device_idx);
  desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  C10_CUDA_DRIVER_CHECK(driver_api->cuMemSetAccess_(*dev_ptr, size, &desc, 1));
#else
  TORCH_CHECK(
      false, "CUDASymmetricMemory requires PYTORCH_C10_DRIVER_API_SUPPORTED");
#endif
}

} // namespace

namespace c10d {
namespace symmetric_memory {

AllocationRef::AllocationRef(void* ptr, HandleType handle, size_t block_size, int device_idx)
    : ptr(ptr), handle(handle), block_size(block_size), device_idx(device_idx) {}

AllocationRef::~AllocationRef() {
#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
  // Leak the cuda allocations during static deinitialization
  if (is_finalizing()) {
    return;
  }
  auto driver_api = c10::cuda::DriverAPI::get();
  c10::cuda::CUDAGuard guard(device_idx);
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  C10_CUDA_DRIVER_CHECK(
      driver_api->cuMemUnmap_(reinterpret_cast<CUdeviceptr>(ptr), block_size));
  C10_CUDA_DRIVER_CHECK(driver_api->cuMemRelease_(handle));
#else
  TORCH_CHECK(
      false, "CUDASymmetricMemory requires PYTORCH_C10_DRIVER_API_SUPPORTED");
#endif
}

CUDASymmetricMemory::CUDASymmetricMemory(
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

std::vector<void*> CUDASymmetricMemory::get_buffer_ptrs() {
  return buffers_;
}

std::vector<void*> CUDASymmetricMemory::get_signal_pad_ptrs() {
  return signal_pads_;
}

void** CUDASymmetricMemory::get_buffer_ptrs_dev() {
  return buffers_dev_;
}

void** CUDASymmetricMemory::get_signal_pad_ptrs_dev() {
  return signal_pads_dev_;
}

size_t CUDASymmetricMemory::get_buffer_size() {
  return buffer_size_;
}

size_t CUDASymmetricMemory::get_signal_pad_size() {
  return signal_pad_size;
}

bool CUDASymmetricMemory::has_multicast_support() {
  return mc_addr_ != nullptr;
}

void* CUDASymmetricMemory::get_multicast_ptr() {
  return mc_addr_;
}

at::Tensor CUDASymmetricMemory::get_buffer(
    int rank,
    c10::IntArrayRef sizes,
    c10::ScalarType dtype,
    int64_t storage_offset) {
  const size_t numel = std::accumulate(
      sizes.begin(),
      sizes.end(),
      static_cast<size_t>(1),
      std::multiplies<size_t>());
  const auto element_size = c10::elementSize(dtype);
  const auto req_size = (numel + storage_offset) * element_size;
  TORCH_CHECK(
      req_size <= buffer_size_,
      "CUDASymmetricMemory::get_buffer: the requested size (",
      req_size,
      " bytes) exceeds the allocated size (",
      buffer_size_,
      " bytes)");
  auto data_ptr = reinterpret_cast<uint8_t*>(buffers_[rank]) +
      storage_offset * element_size;
  auto device = c10::Device(c10::DeviceType::CUDA, local_device_idx_);
  auto options = at::TensorOptions().dtype(dtype).device(device);
  return at::for_blob(data_ptr, sizes)
      .options(options)
      .target_device(device)
      .make_tensor();
}

at::Tensor CUDASymmetricMemory::get_signal_pad(
    int rank,
    c10::IntArrayRef sizes,
    std::optional<c10::ScalarType> dtype,
    int64_t storage_offset) {
  // If the dtype is unspecified, default it to UInt32, as it
  // is the most common type for signaling purposes.
  if (!dtype.has_value()) {
    dtype = c10::ScalarType::UInt32;
  }

  // If the shape is unspecified, treat the signal pad as a 1d tensor.
  const auto element_size = c10::elementSize(*dtype);
  std::vector<int64_t> shape;
  if (sizes.size() != 0) {
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
      "CUDASymmetricMemory::get_signal_pad: the requested size (",
      req_size,
      " bytes) exceeds the allocated size (",
      signal_pad_size,
      " bytes)");
  auto data_ptr = reinterpret_cast<uint8_t*>(signal_pads_[rank]) +
      storage_offset * element_size;
  auto device = c10::Device(c10::DeviceType::CUDA, local_device_idx_);
  auto options = at::TensorOptions().dtype(*dtype).device(device);
  return at::for_blob(data_ptr, shape)
      .options(options)
      .target_device(device)
      .make_tensor();
}

void check_channel(int channel, int world_size) {
  TORCH_CHECK(
      channel >= 0,
      "channel for barrier(), put_signal() and wait_signal() ",
      "must be greater than 0 (got ",
      channel,
      ")");
  const size_t num_channels = signal_pad_size / sizeof(uint32_t) * world_size;
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
    auto put_success = try_put_signal<MemOpSem::Release>(
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
    auto wait_success = try_wait_signal<MemOpSem::Acquire>(
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
  barrier_kernel<<<1, C10_WARP_SIZE, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<uint32_t**>(signal_pads_dev_),
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
    bool success = try_put_signal<MemOpSem::Release>(
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
  put_signal_kernel<<<1, C10_WARP_SIZE, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<uint32_t**>(signal_pads_dev_),
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
    bool success = try_wait_signal<MemOpSem::Acquire>(
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
  wait_signal_kernel<<<1, C10_WARP_SIZE, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<uint32_t**>(signal_pads_dev_),
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

void* CUDASymmetricMemoryAllocator::alloc(
    size_t size,
    int device_idx,
    const std::optional<std::string>& group_name) {
#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
  c10::cuda::CUDAGuard guard(device_idx);
  device_idx = static_cast<int>(guard.current_device().index());

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  // NOLINTNEXTLINE(bugprone-signed-char-misuse)
  prop.location.id = device_idx;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

  size_t signal_pad_offset = at::round_up(size, 16UL);
  size_t block_size = signal_pad_offset + signal_pad_size;

  size_t granularity;
  auto driver_api = c10::cuda::DriverAPI::get();
  C10_CUDA_DRIVER_CHECK(driver_api->cuMemGetAllocationGranularity_(
      &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
  block_size = at::round_up(block_size, granularity);

  HandleType handle;
  C10_CUDA_DRIVER_CHECK(
      driver_api->cuMemCreate_(&handle, block_size, &prop, 0));

  void* ptr = nullptr;
  map_block(&ptr, handle, block_size, device_idx);

  AT_CUDA_CHECK(cudaMemset(ptr, 0, block_size));

  auto alloc_ref = c10::make_intrusive<AllocationRef>(ptr, handle, block_size, device_idx);
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
#else
  TORCH_CHECK(
      false, "CUDASymmetricMemory requires PYTORCH_C10_DRIVER_API_SUPPORTED");
#endif
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
};

void validate_rendezvous_requests(
    const std::vector<RendezvousRequest>& reqs,
    int world_size) {
  TORCH_CHECK(reqs.size() == (size_t)world_size);

  std::unordered_set<int> device_indices;
  device_indices.reserve(world_size);
  for (auto req : reqs) {
    device_indices.insert(req.device_idx);
  }
  if (!allow_overlapping_devices() &&
      device_indices.size() < (size_t)world_size) {
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

static void init_multicast_for_block(
    HandleType& mc_handle,
    void*& mc_addr,
    const c10::intrusive_ptr<Block>& block,
    IpcChannel& ipc_channel,
    const std::vector<int>& pids,
    const c10::intrusive_ptr<c10d::Store>& store,
    int rank,
    int world_size) {
#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED) && \
    defined(CUDART_SUPPORTS_MULTICAST)
  auto driver_api = c10::cuda::DriverAPI::get();
  if (rank == 0) {
    CUmulticastObjectProp mc_prop{};
    mc_prop.numDevices = world_size;
    mc_prop.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    mc_prop.size = block->block_size;

    auto err = driver_api->cuMulticastCreate_(&mc_handle, &mc_prop);
    if (err != CUDA_SUCCESS) {
      const char* err_str;
      CUresult get_error_str_err = driver_api->cuGetErrorString_(err, &err_str);
      if (get_error_str_err != CUDA_SUCCESS) {
        err_str = "unknown cuda driver error";
      }
      LOG(WARNING)
          << "SymmetricMemory: cuMulticastCreate failed with: \"" << err_str
          << "\". Gracefully skipping multicast initialization. "
          << "However, this is unexpected. Please report the issue on GitHub.";
      // Allow peers gracefully skip multicast initialization by sending -1
      ipc_channel.broadcast_fds(rank, 0, pids, -1);
      return;
    }

    int mc_fd;
    C10_CUDA_DRIVER_CHECK(driver_api->cuMemExportToShareableHandle_(
        &mc_fd, mc_handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
    ipc_channel.broadcast_fds(rank, 0, pids, mc_fd);
    // Ref count is incremented as soon as SCM_RIGHTS send happens
    close(mc_fd);
  } else {
    int mc_fd = ipc_channel.broadcast_fds(rank, 0, pids, -1);
    if (mc_fd == -1) {
      return;
    }
    C10_CUDA_DRIVER_CHECK(driver_api->cuMemImportFromShareableHandle_(
        &mc_handle,
        (void*)(uintptr_t)mc_fd,
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
    close(mc_fd);
  }

  // All rank adds their physical allocation to the multicast object
  C10_CUDA_DRIVER_CHECK(
      driver_api->cuMulticastAddDevice_(mc_handle, block->device_idx));
  C10_CUDA_DRIVER_CHECK(driver_api->cuMulticastBindMem_(
      mc_handle, 0, block->alloc_ref->handle, 0, block->block_size, 0));

  map_block(&mc_addr, mc_handle, block->block_size, block->device_idx);
  store_barrier(store, rank, world_size);
#endif
}

c10::intrusive_ptr<SymmetricMemory> CUDASymmetricMemoryAllocator::rendezvous(
    void* ptr,
    const std::optional<std::string>& group_name) {
#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
  auto block = find_block(ptr);
  if (block == nullptr) {
    return nullptr;
  }

  // The group_name passed to rendezvous() takes precedence over
  // the default group_name specified during allocation.
  std::string group_name_;
  if (group_name.has_value()) {
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

  auto it = block->symm_mems.find(group_name_);
  if (it != block->symm_mems.end()) {
    return it->second;
  }

  IpcChannel ipc_channel;
  auto group_info = get_group_info(group_name_);
  auto store = group_info.store;
  int rank = group_info.rank;
  int world_size = group_info.world_size;

  auto driver_api = c10::cuda::DriverAPI::get();
  int block_fd;
  C10_CUDA_DRIVER_CHECK(driver_api->cuMemExportToShareableHandle_(
      &block_fd,
      block->alloc_ref->handle,
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
      0));

  auto local_req = RendezvousRequest{
      .device_idx = block->device_idx,
      .pid = getpid(),
      .block_size = block->block_size,
      .buffer_size = block->buffer_size,
      .signal_pad_offset = block->signal_pad_offset,
      .has_multicast_support = device_has_multicast_support(block->device_idx)};
  auto reqs = store_all_gather(store, rank, world_size, local_req);
  validate_rendezvous_requests(reqs, world_size);

  std::vector<int> pids(world_size);
  for (int r = 0; r < world_size; ++r) {
    pids[r] = reqs[r].pid;
  }
  auto imported_fds = ipc_channel.all_gather_fds(rank, pids, block_fd);

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
    C10_CUDA_DRIVER_CHECK(driver_api->cuMemImportFromShareableHandle_(
        &handles[r],
        (void*)(uintptr_t)imported_fds[r],
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
    map_block(&buffers[r], handles[r], block->block_size, block->device_idx);
    signal_pads[r] = (void*)((uintptr_t)buffers[r] + block->signal_pad_offset);
    close(imported_fds[r]);
  }
  store_barrier(store, rank, world_size);
  close(block_fd);

  HandleType mc_handle{};
  void* mc_addr = nullptr;
  bool group_has_multicast_support = check_group_multicast_support(reqs);
  if (!allow_overlapping_devices() && group_has_multicast_support) {
    init_multicast_for_block(
        mc_handle, mc_addr, block, ipc_channel, pids, store, rank, world_size);
  }

  std::vector<c10::intrusive_ptr<AllocationRef>> alloc_refs;
  for (int r = 0; r < world_size; ++r) {
    if (r == rank) {
      alloc_refs.emplace_back(block->alloc_ref);
      continue;
    }
    alloc_refs.push_back(c10::make_intrusive<AllocationRef>(
        buffers[r], handles[r], block->block_size, block->device_idx));
  }

  auto symm_mem = c10::make_intrusive<CUDASymmetricMemory>(
      std::move(alloc_refs),
      std::move(buffers),
      std::move(signal_pads),
      mc_handle,
      mc_addr,
      block->buffer_size,
      block->device_idx,
      group_info.rank,
      group_info.world_size);
  block->symm_mems[group_name_] = symm_mem;
  return symm_mem;
#else
  TORCH_CHECK(
      false, "CUDASymmetricMemory requires PYTORCH_C10_DRIVER_API_SUPPORTED");
#endif
}

bool CUDASymmetricMemoryAllocator::has_multicast_support(int device_idx) {
  return device_has_multicast_support(device_idx);
}

c10::intrusive_ptr<Block> CUDASymmetricMemoryAllocator::find_block(void* ptr) {
  std::shared_lock lock(mutex_);
  auto it = ptr_to_block_.find(ptr);
  if (it == ptr_to_block_.end()) {
    return nullptr;
  }
  return it->second;
}

struct RegisterCUDASymmetricMemoryAllocator {
  RegisterCUDASymmetricMemoryAllocator() {
    register_allocator(
        c10::DeviceType::CUDA,
        c10::make_intrusive<CUDASymmetricMemoryAllocator>());
  }
};

static RegisterCUDASymmetricMemoryAllocator register_allocator_;

} // namespace symmetric_memory
} // namespace c10d
