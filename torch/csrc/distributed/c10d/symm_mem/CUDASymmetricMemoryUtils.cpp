#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <unistd.h>

#include <c10/util/error.h>

#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
#include <c10/cuda/driver_api.h>
#elif defined(USE_ROCM)
#include <c10/hip/HIPException.h>
#include <hip/hip_runtime_api.h>
#endif

#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/cuda/utils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryUtils.hpp>

namespace c10d::symmetric_memory {

bool device_has_multicast_support(int device_idx) {
  if (c10::utils::check_env("TORCH_SYMM_MEM_DISABLE_MULTICAST") == true) {
    return false;
  }
  return c10d::cuda::deviceSupportsMulticast(device_idx);
}

bool allow_overlapping_devices() {
  return c10::utils::check_env("TORCH_SYMM_MEM_ALLOW_OVERLAPPING_DEVICES") ==
      true;
}

// Query environment variable to get the backend used for CUDA Symmetric Memory.
std::string getSymmMemBackendCUDA() {
  // TORCH_SYMMMEM environment variable can be used to indicate the preferred
  // backend.
  static auto val = c10::utils::get_env("TORCH_SYMMMEM");
  if (val.has_value()) {
    TORCH_CHECK(
        val.value() == "CUDA" || val.value() == "NVSHMEM" ||
            val.value() == "NCCL",
        "TORCH_SYMMMEM environment variable must be one of 'CUDA', 'NVSHMEM', 'NCCL'.")
    return val.value();
  }
  // If TORCH_SYMMMEM is not set, check if NVSHMEM is available (for broader
  // support).
  // TODO: uncomment this once all single-node tests work with NVSHMEM
  // if (is_nvshmem_available()) {
  //   return "NVSHMEM";
  // }
  return "CUDA";
}

IpcChannel::IpcChannel()
    : socket_name_(get_socket_name(getpid())),
      socket_(socket(AF_UNIX, SOCK_DGRAM, 0)) {
  // On success, a file descriptor for the new socket is returned.
  //  On error, -1 is returned, and errno is set to indicate the error.
  TORCH_CHECK(
      socket_ != -1, "Failed to create socket: ", c10::utils::str_error(errno));

  struct sockaddr_un addr = {.sun_family = AF_UNIX};
  std::copy(socket_name_.begin(), socket_name_.end(), addr.sun_path);

  TORCH_CHECK(
      bind(socket_, (struct sockaddr*)&addr, SUN_LEN(&addr)) == 0,
      "Failed to bind socket: ",
      c10::utils::str_error(errno));
}

IpcChannel::~IpcChannel() {
  close(socket_);
  unlink(socket_name_.c_str());
}

void IpcChannel::send_fd(int dst_pid, int fd) {
  // Because file descriptors are process-local kernel objects, and we can’t
  // pass them via normal socket payloads (like write() or send()).  Unix domain
  // sockets provide a mechanism to pass actual FDs via sendmsg()/recvmsg().
  // Define destination socket address
  struct sockaddr_un addr = {.sun_family = AF_UNIX};
  auto socket_name = get_socket_name(dst_pid);
  std::copy(socket_name.begin(), socket_name.end(), addr.sun_path);

  // Prepare data to send
  // Data being sent is "fd", the value of fd will be sent as auxiliary data
  // (control message)
  struct iovec io = {.iov_base = (void*)("fd"), .iov_len = 2};

  // Prepare control message data buffer and zero it out
  // NOLINTNEXTLINE(*array*)
  char cbuf[CMSG_SPACE(sizeof(int))];
  memset(cbuf, 0, sizeof(cbuf));

  // Create message header
  struct msghdr msg{
      // destination socket address and size of it
      // message content in msg_iov and number of such structs (1 in our case)
      // auxiliary data with the value of fd and size of it
      .msg_name = (void*)&addr,
      .msg_namelen = sizeof(struct sockaddr_un),
      .msg_iov = &io,
      .msg_iovlen = 1,
      .msg_control = cbuf,
      .msg_controllen = sizeof(cbuf)};

  // This points to the first control message header
  // With SCM_RIGHTS we let the kernel know that we are passing file
  // descriptors.
  auto cmsg = CMSG_FIRSTHDR(&msg);
  cmsg->cmsg_len = CMSG_LEN(sizeof(int));
  // Specify socket level message
  cmsg->cmsg_level = SOL_SOCKET;
  // SCM_RIGHTS is the type used to pass file descriptors
  cmsg->cmsg_type = SCM_RIGHTS;

  if (fd != -1) {
    std::copy(
        reinterpret_cast<const char*>(&fd),
        reinterpret_cast<const char*>(&fd) + sizeof(fd),
        reinterpret_cast<char*>(CMSG_DATA(cmsg)));
  } else {
    msg.msg_controllen = 0;
  }

  // Finally send the message
  TORCH_CHECK(
      sendmsg(socket_, &msg, 0) > 0,
      "Failed to send fd: ",
      c10::utils::str_error(errno));
}

int IpcChannel::recv_fd() {
  // Prepare buffer for regular message "fd"
  // NOLINTNEXTLINE(*array*)
  char buf[2];
  memset(&buf, 0, sizeof(buf));
  struct iovec io = {.iov_base = (void*)buf, .iov_len = sizeof(buf)};

  // Prepare buffer for control message and zero it out
  // NOLINTNEXTLINE(*array*)
  char cbuf[CMSG_SPACE(sizeof(int))];
  memset(cbuf, 0, sizeof(cbuf));

  // Define socket address to receive on: family AF_UNIX means unix domain
  // socket
  struct sockaddr_un addr = {.sun_family = AF_UNIX};
  std::copy(socket_name_.begin(), socket_name_.end(), addr.sun_path);

  // Prepare message header
  struct msghdr msg = {
      .msg_name = (void*)&addr,
      .msg_namelen = sizeof(struct sockaddr_un),
      .msg_iov = &io,
      .msg_iovlen = 1,
      .msg_control = cbuf,
      .msg_controllen = sizeof(cbuf)};

  // Receive message on socket_
  TORCH_CHECK(
      recvmsg(socket_, &msg, 0) > 0,
      "Failed to receive fd: ",
      c10::utils::str_error(errno));

  if (msg.msg_controllen == 0) {
    return -1;
  }

  // Extract control message and validate its content
  auto cmsg = CMSG_FIRSTHDR(&msg);
  TORCH_CHECK(cmsg != nullptr);
  TORCH_CHECK(cmsg->cmsg_len == CMSG_LEN(sizeof(int)));
  TORCH_CHECK(cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS);
  return *reinterpret_cast<int*>(CMSG_DATA(cmsg));
}

std::vector<int> IpcChannel::all_gather_fds(
    int rank,
    const std::vector<int>& pids,
    int fd) {
  int world_size = (int)pids.size();
  std::vector<int> fds(pids.size());
  fds[rank] = fd;

  int dst_rank = (rank + 1) % world_size;
  for (int step = 1; step < world_size; ++step) {
    int src_rank = (rank + world_size - step) % world_size;
    send_fd(pids[dst_rank], fd);
    fd = recv_fd();
    fds[src_rank] = fd;
  }
  return fds;
}

int IpcChannel::broadcast_fds(
    int rank,
    int src_rank,
    const std::vector<int>& pids,
    int fd) {
  int world_size = (int)pids.size();

  if (rank == src_rank) {
    for (int dst_rank = 0; dst_rank < world_size; ++dst_rank) {
      if (dst_rank == rank) {
        continue;
      }
      send_fd(pids[dst_rank], fd);
    }
    return fd;
  }
  return recv_fd();
}

std::string IpcChannel::get_socket_name(int pid) {
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

void map_block(
    void** ptr,
    c10d::symmetric_memory::HandleType handle,
    size_t size,
    int device_idx) {
#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
  auto driver_api = c10::cuda::DriverAPI::get();
  auto dev_ptr = reinterpret_cast<CUdeviceptr*>(ptr);
  // Allocate virtual address space
  C10_CUDA_DRIVER_CHECK(
      driver_api->cuMemAddressReserve_(dev_ptr, size, 0ULL, 0, 0ULL));
  // Map the physical memory to the virtual address
  C10_CUDA_DRIVER_CHECK(driver_api->cuMemMap_(*dev_ptr, size, 0, handle, 0ULL));

  // Set access permissions
  CUmemAccessDesc desc;
  desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  // NOLINTNEXTLINE(bugprone-signed-char-misuse)
  desc.location.id = device_idx;
  desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  C10_CUDA_DRIVER_CHECK(driver_api->cuMemSetAccess_(*dev_ptr, size, &desc, 1));
#elif defined(USE_ROCM)
  C10_HIP_CHECK(hipMemAddressReserve(ptr, size, 0ULL, 0, 0ULL));
  C10_HIP_CHECK(hipMemMap(
      *ptr,
      size,
      0,
      reinterpret_cast<hipMemGenericAllocationHandle_t>(handle),
      0ULL));
  C10_HIP_CHECK(hipMemMap(
      *ptr,
      size,
      0,
      reinterpret_cast<hipMemGenericAllocationHandle_t>(handle),
      0ULL));

  hipMemAccessDesc desc;
  desc.location.type = hipMemLocationTypeDevice;
  // NOLINTNEXTLINE(bugprone-signed-char-misuse)
  desc.location.id = static_cast<int>(device_idx);
  desc.flags = hipMemAccessFlagsProtReadWrite;
  C10_HIP_CHECK(hipMemSetAccess(*ptr, size, &desc, 1));
#else
  TORCH_CHECK(
      false, "CUDASymmetricMemory requires PYTORCH_C10_DRIVER_API_SUPPORTED");
#endif
}

} // namespace c10d::symmetric_memory
