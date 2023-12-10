#include <array>
#include <cstring>
#include <string>
#include <unordered_map>

#include <libshm/err.h>
#include <libshm/libshm.h>
#include <libshm/socket.h>

std::unordered_map<std::string, ClientSocket> managers;
std::string manager_executable_path;

AllocInfo get_alloc_info(const char* filename) {
  AllocInfo info = {0};
  info.pid = getpid();
  info.free = false;
  size_t len = strlen(filename);
  if (len >= sizeof(info.filename)) {
    throw std::runtime_error("MapAllocatorContext_filename too long");
  }
  memcpy(info.filename, filename, len + 1);
  return info;
}

void start_manager() {
  std::array<int, 2> pipe_ends;
  SYSCHECK_ERR_RETURN_NEG1(pipe(pipe_ends.data()));

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  pid_t pid;
  SYSCHECK_ERR_RETURN_NEG1(pid = fork());
  if (!pid) {
    SYSCHECK_ERR_RETURN_NEG1(close(pipe_ends[0]));
    SYSCHECK_ERR_RETURN_NEG1(dup2(pipe_ends[1], 1)); // Replace stdout
    SYSCHECK_ERR_RETURN_NEG1(close(pipe_ends[1]));
    execl(manager_executable_path.c_str(), "torch_shm_manager", NULL);

    std::string msg("ERROR: execl failed: ");
    msg += std::strerror(errno);
    msg += '\n';
    auto res = write(1, msg.c_str(), msg.size());
    (void)res;

    exit(1);
  }
  SYSCHECK_ERR_RETURN_NEG1(close(pipe_ends[1]));

  constexpr auto MAX_BUFFER_SIZE = 1000;
  std::array<char, MAX_BUFFER_SIZE> buffer;
  std::string handle;
  while (handle.empty() || handle.back() != '\n') {
    const auto bytes_read = read(pipe_ends[0], buffer.data(), buffer.size());
    SYSCHECK_ERR_RETURN_NEG1(bytes_read);
    if (bytes_read == 0) {
      break;
    }
    handle.append(buffer.data(), bytes_read);
  }
  SYSCHECK_ERR_RETURN_NEG1(close(pipe_ends[0]));
  if (handle.length() == 0) {
    std::string msg("no response from torch_shm_manager at \"");
    msg += manager_executable_path;
    msg += "\"";
    throw std::runtime_error(msg);
  }

  handle.pop_back(); // remove \n
  if (handle.rfind("ERROR: ", 0) == 0) {
    std::string msg("torch_shm_manager at \"");
    msg += manager_executable_path;
    msg += "\": ";
    msg += handle.substr(7); // remove "ERROR: "
    throw std::runtime_error(msg);
  }

  ClientSocket manager{handle};
  managers.emplace(std::move(handle), std::move(manager));
}

ClientSocket& get_manager_socket(const std::string& manager_handle) {
  auto it = managers.find(manager_handle);
  if (it == managers.end()) {
    auto socket = ClientSocket(manager_handle);
    auto result = managers.emplace(manager_handle, std::move(socket));
    return result.first->second;
  } else {
    return it->second;
  }
}

void libshm_init(const char* manager_exec_path) {
  manager_executable_path = std::string(manager_exec_path);
}

THManagedMapAllocatorInit::THManagedMapAllocatorInit(
    const char* manager_handle,
    const char* filename)
    : manager_handle_(manager_handle ? manager_handle : "") {
  // TODO: unlock GIL when contacting the manager
  try {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    ClientSocket* socket;
    if (!manager_handle_.empty()) {
      socket = &get_manager_socket(manager_handle_);
    } else {
      if (managers.empty()) {
        start_manager();
      }
      const auto& manager = managers.begin();
      manager_handle_ = manager->first;
      socket = &manager->second;
    }
    AllocInfo info = get_alloc_info(filename);
    socket->register_allocation(info);
  } catch (std::exception& e) {
    TORCH_CHECK(false, e.what());
  }
}

THManagedMapAllocator::THManagedMapAllocator(
    const char* manager_handle,
    const char* filename,
    int flags,
    size_t size)
    : THManagedMapAllocatorInit(manager_handle, filename),
      at::RefcountedMapAllocator(filename, flags, size) {}

void THManagedMapAllocator::close() {
  if (closed_)
    return;
  AllocInfo info = get_alloc_info(filename());
  info.free = true;
  ClientSocket& socket = get_manager_socket(manager_handle_);
  at::RefcountedMapAllocator::close();
  socket.register_deallocation(info);
}

static void deleteTHManagedMapAllocator(void* ptr) {
  delete static_cast<THManagedMapAllocator*>(ptr);
}

at::DataPtr THManagedMapAllocator::makeDataPtr(
    const char* manager_handle,
    const char* filename,
    int flags,
    size_t size) {
  auto* context =
      new THManagedMapAllocator(manager_handle, filename, flags, size);
  return {
      context->data(),
      context,
      &deleteTHManagedMapAllocator,
      at::DeviceType::CPU};
}

THManagedMapAllocator* THManagedMapAllocator::fromDataPtr(
    const at::DataPtr& dptr) {
  return dptr.cast_context<THManagedMapAllocator>(&deleteTHManagedMapAllocator);
}
