#include <cstring>
#include <string>
#include <unordered_map>

#include <TH/TH.h>
#include <libshm/err.h>
#include <libshm/socket.h>
#include <libshm/libshm.h>

std::unordered_map<std::string, ClientSocket> managers;
std::string manager_executable_path;

AllocInfo get_alloc_info(const char* filename) {
  AllocInfo info = {0};
  info.pid = getpid();
  info.free = false;
  size_t len = strlen(filename);
  if (len >= sizeof(info.filename)) {
    throw std::runtime_error("THMapAllocatorContext_filename too long");
  }
  memcpy(info.filename, filename, len + 1);
  return info;
}

void start_manager() {
  int pipe_ends[2];
  SYSCHECK_ERR_RETURN_NEG1(pipe(pipe_ends));

  pid_t pid;
  SYSCHECK_ERR_RETURN_NEG1(pid = fork());
  if (!pid) {
    SYSCHECK_ERR_RETURN_NEG1(close(pipe_ends[0]));
    SYSCHECK_ERR_RETURN_NEG1(dup2(pipe_ends[1], 1)); // Replace stdout
    SYSCHECK_ERR_RETURN_NEG1(close(pipe_ends[1]));
    execl(manager_executable_path.c_str(), "torch_shm_manager", NULL);
    exit(1);
  }
  SYSCHECK_ERR_RETURN_NEG1(close(pipe_ends[1]));

  ssize_t bytes_read;
  char buffer[1000];
  std::string handle;
  for (;;) {
    SYSCHECK_ERR_RETURN_NEG1(bytes_read = read(pipe_ends[0], buffer, sizeof(buffer)));
    handle.append(buffer, bytes_read);
    if (bytes_read == 0 || handle[handle.length() - 1] == '\n') {
      break;
    }
  }
  SYSCHECK_ERR_RETURN_NEG1(close(pipe_ends[0]));
  if (handle.length() == 0) {
    std::string msg("error executing torch_shm_manager at \"");
    msg += manager_executable_path;
    msg += "\"";
    throw std::runtime_error(msg);
  }

  handle.pop_back(); // remove \n
  if (handle == "ERROR")
    throw std::exception();

  ClientSocket manager {handle};
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

void libshm_init(const char *manager_exec_path) {
  manager_executable_path = std::string(manager_exec_path);
}

THManagedMapAllocatorInit::THManagedMapAllocatorInit(const char* manager_handle, const char* filename)
  : manager_handle_(manager_handle ? manager_handle : "") {
  // TODO: unlock GIL when contacting the manager
  try {
    ClientSocket *socket;
    if (!manager_handle_.empty()) {
      socket = &get_manager_socket(manager_handle_);
    } else {
      if (managers.size() == 0) {
        start_manager();
      }
      const auto &manager = managers.begin();
      manager_handle_ = manager->first;
      socket = &manager->second;
    }
    AllocInfo info = get_alloc_info(filename);
    socket->register_allocation(info);
  } catch(std::exception &e) {
    THError(e.what());
  }
}

THManagedMapAllocator::THManagedMapAllocator(const char *manager_handle, const char *filename, int flags, ptrdiff_t size)
  : THManagedMapAllocatorInit(manager_handle, filename), THRefcountedMapAllocator(filename, flags, size) {}

void THManagedMapAllocator::close() {
  if (closed_) return;
  AllocInfo info = get_alloc_info(filename());
  info.free = true;
  ClientSocket &socket = get_manager_socket(manager_handle_);
  THRefcountedMapAllocator::close();
  socket.register_deallocation(info);
}

static void deleteTHManagedMapAllocator(void* ptr) {
  delete static_cast<THManagedMapAllocator*>(ptr);
}

at::DataPtr THManagedMapAllocator::makeDataPtr(const char* manager_handle, const char* filename, int flags, ptrdiff_t size) {
  auto* context = new THManagedMapAllocator(manager_handle, filename, flags, size);
  return {context->data(), context, &deleteTHManagedMapAllocator, at::DeviceType::CPU};
}

THManagedMapAllocator* THManagedMapAllocator::fromDataPtr(const at::DataPtr& dptr) {
  return dptr.cast_context<THManagedMapAllocator>(&deleteTHManagedMapAllocator);
}
