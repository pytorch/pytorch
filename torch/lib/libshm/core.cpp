#include <cstring>
#include <string>
#include <unordered_map>

#include <TH/TH.h>
#include "err.h"
#include "socket.h"
#include "libshm.h"

std::unordered_map<std::string, ClientSocket> managers;
std::string manager_executable_path;

void libshm_init(const char *manager_exec_path) {
  manager_executable_path = std::string(manager_exec_path);
}

libshm_context * libshm_context_new(const char *manager_handle, const char *filename, int flags) {
  libshm_context *ctx = new libshm_context();
  if (!manager_handle) {
    ctx->manager_handle = nullptr;
  } else {
    size_t handle_length = std::strlen(manager_handle);
    ctx->manager_handle = new char[handle_length+1];
    memcpy(ctx->manager_handle, manager_handle, handle_length+1);
  }
  ctx->th_context = THMapAllocatorContext_new(filename, flags);
  return ctx;
}

void libshm_context_free(libshm_context *ctx) {
  delete[] ctx->manager_handle;
  delete ctx;
}

void start_manager() {
  int pipe_ends[2];
  SYSCHECK(pipe(pipe_ends));

  pid_t pid;
  SYSCHECK(pid = fork());
  if (!pid) {
    close(pipe_ends[0]);
    dup2(pipe_ends[1], 1); // Replace stdout
    close(pipe_ends[1]);
    execl(manager_executable_path.c_str(), "torch_shm_manager", NULL);
    exit(1);
  }
  SYSCHECK(close(pipe_ends[1]));

  ssize_t bytes_read;
  char buffer[1000];
  std::string handle;
  for (;;) {
    SYSCHECK(bytes_read = read(pipe_ends[0], buffer, sizeof(buffer)));
    handle.append(buffer, bytes_read);
    if (bytes_read == 0 || handle[handle.length() - 1] == '\n') {
      break;
    }
  }
  SYSCHECK(close(pipe_ends[0]));
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

ClientSocket& get_manager_socket(char *manager_handle) {
  std::string str_handle(manager_handle);
  auto it = managers.find(str_handle);
  if (it == managers.end()) {
    auto socket = ClientSocket(str_handle);
    auto result = managers.emplace(std::move(str_handle), std::move(socket));
    return result.first->second;
  } else {
    return it->second;
  }
}

char * copy_handle(const std::string &handle) {
  char *new_handle = new char[handle.length()+1];
  memcpy(new_handle, handle.c_str(), handle.length() + 1);
  return new_handle;
}

AllocInfo get_alloc_info(libshm_context *ctx) {
  AllocInfo info = {0};
  info.pid = getpid();
  info.free = false;
  const char *filename = THMapAllocatorContext_filename(ctx->th_context);
  size_t len = strlen(filename);
  if (len >= sizeof(info.filename)) {
    throw std::runtime_error("THMapAllocatorContext_filename too long");
  }
  memcpy(info.filename, filename, len + 1);
  return info;
}

void * libshm_alloc(void *_ctx, ptrdiff_t size) {
  // TODO: unlock GIL when contacting the manager
  auto *ctx = (libshm_context*)_ctx;
  try {
    THMapAllocatorContext *th_context = ctx->th_context;
    ClientSocket *socket;
    if (ctx->manager_handle) {
      socket = &get_manager_socket(ctx->manager_handle);
    } else {
      if (managers.size() == 0)
          start_manager();
      const auto &manager = managers.begin();
      ctx->manager_handle = copy_handle(manager->first);
      socket = &manager->second;
    }
    AllocInfo info = get_alloc_info(ctx);
    socket->register_allocation(info);
  } catch(std::exception &e) {
    THError(e.what());
  }
  return THRefcountedMapAllocator.malloc(ctx->th_context, size);
}

void * libshm_realloc(void *_ctx, void *data, ptrdiff_t size) {
  THError("cannot realloc shared memory");
  return NULL;
}

void libshm_free(void *_ctx, void *data) {
  auto *ctx = (libshm_context*)_ctx;
  AllocInfo info = get_alloc_info(ctx);
  info.free = true;
  ClientSocket &socket = get_manager_socket(ctx->manager_handle);
  THRefcountedMapAllocator.free(ctx->th_context, data);
  libshm_context_free(ctx);
  socket.register_deallocation(info);
}

THAllocator THManagedSharedAllocator = {
  libshm_alloc,
  libshm_realloc,
  libshm_free,
};
