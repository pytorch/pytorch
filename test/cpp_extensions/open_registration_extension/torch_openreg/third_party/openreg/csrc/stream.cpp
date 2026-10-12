#include <include/openreg.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <queue>
#include <set>
#include <sstream>
#include <thread>

#include <fcntl.h>
#include <semaphore.h>
#include <sys/mman.h>
#include <unistd.h>

static std::mutex g_mutex;
static std::once_flag g_flag;
static std::vector<std::set<orStream_t>> g_streams_per_device;

static void initialize_registries() {
  int device_count = 0;
  orGetDeviceCount(&device_count);
  g_streams_per_device.resize(device_count);
}

struct orEventImpl {
  std::mutex mtx;
  std::condition_variable cv;
  std::atomic<bool> completed{true};
  int device_index = -1;
  bool timing_enabled{false};
  std::chrono::high_resolution_clock::time_point completion_time;

  // Set once this event has been shared across processes via
  // orEventIpcGetHandle/orEventIpcOpenHandle. The semaphore lives in a POSIX
  // shared-memory segment, so synchronize() can block on it directly instead
  // of on `cv`, which cannot be signaled across process boundaries.
  sem_t* ipc_sem = nullptr;
  // Only the exporter owns the shared-memory segment's name and is
  // responsible for unlinking it.
  std::string ipc_shm_name;

  orEventImpl() = default;
  orEventImpl(const orEventImpl&) = delete;
  orEventImpl& operator=(const orEventImpl&) = delete;

  ~orEventImpl() {
    if (ipc_sem) {
      munmap(static_cast<void*>(ipc_sem), sizeof(sem_t));
    }
    if (!ipc_shm_name.empty()) {
      shm_unlink(ipc_shm_name.c_str());
    }
  }
};

struct orEvent {
  std::shared_ptr<orEventImpl> impl;
};

struct orStream {
  std::queue<std::function<void()>> tasks;
  std::mutex mtx;
  std::condition_variable cv;
  std::thread worker;
  std::atomic<bool> stop_flag{false};
  int device_index = -1;

  orStream() {
    worker = std::thread([this] {
      while (true) {
        std::function<void()> task;
        {
          std::unique_lock<std::mutex> lock(this->mtx);
          this->cv.wait(lock, [this] {
            return this->stop_flag.load() || !this->tasks.empty();
          });
          if (this->stop_flag.load() && this->tasks.empty()) {
            return;
          }
          task = std::move(this->tasks.front());
          this->tasks.pop();
        }
        task();
      }
    });
  }

  ~orStream() {
    {
      std::lock_guard<std::mutex> lock(mtx);
      stop_flag.store(true);
    }
    cv.notify_one();
    worker.join();
  }
};

orError_t openreg::addTaskToStream(
    orStream_t stream,
    std::function<void()> task) {
  if (!stream)
    return orErrorUnknown;

  {
    std::lock_guard<std::mutex> lock(stream->mtx);
    stream->tasks.push(std::move(task));
  }

  stream->cv.notify_one();
  return orSuccess;
}

orError_t orEventCreateWithFlags(orEvent_t* event, unsigned int flags) {
  if (!event)
    return orErrorUnknown;

  auto impl = std::make_shared<orEventImpl>();
  orGetDevice(&(impl->device_index));
  if (flags & orEventEnableTiming) {
    impl->timing_enabled = true;
  }

  *event = new orEvent{std::move(impl)};
  return orSuccess;
}

orError_t orEventCreate(orEvent_t* event) {
  return orEventCreateWithFlags(event, orEventDisableTiming);
}

orError_t orEventDestroy(orEvent_t event) {
  if (!event)
    return orErrorUnknown;

  delete event;
  return orSuccess;
}

orError_t orEventRecord(orEvent_t event, orStream_t stream) {
  if (!event || !stream)
    return orErrorUnknown;

  if (event->impl->device_index != stream->device_index)
    return orErrorUnknown;

  auto event_impl = event->impl;
  event_impl->completed.store(false);
  auto record_task = [event_impl]() {
    if (event_impl->timing_enabled) {
      event_impl->completion_time = std::chrono::high_resolution_clock::now();
    }

    {
      std::lock_guard<std::mutex> lock(event_impl->mtx);
      event_impl->completed.store(true);
    }

    event_impl->cv.notify_all();

    if (event_impl->ipc_sem) {
      sem_post(event_impl->ipc_sem);
    }
  };

  return openreg::addTaskToStream(stream, record_task);
}

orError_t orEventSynchronize(orEvent_t event) {
  if (!event)
    return orErrorUnknown;

  auto event_impl = event->impl;
  if (event_impl->ipc_sem) {
    // A condition_variable cannot be signaled across process boundaries, so
    // IPC events are synchronized through a shared POSIX semaphore instead.
    sem_wait(event_impl->ipc_sem);
    return orSuccess;
  }

  std::unique_lock<std::mutex> lock(event_impl->mtx);
  event_impl->cv.wait(lock, [&] { return event_impl->completed.load(); });

  return orSuccess;
}

orError_t orEventQuery(orEvent_t event) {
  if (!event)
    return orErrorUnknown;

  return event->impl->completed.load() ? orSuccess : orErrorNotReady;
}

orError_t orEventElapsedTime(float* ms, orEvent_t start, orEvent_t end) {
  if (!ms || !start || !end)
    return orErrorUnknown;

  auto start_impl = start->impl;
  auto end_impl = end->impl;

  if (start_impl->device_index != end_impl->device_index) {
    return orErrorUnknown;
  }

  if (!start_impl->timing_enabled || !end_impl->timing_enabled) {
    return orErrorUnknown;
  }

  if (!start_impl->completed.load() || !end_impl->completed.load()) {
    return orErrorUnknown;
  }

  auto duration = end_impl->completion_time - start_impl->completion_time;
  *ms = std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(
            duration)
            .count();

  return orSuccess;
}

orError_t orEventIpcGetHandle(orIpcEventHandle_t* handle, orEvent_t event) {
  if (!handle || !event)
    return orErrorUnknown;

  auto event_impl = event->impl;
  std::lock_guard<std::mutex> lock(event_impl->mtx);

  if (!event_impl->ipc_sem) {
    // Back this event with a named shared-memory semaphore so another
    // process can wait on it via orEventIpcOpenHandle.
    static std::atomic<uint64_t> ipc_event_counter{0};
    std::ostringstream name_stream;
    name_stream << "/or_event_" << getpid() << "_"
                << ipc_event_counter.fetch_add(1);
    std::string shm_name = name_stream.str();

    int fd = shm_open(shm_name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0)
      return orErrorUnknown;

    if (ftruncate(fd, sizeof(sem_t)) != 0) {
      close(fd);
      shm_unlink(shm_name.c_str());
      return orErrorUnknown;
    }

    void* addr = mmap(
        nullptr, sizeof(sem_t), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (addr == MAP_FAILED) {
      shm_unlink(shm_name.c_str());
      return orErrorUnknown;
    }

    auto* sem = static_cast<sem_t*>(addr);
    if (sem_init(sem, /*pshared=*/1, 0) != 0) {
      munmap(addr, sizeof(sem_t));
      shm_unlink(shm_name.c_str());
      return orErrorUnknown;
    }

    event_impl->ipc_sem = sem;
    event_impl->ipc_shm_name = shm_name;
  }

  std::memset(handle->reserved, 0, OR_IPC_HANDLE_SIZE);
  std::strncpy(
      handle->reserved, event_impl->ipc_shm_name.c_str(),
      OR_IPC_HANDLE_SIZE - 1);
  return orSuccess;
}

orError_t orEventIpcOpenHandle(orEvent_t* event, orIpcEventHandle_t handle) {
  if (!event)
    return orErrorUnknown;

  // Ensure the shared-memory name is null-terminated regardless of handle
  // contents received from another process.
  char shm_name[OR_IPC_HANDLE_SIZE + 1] = {};
  std::memcpy(shm_name, handle.reserved, OR_IPC_HANDLE_SIZE);

  int fd = shm_open(shm_name, O_RDWR, 0600);
  if (fd < 0)
    return orErrorUnknown;

  void* addr =
      mmap(nullptr, sizeof(sem_t), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (addr == MAP_FAILED)
    return orErrorUnknown;

  auto impl = std::make_shared<orEventImpl>();
  orGetDevice(&(impl->device_index));
  impl->ipc_sem = static_cast<sem_t*>(addr);
  // The importer maps the exporter's segment but does not own its name, so
  // it must not unlink it.

  *event = new orEvent{std::move(impl)};
  return orSuccess;
}

orError_t orStreamCreateWithPriority(
    orStream_t* stream,
    [[maybe_unused]] unsigned int flag,
    int priority) {
  if (!stream) {
    return orErrorUnknown;
  }

  int min_p, max_p;
  orDeviceGetStreamPriorityRange(&min_p, &max_p);
  if (priority < min_p || priority > max_p) {
    return orErrorUnknown;
  }

  int current_device = 0;
  orGetDevice(&current_device);

  orStream_t new_stream = nullptr;
  new_stream = new orStream();
  new_stream->device_index = current_device;

  {
    std::lock_guard<std::mutex> lock(g_mutex);
    std::call_once(g_flag, initialize_registries);
    g_streams_per_device[current_device].insert(new_stream);
  }

  *stream = new_stream;

  return orSuccess;
}

orError_t orStreamCreate(orStream_t* stream) {
  int min_p, max_p;
  orDeviceGetStreamPriorityRange(&min_p, &max_p);

  return orStreamCreateWithPriority(stream, 0, max_p);
}

orError_t orStreamGetPriority(
    [[maybe_unused]] orStream_t stream,
    int* priority) {
  // Since OpenReg has only one priority level, the following code
  // returns 0 directly for convenience.
  *priority = 0;

  return orSuccess;
}

orError_t orStreamDestroy(orStream_t stream) {
  if (!stream)
    return orErrorUnknown;

  {
    std::lock_guard<std::mutex> lock(g_mutex);

    int device_idx = stream->device_index;
    if (device_idx >= 0 && device_idx < g_streams_per_device.size()) {
      g_streams_per_device[device_idx].erase(stream);
    }
  }

  delete stream;
  return orSuccess;
}

orError_t orStreamQuery(orStream_t stream) {
  if (!stream) {
    return orErrorUnknown;
  }

  std::lock_guard<std::mutex> lock(stream->mtx);
  return stream->tasks.empty() ? orSuccess : orErrorNotReady;
}

orError_t orStreamSynchronize(orStream_t stream) {
  if (!stream)
    return orErrorUnknown;

  orEvent_t event;
  orEventCreate(&event);
  orEventRecord(event, stream);

  orError_t status = orEventSynchronize(event);
  orEventDestroy(event);

  return status;
}

orError_t orStreamWaitEvent(orStream_t stream, orEvent_t event, unsigned int) {
  if (!stream || !event)
    return orErrorUnknown;

  auto event_impl = event->impl;
  auto wait_task = [event_impl]() {
    std::unique_lock<std::mutex> lock(event_impl->mtx);
    event_impl->cv.wait(lock, [&] { return event_impl->completed.load(); });
  };

  return openreg::addTaskToStream(stream, wait_task);
}

orError_t orDeviceGetStreamPriorityRange(
    int* leastPriority,
    int* greatestPriority) {
  if (!leastPriority || !greatestPriority) {
    return orErrorUnknown;
  }

  // OpenReg priority levels are 0 and 1
  *leastPriority = 0;
  *greatestPriority = 1;
  return orSuccess;
}

orError_t orDeviceSynchronize(void) {
  int current_device = 0;
  orGetDevice(&current_device);

  std::vector<orStream_t> streams;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    std::call_once(g_flag, initialize_registries);

    auto& streams_on_device = g_streams_per_device[current_device];
    streams.assign(streams_on_device.begin(), streams_on_device.end());
  }

  for (orStream_t stream : streams) {
    orError_t status = orStreamSynchronize(stream);
    if (status != orSuccess) {
      return status;
    }
  }

  return orSuccess;
}
