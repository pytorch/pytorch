#include <include/openreg.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <set>
#include <thread>

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
    stop_flag.store(true);
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
  };

  return openreg::addTaskToStream(stream, record_task);
}

orError_t orEventSynchronize(orEvent_t event) {
  if (!event)
    return orErrorUnknown;

  auto event_impl = event->impl;
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
