
#include <torch/csrc/autograd/profiler_kineto.h>
#include <torch/torch.h>
#include <string>

using namespace torch::autograd::profiler;

void blueprint(const std::string& text) {
  printf("\33[94m%s\33[0m\n", text.c_str());
}

/**
 * We're emulating a C++ training engine calling into Python to allow Python
 * code controlling how profiling should be done.
 */
class ProfilerEventHandler
    : public std::enable_shared_from_this<ProfilerEventHandler> {
 public:
  static std::shared_ptr<ProfilerEventHandler> Handler;
  static void Register(const std::shared_ptr<ProfilerEventHandler>& handler) {
    Handler = handler;
  }

 public:
  virtual ~ProfilerEventHandler() {}
  virtual void onIterationStart(int) {}
  virtual void emulateTraining(int, int) {}
};
std::shared_ptr<ProfilerEventHandler> ProfilerEventHandler::Handler;

class ProfilerEventHandlerTrampoline : public ProfilerEventHandler {
 public:
  virtual void onIterationStart(int iteration) override {
    PYBIND11_OVERRIDE(void, ProfilerEventHandler, onIterationStart, iteration);
  }
  virtual void emulateTraining(int iteration, int thread_id) override {
    PYBIND11_OVERRIDE(
        void, ProfilerEventHandler, emulateTraining, iteration, thread_id);
  }
};

/**
 * This is the entry point for the C++ training engine.
 */
void start_threads(int thread_count, int iteration_count, bool attach) {
  blueprint("start_cpp_threads called");

  static std::atomic<int> barrier = 0;
  barrier = 0;
  thread_local bool enabled_in_main_thread = false;

  std::vector<std::thread> threads;
  for (int id = 0; id < thread_count; id++) {
    blueprint("starting thread " + std::to_string(id));
    threads.emplace_back([thread_count, iteration_count, id, attach]() {
      for (int iteration = 0; iteration < iteration_count; iteration++) {
        if (id == 0) {
          ProfilerEventHandler::Handler->onIterationStart(iteration);
        }

        // this barrier makes sure all child threads will be turned on
        // with profiling when main thread is enabled
        ++barrier;
        while (barrier % thread_count) {
          std::this_thread::yield();
        }

        if (id > 0 && attach) {
          bool enabled = isProfilerEnabledInMainThread();
          if (enabled != enabled_in_main_thread) {
            if (enabled) {
              enableProfilerInChildThread();
            } else {
              disableProfilerInChildThread();
            }
            enabled_in_main_thread = enabled;
          }
        }

        ProfilerEventHandler::Handler->emulateTraining(iteration, id);
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }
}

PYBIND11_MODULE(profiler_test_cpp_thread_lib, m) {
  py::class_<
      ProfilerEventHandler,
      ProfilerEventHandlerTrampoline,
      std::shared_ptr<ProfilerEventHandler>>(m, "ProfilerEventHandler")
      .def(py::init<>())
      .def_static("Register", &ProfilerEventHandler::Register)
      .def(
          "onIterationStart",
          &ProfilerEventHandler::onIterationStart,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "emulateTraining",
          &ProfilerEventHandler::emulateTraining,
          py::call_guard<py::gil_scoped_release>());

  m.def(
      "start_threads",
      &start_threads,
      py::call_guard<py::gil_scoped_release>());
};
