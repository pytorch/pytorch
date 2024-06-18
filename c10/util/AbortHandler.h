#include <c10/macros/Macros.h>
#include <c10/util/Backtrace.h>
#include <c10/util/env.h>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <mutex>
#include <optional>

namespace c10 {
class AbortHandlerHelper {
 public:
  static AbortHandlerHelper& getInstance() {
#ifdef _WIN32
    thread_local
#endif // _WIN32
        static AbortHandlerHelper instance;
    return instance;
  }

  void set(std::terminate_handler handler) {
    std::lock_guard<std::mutex> lk(mutex);
    if (!inited) {
      prev = std::set_terminate(handler);
      curr = std::get_terminate();
      inited = true;
    }
  }

  std::terminate_handler getPrev() const {
    return prev;
  }

 private:
  std::terminate_handler prev = nullptr;
  std::terminate_handler curr = nullptr;
  bool inited = false;
  std::mutex mutex;
  AbortHandlerHelper() = default;
  ~AbortHandlerHelper() {
    // Only restore the handler if we are the current one
    if (inited && curr == std::get_terminate()) {
      std::set_terminate(prev);
    }
  }

 public:
  AbortHandlerHelper(AbortHandlerHelper const&) = delete;
  void operator=(AbortHandlerHelper const&) = delete;
};

namespace detail {
C10_ALWAYS_INLINE void terminate_handler() {
  std::cout << "Unhandled exception caught in c10/util/AbortHandler.h" << '\n';
  auto backtrace = get_backtrace();
  std::cout << backtrace << '\n' << std::flush;
  auto prev_handler = AbortHandlerHelper::getInstance().getPrev();
  if (prev_handler) {
    prev_handler();
  } else {
    std::abort();
  }
}
} // namespace detail

C10_ALWAYS_INLINE void set_terminate_handler() {
  bool use_custom_terminate = false;
  // On Windows it is enabled by default based on
  // https://github.com/pytorch/pytorch/pull/50320#issuecomment-763147062
#ifdef _WIN32
  use_custom_terminate = true;
#endif // _WIN32
  auto result = c10::utils::check_env("TORCH_CUSTOM_TERMINATE");
  if (result != std::nullopt) {
    use_custom_terminate = result.value();
  }
  if (use_custom_terminate) {
    AbortHandlerHelper::getInstance().set(detail::terminate_handler);
  }
}
} // namespace c10
