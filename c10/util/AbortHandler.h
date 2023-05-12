#include <c10/util/Backtrace.h>
#include <cstdlib>
#include <exception>
#include <iostream>

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
    if (!inited) {
      prev = std::set_terminate(handler);
      curr = std::get_terminate();
      inited = true;
    }
  }

  std::terminate_handler getPrev() {
    return prev;
  }

 private:
  std::terminate_handler prev = nullptr;
  std::terminate_handler curr = nullptr;
  bool inited = false;
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
void terminate_handler() {
  std::cout << "Unhandled exception caught in c10/util/AbortHandler.h"
            << std::endl;
  auto backtrace = get_backtrace();
  std::cout << backtrace << std::endl;
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
#ifdef _WIN32
  use_custom_terminate = true;
#endif // _WIN32
  const char* value_str = std::getenv("USE_CUSTOM_TERMINATE");
  std::string value{value_str != nullptr ? value_str : ""};
  if (!value.empty()) {
    use_custom_terminate = false;
    std::transform(
        value.begin(), value.end(), value.begin(), [](unsigned char c) {
          return toupper(c);
        });
    if (value == "1" || value == "ON") {
      use_custom_terminate = true;
    }
  }
  if (use_custom_terminate) {
    AbortHandlerHelper::getInstance().set(detail::terminate_handler);
  }
}
} // namespace c10
