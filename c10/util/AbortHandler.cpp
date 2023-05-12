#include <c10/util/AbortHandler.h>

namespace c10 {
void set_terminate_handler() {
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
