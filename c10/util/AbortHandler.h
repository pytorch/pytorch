#include <c10/util/Exception.h>
#include <cstdlib>
#include <exception>

namespace c10 {
namespace detail {
void terminate_handler() {
  TORCH_CHECK(false, "Unhandled exception caught");
  std::abort();
}
}

void set_terminate_handler() {
  std::set_terminate(detail::terminate_handler);
}
}
