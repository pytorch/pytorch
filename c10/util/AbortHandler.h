#include <c10/util/Backtrace.h>
#include <cstdlib>
#include <iostream>
#include <exception>

namespace c10 {
namespace detail {
void terminate_handler() {
  std::cout << "Unhandled exception caught in c10/util/AbortHandler.h" << std::endl;
  auto backtrace = get_backtrace();
  std::cout << backtrace << std::endl;
  std::abort();
}
}

void set_terminate_handler() {
  std::set_terminate(detail::terminate_handler);
}
}
