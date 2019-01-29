#include <ATen/core/dispatch/Dispatcher.h>

namespace c10 {
C10_EXPORT Dispatcher& Dispatcher::singleton() {
  static Dispatcher _singleton;
  return _singleton;
}
}
