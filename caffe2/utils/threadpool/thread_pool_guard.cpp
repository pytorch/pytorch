#include <caffe2/utils/threadpool/thread_pool_guard.h>

namespace caffe2 {

thread_local bool _NoPThreadPoolGuard_enabled = false;

bool _NoPThreadPoolGuard::is_enabled() {
  return _NoPThreadPoolGuard_enabled;
}

void _NoPThreadPoolGuard::set_enabled(bool enabled) {
  _NoPThreadPoolGuard_enabled = enabled;
}

} // namespace at
