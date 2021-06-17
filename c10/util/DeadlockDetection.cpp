#include <c10/util/DeadlockDetection.h>

namespace c10 {
namespace impl {

namespace {
PythonGILHooks* python_gil_hooks = nullptr;
}

bool check_python_gil() {
  if (!python_gil_hooks) {
    return false;
  }
  return python_gil_hooks->check_python_gil();
}

void SetPythonGILHooks(PythonGILHooks* hooks) {
  TORCH_INTERNAL_ASSERT(!hooks || !python_gil_hooks);
  python_gil_hooks = hooks;
}

} // namespace impl
} // namespace c10
