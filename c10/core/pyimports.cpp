#include <c10/core/pyimports.h>

namespace c10 {
namespace impl {

namespace {

// All accesses must be protected by kPyImportsHandlerMutex
std::unique_ptr<PyImportsInterface> kPyImportsHandler = nullptr;

// All accesses must be protected by kPyImportsHandlerMutex
IgnoredPyImports& kIgnoredPyImports() {
  static IgnoredPyImports _data;
  return _data;
}

} // namespace

std::mutex& kPyImportsHandlerMutex() {
  static std::mutex _data;
  return _data;
};

void request_pyimport(const char* module, const char* context) {
  std::lock_guard<std::mutex> lock(kPyImportsHandlerMutex());
  if (unsafe_has_pyimports_handler()) {
    kPyImportsHandler->do_pyimport(module, context);
  } else {
    kIgnoredPyImports().emplace_back(module, context);
  }
}

IgnoredPyImports& unsafe_get_ignored_pyimports() {
  return kIgnoredPyImports();
}

void unsafe_set_pyimports_handler(std::unique_ptr<PyImportsInterface> impl) {
  kPyImportsHandler = std::move(impl);
}

bool unsafe_has_pyimports_handler() {
  return kPyImportsHandler != nullptr;
}

} // namespace impl
} // namespace c10
