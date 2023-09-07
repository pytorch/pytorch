#pragma once
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <memory>
#include <mutex>

namespace c10 {
namespace impl {

// If Python is available and `import torch` has occurred, then this
// imports the module. Otherwise, it adds the (module, source) to a list of
// "ignored imports"
//
// Args:
// - module (the name of the module to import)
// - context (we'll print this in error messages)
C10_API void request_pyimport(const char* module, const char* context);

using IgnoredPyImports = std::vector<std::tuple<const char*, const char*>>;

// Retrieve the list of ignored pyimports.
// Must be holding kPyImportsHandlerMutex
C10_API IgnoredPyImports& unsafe_get_ignored_pyimports();

struct C10_API PyImportsInterface {
  virtual ~PyImportsInterface() = default;
  virtual void do_pyimport(const char* module, const char* context) const = 0;
};

// Must be holding kPyImportsHandlerMutex
C10_API bool unsafe_has_pyimports_handler();

// Must be holding kPyImportsHandlerMutex
C10_API void unsafe_set_pyimports_handler(
    std::unique_ptr<PyImportsInterface> impl);

C10_API std::mutex& kPyImportsHandlerMutex();

} // namespace impl
} // namespace c10
