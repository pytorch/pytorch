#pragma once
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <memory>
#include <mutex>

namespace c10 {
namespace impl {

// NOTE: [torch::Library and Python imports]
// We want to be able to define custom ops in Python and C++:
// This is useful for defining custom ops in Python and C++: a user may
// put things like meta kernels in Python but the backend kernels in C++.
//
// When someone is using a custom op from Python, we want to ensure that
// both the C++ shared library and the Python kernels are loaded to avoid
// partially-initialized custom ops.
// When someone is using a custom op from C++, we assume they just want
// to use the C++ kernels.
//
// The design we use to prevent partially-initialized custom ops in
// Python is that a C++ torch::Library object can import a python module
// by calling `request_pyimport`, which will import the module if Python
// is available. This ensures that when the C++ TORCH_LIBRARY block gets
// loaded, the Python module will also be imported.
//
// Implementation details:
// - Due to the libtorch-libtorch_python split, we have a PyImportsHandler
// indirection. PyImportsHandler has a method that imports a module in Python.
// - When `import torch` happens, we initialize the PyImportsHandler
// - If a user wants to load a shared library with custom ops that calls
// `request_pyimport`, they must do so AFTER `import torch` happens. If they
// do it before, we'll raise an error on `import torch`.
// - The PyImportsHandler is protected by a mutex: TORCH_LIBRARY
// static initialization may be multi-threaded.

// request_pyimport is a mechanism to import a module from C++ if Python is
// available.
//
// - If Python is available and `import torch` has occurred, then we
// import the module.
// - If Python is not available, then this is a no-op. If Python ever becomes
// available in the future, then `import torch` will raise an error.
//
// Args:
// - module (the name of the module to import)
// - context (we'll print this in error messages)
C10_API void request_pyimport(const char* module, const char* context);

C10_API std::mutex& kPyImportsHandlerMutex();

struct PyImportsInterface;

// Must be holding kPyImportsHandlerMutex
C10_API bool unsafe_has_pyimports_handler();

// Must be holding kPyImportsHandlerMutex
C10_API void unsafe_set_pyimports_handler(
    std::unique_ptr<PyImportsInterface> impl);

// This is a list of modules that `request_pyimport` did not import because
// Python was not available. If `import torch` ever occurs in the future,
// we'll raise an error, otherwise, if we're running in a C++-only environment,
// nothing will happen.
//
// Access to this list should be protected by kPyImportsHandlerMutex.
using IgnoredPyImports = std::vector<std::tuple<const char*, const char*>>;
C10_API IgnoredPyImports& unsafe_get_ignored_pyimports();

struct C10_API PyImportsInterface {
  virtual ~PyImportsInterface() = default;
  virtual void do_pyimport(const char* module, const char* context) const = 0;
};

} // namespace impl
} // namespace c10
