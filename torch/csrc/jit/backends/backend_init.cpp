#include <torch/csrc/jit/backends/backend_init.h>
#include <torch/csrc/jit/backends/test_backend.h>

namespace torch {
namespace jit {

void initJitBackendBindings(PyObject* module) {
  initTestBackendBindings(module);
}
} // namespace jit
} // namespace torch
